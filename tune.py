"""Bayesian hyperparameter optimization with Optuna (proposal Section 3.5).

Tune ONE algorithm on ONE environment. The search spaces are copied verbatim
from proposal Section 3.5.1.

Usage:
    py tune.py --algo SAC --env LunarLanderContinuous-v3 --trials 20
    py tune.py --algo PPO --env HalfCheetah-v4 --trials 20

Protocol (proposal Section 3.2 + 3.5.2):
  * per-trial budget: 100K (LunarLander) / 300K (HalfCheetah)
  * intermediate value reported to Optuna every 10% of the budget (5 episodes,
    fixed eval seed 0) -> enables MedianPruner early stopping
  * final objective = mean return over 10 episodes (deterministic policy, seed 0)
  * TPE sampler + MedianPruner(n_startup_trials=5), fixed sampler seed
"""
import argparse
import json
import os

# Limit torch/OpenBLAS to one thread. The tiny MLP gradients gain nothing
# from more threads, and the default (all cores) pegs the CPU for no benefit.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import gymnasium as gym
import optuna
from stable_baselines3 import PPO, SAC, TD3

import config
from experiment import evaluate  # reuse the deterministic-eval helper

ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3}

# Per-trial training budget (proposal Section 3.5.2: one tenth of full budget).
TUNE_TIMESTEPS = {"LunarLanderContinuous-v3": 100_000, "HalfCheetah-v4": 300_000}
TRAIN_SEED = 0      # fixed seed across all trials -> hyperparams are the only variable
EVAL_SEED = 0       # dedicated evaluation seed (proposal Section 3.2)
N_INTERMEDIATE_EPISODES = 5
N_FINAL_EPISODES = 10


def suggest_params(trial, algo):
    """Sample a hyperparameter set from the proposal's search space."""
    p = {}
    if algo == "PPO":
        p["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        p["n_steps"] = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        p["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
        p["gamma"] = trial.suggest_float("gamma", 0.90, 0.9999)
        p["gae_lambda"] = trial.suggest_float("gae_lambda", 0.90, 0.99)
        p["clip_range"] = trial.suggest_float("clip_range", 0.10, 0.40)
        p["ent_coef"] = trial.suggest_float("ent_coef", 0.0, 0.01)
    elif algo == "SAC":
        p["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        p["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])
        p["gamma"] = trial.suggest_float("gamma", 0.90, 0.9999)
        p["tau"] = trial.suggest_float("tau", 0.001, 0.02, log=True)
        # buffer_size fixed at 1e6, ent_coef kept "auto" (proposal Section 3.5.1)
    elif algo == "TD3":
        p["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        p["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])
        p["gamma"] = trial.suggest_float("gamma", 0.90, 0.9999)
        p["tau"] = trial.suggest_float("tau", 0.001, 0.02, log=True)
        p["policy_delay"] = trial.suggest_categorical("policy_delay", [1, 2, 3])
        p["target_policy_noise"] = trial.suggest_float("target_policy_noise", 0.10, 0.30)
    else:
        raise ValueError(algo)
    return p


def make_objective(algo, env_id, total_timesteps):
    eval_interval = max(int(total_timesteps * config.CHECKPOINT_RATIO), 1)

    def objective(trial):
        params = suggest_params(trial, algo)
        env = gym.make(env_id)
        env.reset(seed=TRAIN_SEED)
        model = ALGO_CLASSES[algo]("MlpPolicy", env, seed=TRAIN_SEED, verbose=0, **params)

        steps_done = 0
        while steps_done < total_timesteps:
            chunk = min(eval_interval, total_timesteps - steps_done)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            steps_done = model.num_timesteps
            mean_r, _ = evaluate(model, env_id, N_INTERMEDIATE_EPISODES, EVAL_SEED)
            trial.report(mean_r, steps_done)
            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()
        env.close()

        final_mean, _ = evaluate(model, env_id, N_FINAL_EPISODES, EVAL_SEED)
        return final_mean

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=list(ALGO_CLASSES.keys()))
    ap.add_argument("--env", required=True, choices=list(TUNE_TIMESTEPS.keys()))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None,
                    help="override per-trial timesteps (for smoke testing)")
    args = ap.parse_args()

    total = args.steps if args.steps else TUNE_TIMESTEPS[args.env]
    print(f"tuning {args.algo} on {args.env} | {args.trials} trials x {total} steps "
          f"(seed={args.seed})", flush=True)

    # Persistent SQLite storage: completed trials survive a crash/restart.
    out_dir = os.path.join(config.LOG_DIR, args.env)
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, f"{args.algo}_tuning.db")
    out_path = os.path.join(out_dir, f"{args.algo}_tune_best.json")

    study = optuna.create_study(
        study_name=f"{args.algo}_{args.env}_tuning",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        load_if_exists=True,
    )

    n_done = len(study.trials)
    n_remaining = max(0, args.trials - n_done)
    print(f"[resume] {n_done} trials in storage, {n_remaining} to run", flush=True)

    def save_best(study, trial):
        if study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
            with open(out_path, "w") as f:
                json.dump({"value": study.best_value, "params": study.best_params}, f, indent=2)

    if n_remaining > 0:
        study.optimize(make_objective(args.algo, args.env, total),
                       n_trials=n_remaining, callbacks=[save_best])

    print("\n===== best trial =====", flush=True)
    print("value:", study.best_value, flush=True)
    print("params:", json.dumps(study.best_params, indent=2), flush=True)

    importances = optuna.importance.get_param_importances(study)
    print("\n===== param importance (indicative) =====", flush=True)
    for name, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<22} {imp:.4f}", flush=True)

    save_best(study, None)
    print(f"\nsaved best params -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
