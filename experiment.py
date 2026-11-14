"""Train one algorithm on one environment for one seed (SB3 default hyperparams).

This is the core "baseline" runner. It:
  1. trains PPO / SAC / TD3 with Stable-Baselines3 defaults,
  2. evaluates every 10% of the budget (fixed eval seed) and logs a reward curve,
  3. does a final validation over 100 episodes (5 seeds x 20) and logs mean +/- std,
  4. saves the trained model.

Usage (from this directory):
    py experiment.py --algo PPO --env LunarLanderContinuous-v2 --seed 0
    py experiment.py --algo SAC --env HalfCheetah-v4 --seed 0
    py experiment.py --algo TD3 --env LunarLanderContinuous-v2 --seed 0 --smoke   # 5000-step sanity check
"""
import argparse
import csv
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3

import config

ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


def collect_returns(model, env, n_episodes):
    """Run `n_episodes` on an already-seeded env with a deterministic policy.

    Returns the list of episode returns. Deterministic=True matches the
    proposal: TD3 uses its greedy policy, SAC uses the mean of the action
    distribution, PPO uses its deterministic mode.
    """
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()  # continues the RNG stream seeded earlier
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    return returns


def evaluate(model, env_id, n_episodes, seed):
    """Evaluate over `n_episodes` using a fixed seed. Returns (mean, std)."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    returns = collect_returns(model, env, n_episodes)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=list(ALGO_CLASSES.keys()))
    parser.add_argument("--env", required=True, choices=list(config.ENV_CONFIG.keys()))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny 5000-step run to sanity-check the setup")
    args = parser.parse_args()

    env_id = args.env
    total_timesteps = 5000 if args.smoke else config.ENV_CONFIG[env_id]["total_timesteps"]
    eval_interval = max(int(total_timesteps * config.CHECKPOINT_RATIO), 1)

    log_dir = os.path.join(config.LOG_DIR, env_id)
    model_dir = os.path.join(config.MODEL_DIR, env_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    curve_path = os.path.join(log_dir, f"{args.algo}_seed{args.seed}_curve.csv")
    final_path = os.path.join(log_dir, f"{args.algo}_seed{args.seed}_final.csv")
    model_path = os.path.join(model_dir, f"{args.algo}_{env_id}_seed{args.seed}")

    def save_checkpoint(curve_rows, model):
        """Persist the learning curve + model so a kill doesn't wipe progress."""
        with open(curve_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestep", "mean_return", "std_return"])
            w.writerows(curve_rows)
        model.save(model_path)

    # --- Resume from a checkpoint if one exists, else start fresh ----------
    env = gym.make(env_id)
    env.reset(seed=args.seed)
    curve_rows = []
    steps_done = 0
    if os.path.exists(model_path + ".zip") and os.path.exists(curve_path):
        with open(curve_path, newline="") as f:
            for row in csv.DictReader(f):
                curve_rows.append((int(row["timestep"]), float(row["mean_return"]),
                                   float(row["std_return"])))
        steps_done = curve_rows[-1][0] if curve_rows else 0
    if steps_done >= eval_interval:
        # Genuine checkpoint (>= one eval interval) -> resume from it.
        model = ALGO_CLASSES[args.algo].load(model_path, env=env)
        print(f"[resume] {args.algo} {env_id} seed={args.seed} at step={steps_done}", flush=True)
    else:
        # Fresh start (or a smoke-test leftover too short to trust).
        model = ALGO_CLASSES[args.algo]("MlpPolicy", env, seed=args.seed, verbose=0)
        curve_rows = []
        steps_done = 0

    # --- Train (with periodic evaluation + checkpointing) ------------------
    while steps_done < total_timesteps:
        model.learn(total_timesteps=min(eval_interval, total_timesteps - steps_done),
                    reset_num_timesteps=False)
        steps_done = model.num_timesteps
        mean_r, std_r = evaluate(model, env_id, config.N_EVAL_EPISODES, config.EVAL_SEED)
        curve_rows.append((steps_done, mean_r, std_r))
        print(f"[{args.algo} {env_id} seed={args.seed}] step={steps_done}/{total_timesteps} "
              f"eval_return={mean_r:.2f} ± {std_r:.2f}", flush=True)
        save_checkpoint(curve_rows, model)
    env.close()

    # --- Final validation: 5 eval seeds x 20 episodes = 100 episodes ------
    all_returns = []
    for s in config.FINAL_EVAL_SEEDS:
        eval_env = gym.make(env_id)
        eval_env.reset(seed=s)
        all_returns.extend(collect_returns(model, eval_env,
                                           config.N_FINAL_EPISODES_PER_SEED))
        eval_env.close()

    final_mean = float(np.mean(all_returns))
    final_std = float(np.std(all_returns))
    with open(final_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mean_return", "std_return", "n_episodes"])
        w.writerow([final_mean, final_std, len(all_returns)])

    model.save(model_path)
    print(f"[DONE] {args.algo} {env_id} seed={args.seed} "
          f"final reward = {final_mean:.2f} ± {final_std:.2f} "
          f"(over {len(all_returns)} episodes)", flush=True)


if __name__ == "__main__":
    main()
