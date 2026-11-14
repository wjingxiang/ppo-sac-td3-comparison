"""Re-validate the best tuned TD3 config at full budget (proposal Section 3.5.2).

Loads the best params from TD3_tune_best.json and retrains TD3 with them at the
full 1M-step budget for one seed, using the same 100-episode final validation as
the baseline. Run 5 seeds (in parallel) for the full re-validation.

Usage:
    py validate_tuned.py --seed 0     # one seed; launch 0..4 in parallel
"""
import argparse
import csv
import json
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3

import config
from experiment import collect_returns

ENV_ID = "LunarLanderContinuous-v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    best_path = os.path.join(config.LOG_DIR, ENV_ID, "TD3_tune_best.json")
    with open(best_path) as f:
        params = json.load(f)["params"]

    total = config.ENV_CONFIG[ENV_ID]["total_timesteps"]
    log_dir = os.path.join(config.LOG_DIR, ENV_ID)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(ENV_ID)
    env.reset(seed=args.seed)
    model = TD3("MlpPolicy", env, seed=args.seed, verbose=0, **params)
    model.learn(total_timesteps=total)
    env.close()

    # 100-episode final validation (5 eval seeds x 20 episodes), same as baseline
    all_returns = []
    for s in config.FINAL_EVAL_SEEDS:
        e = gym.make(ENV_ID)
        e.reset(seed=s)
        all_returns.extend(collect_returns(model, e, config.N_FINAL_EPISODES_PER_SEED))
        e.close()

    mean = float(np.mean(all_returns))
    std = float(np.std(all_returns))

    out = os.path.join(log_dir, f"TD3_tuned_seed{args.seed}_final.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mean_return", "std_return", "n_episodes"])
        w.writerow([mean, std, len(all_returns)])

    print(f"[DONE] tuned TD3 seed={args.seed} final = {mean:.2f} ± {std:.2f} "
          f"(over {len(all_returns)} episodes)", flush=True)


if __name__ == "__main__":
    main()
