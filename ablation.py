"""Ablation study: remove one TD3 component at a time to measure its contribution.

"Ablation" literally means surgical removal. Here we take the best algorithm (TD3),
cut out one of its three tricks, retrain, and see how much the score drops. The bigger
the drop, the more important that trick is.

Variants (all trained on LunarLanderContinuous-v3):
  full         -> default TD3 (reference, all three tricks on)
  no_delay     -> policy_delay=1        (ablate: delayed policy updates)
  no_smoothing -> target_policy_noise=0 (ablate: target policy smoothing)
  ddpg         -> DDPG (single critic)  (ablate: clipped double-Q; DDPG also drops the
                                          other two, so it is the "no tricks" baseline)

Usage:
    py ablation.py --variant no_delay --seed 0
    py ablation.py --variant full --seed 0 --steps 2000    # 2000-step smoke test
"""
import argparse
import csv
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3, DDPG

import config
from experiment import collect_returns, evaluate

ENV_ID = "LunarLanderContinuous-v3"

# Each variant is a dict of TD3 constructor overrides (empty = SB3 defaults).
TD3_VARIANTS = {
    "full":         {},
    "no_delay":     {"policy_delay": 1},
    "no_smoothing": {"target_policy_noise": 0.0},
}
# 300k steps: enough for full TD3 to converge (~240k) so the drop of ablated
# variants is visible, but ~3x cheaper than the 1M baseline budget.
ABLATION_STEPS = 300_000


def make_model(variant, env, seed):
    if variant == "ddpg":
        # DDPG is TD3's single-critic ancestor: no clipped double-Q, no delay, no smoothing.
        return DDPG("MlpPolicy", env, seed=seed, verbose=0)
    return TD3("MlpPolicy", env, seed=seed, verbose=0, **TD3_VARIANTS[variant])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(TD3_VARIANTS) + ["ddpg"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--steps", type=int, default=ABLATION_STEPS)
    args = ap.parse_args()

    total = args.steps
    eval_interval = max(int(total * config.CHECKPOINT_RATIO), 1)
    log_dir = os.path.join(config.LOG_DIR, ENV_ID)
    os.makedirs(log_dir, exist_ok=True)
    curve_path = os.path.join(log_dir, f"ablation_{args.variant}_seed{args.seed}_curve.csv")
    final_path = os.path.join(log_dir, f"ablation_{args.variant}_seed{args.seed}_final.csv")

    env = gym.make(ENV_ID)
    env.reset(seed=args.seed)
    model = make_model(args.variant, env, args.seed)

    curve_rows = []
    steps_done = 0
    while steps_done < total:
        model.learn(total_timesteps=min(eval_interval, total - steps_done),
                    reset_num_timesteps=False)
        steps_done = model.num_timesteps
        mean_r, std_r = evaluate(model, ENV_ID, config.N_EVAL_EPISODES, config.EVAL_SEED)
        curve_rows.append((steps_done, mean_r, std_r))
        print(f"[ablation {args.variant} seed={args.seed}] {steps_done}/{total} "
              f"eval={mean_r:.1f} ± {std_r:.1f}", flush=True)
        with open(curve_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestep", "mean_return", "std_return"])
            w.writerows(curve_rows)
    env.close()

    # Final validation: 100 episodes (5 eval seeds x 20), same protocol as baseline.
    all_returns = []
    for s in config.FINAL_EVAL_SEEDS:
        e = gym.make(ENV_ID)
        e.reset(seed=s)
        all_returns.extend(collect_returns(model, e, config.N_FINAL_EPISODES_PER_SEED))
        e.close()
    mean = float(np.mean(all_returns))
    std = float(np.std(all_returns))
    with open(final_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mean_return", "std_return", "n_episodes"])
        w.writerow([mean, std, len(all_returns)])
    print(f"[DONE ablation {args.variant} seed={args.seed}] final={mean:.2f} ± {std:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
