"""Compute the four baseline metrics from saved logs (Section 3.3 of proposal).

For each environment, prints and saves a summary table per algorithm:
  1. final reward      : mean +/- std over the 5 training seeds
  2. convergence speed : proportion that reached the threshold + mean steps
                         (skipped until you set `threshold` in config.py)
  3. sample efficiency : mean return at the ~50% checkpoint
  4. training stability: mean over checkpoints of the across-seed std
"""
import csv
import os

import numpy as np

import config


def load_curve(path):
    ts, means = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts.append(int(row["timestep"]))
            means.append(float(row["mean_return"]))
    return np.array(ts), np.array(means)


def load_final(path):
    with open(path, newline="") as f:
        row = next(csv.DictReader(f))
    return float(row["mean_return"]), float(row["std_return"])


def main():
    for env_id, cfg in config.ENV_CONFIG.items():
        print(f"\n===== {env_id} =====")
        total = cfg["total_timesteps"]
        threshold = cfg.get("threshold")
        summary_rows = []

        for algo in config.ALGOS:
            finals, curves = [], []
            for seed in config.TRAIN_SEEDS:
                curve_path = os.path.join(config.LOG_DIR, env_id,
                                          f"{algo}_seed{seed}_curve.csv")
                final_path = os.path.join(config.LOG_DIR, env_id,
                                          f"{algo}_seed{seed}_final.csv")
                if not (os.path.exists(curve_path) and os.path.exists(final_path)):
                    continue
                finals.append(load_final(final_path)[0])
                curves.append(load_curve(curve_path))

            if not finals:
                print(f"  [{algo}] no data yet (run experiment.py first)")
                continue

            # Metric 1: final reward
            final_mean = float(np.mean(finals))
            final_std = float(np.std(finals))

            # Metric 3: sample efficiency at ~50% checkpoint (mean over seeds)
            ts = curves[0][0]
            mid_idx = int(np.argmin(np.abs(ts - total * 0.5)))
            sample_eff = float(np.mean([c[1][mid_idx] for c in curves]))

            # Metric 4: training stability (mean across-seed std per checkpoint)
            per_ckpt = np.array([c[1] for c in curves])       # (n_seeds, n_ckpt)
            stability = float(per_ckpt.std(axis=0).mean())

            # Metric 2: convergence speed
            if threshold is None:
                conv_str = "threshold not set (run pilot first)"
            else:
                hit_steps = []
                for c in curves:
                    idx = np.argmax(c[1] >= threshold) if (c[1] >= threshold).any() else None
                    if idx is not None:
                        hit_steps.append(int(c[0][idx]))
                prop = len(hit_steps) / len(curves)
                conv_str = (f"{prop*100:.0f}% converged, "
                            f"mean {np.mean(hit_steps):.0f} steps"
                            if hit_steps else "0% converged")

            print(f"\n{algo}:")
            print(f"  final reward       = {final_mean:.2f} ± {final_std:.2f}")
            print(f"  sample efficiency = {sample_eff:.2f} (at 50% budget)")
            print(f"  training stability= {stability:.2f} (avg std across seeds)")
            print(f"  convergence       = {conv_str}")

            summary_rows.append([algo, final_mean, final_std, sample_eff,
                                 stability, conv_str])

        # Save summary table (only if this env has any data)
        if not summary_rows:
            continue
        out_path = os.path.join(config.LOG_DIR, env_id, "summary.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["algo", "final_reward_mean", "final_reward_std",
                        "sample_efficiency_50pct", "training_stability",
                        "convergence"])
            w.writerows(summary_rows)
        print(f"\n  -> summary saved to {out_path}")


if __name__ == "__main__":
    main()
