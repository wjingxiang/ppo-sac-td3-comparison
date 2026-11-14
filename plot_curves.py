"""Plot learning curves (mean +/- std across seeds) per environment."""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

import config


def load_curve(path):
    ts, means = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts.append(int(row["timestep"]))
            means.append(float(row["mean_return"]))
    return np.array(ts), np.array(means)


def main():
    for env_id in config.ENV_CONFIG:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for algo in config.ALGOS:
            curves = []
            for seed in config.TRAIN_SEEDS:
                p = os.path.join(config.LOG_DIR, env_id,
                                 f"{algo}_seed{seed}_curve.csv")
                if os.path.exists(p):
                    curves.append(load_curve(p))
            if not curves:
                continue
            ts = curves[0][0]
            means = np.array([c[1] for c in curves])
            ax.plot(ts, means.mean(axis=0), label=algo)
            ax.fill_between(ts, means.mean(axis=0) - means.std(axis=0),
                            means.mean(axis=0) + means.std(axis=0), alpha=0.2)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean return")
        ax.set_title(env_id)
        ax.legend()
        out = os.path.join(config.LOG_DIR, env_id, "learning_curves.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
