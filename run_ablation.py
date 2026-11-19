"""Launch the ablation grid in parallel: variants x seeds, with N workers.

Usage:
    py run_ablation.py                                              # defaults below
    py run_ablation.py --variants full no_delay --seeds 0 1 --workers 2
    py run_ablation.py --workers 3                                  # 3 concurrent runs

Each run's log goes to logs/LunarLanderContinuous-v3/ablation_<variant>_seed<N>.runlog.txt
so you can tail it. Re-running skips nothing (it always retrains); results land in
ablation_<variant>_seed<N>_final.csv / _curve.csv.
"""
import argparse
import concurrent.futures
import itertools
import os
import subprocess
import sys

import config

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_ID = "LunarLanderContinuous-v3"
LOG_DIR = os.path.join(config.LOG_DIR, ENV_ID)


def run_one(variant, seed):
    tag = f"{variant}_seed{seed}"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"ablation_{tag}.runlog.txt")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with open(log_path, "w") as f:
        rc = subprocess.run(
            [sys.executable, os.path.join(BASE, "ablation.py"),
             "--variant", variant, "--seed", str(seed)],
            stdout=f, stderr=subprocess.STDOUT, env=env,
        ).returncode
    return tag, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*",
                    default=["full", "no_delay", "no_smoothing", "ddpg"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent runs; keep <=4 on a 16GB machine")
    args = ap.parse_args()

    runs = list(itertools.product(args.variants, args.seeds))
    print(f"ablation: {len(runs)} runs ({args.variants} x {args.seeds}), "
          f"{args.workers} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, v, s) for v, s in runs]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            tag, rc = fut.result()
            done += 1
            print(f"[{done}/{len(runs)}] {tag} rc={rc}", flush=True)
    print("ALL_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
