"""Run a grid of (algo, env, seed) runs with N parallel workers.

This is the parallel version of run_baseline.py: instead of running the
whole grid one-by-one, it launches up to `--workers` training processes at
the same time (useful on a many-core CPU).

Usage:
    py run_parallel.py --env LunarLanderContinuous-v3 --workers 6
    py run_parallel.py --env HalfCheetah-v4 --workers 4

Each run's stdout+stderr is written to:
    logs/<env>/<algo>_<env>_seed<N>.runlog.txt
so you can `tail` it to watch progress. A run whose final.csv already exists
is skipped, so the script is safe to re-run after a crash.
"""
import argparse
import concurrent.futures
import csv
import itertools
import os
import subprocess
import sys

import config

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment.py")


def is_done(algo, env_id, seed):
    """A run counts as done only if its curve actually reached the full budget.

    (The smoke test also writes a final.csv, but only after 5000 steps, so it
    must NOT be treated as a finished run.)
    """
    curve_path = os.path.join(config.LOG_DIR, env_id, f"{algo}_seed{seed}_curve.csv")
    final_path = os.path.join(config.LOG_DIR, env_id, f"{algo}_seed{seed}_final.csv")
    if not (os.path.exists(curve_path) and os.path.exists(final_path)):
        return False
    total = config.ENV_CONFIG[env_id]["total_timesteps"]
    with open(curve_path, newline="") as f:
        last_ts = max((int(row["timestep"]) for row in csv.DictReader(f)), default=0)
    return last_ts >= total - 1


def run_one(algo, env_id, seed, threads):
    tag = f"{algo}_{env_id}_seed{seed}"
    if is_done(algo, env_id, seed):
        print(f"skip {tag} (already done to full budget)", flush=True)
        return tag, "skipped"

    log_path = os.path.join(config.LOG_DIR, env_id, f"{tag}.runlog.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env_vars = {**os.environ, "OMP_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads)}
    with open(log_path, "w") as f:
        rc = subprocess.run(
            [sys.executable, SCRIPT, "--algo", algo, "--env", env_id, "--seed", str(seed)],
            stdout=f, stderr=subprocess.STDOUT, env=env_vars,
        ).returncode
    return tag, f"rc={rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--algos", nargs="*", default=config.ALGOS)
    ap.add_argument("--seeds", nargs="*", type=int, default=config.TRAIN_SEEDS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch/OpenBLAS threads per run; 1 is fastest for these "
                         "tiny MLPs (measured: 1 thread 112 steps/s vs 8 threads 105 steps/s)")
    args = ap.parse_args()

    runs = list(itertools.product(args.algos, [args.env], args.seeds))
    print(f"env={args.env}  total={len(runs)}  workers={args.workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, a, e, s, args.threads) for a, e, s in runs]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            tag, status = fut.result()
            done += 1
            print(f"[{done}/{len(runs)}] {tag} {status}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
