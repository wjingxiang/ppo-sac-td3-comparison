"""Run the full baseline grid: 3 algorithms x 2 environments x 5 seeds = 30 runs.

Runs sequentially. For parallel execution (recommended on a multi-core CPU),
run individual `experiment.py` commands in separate terminals instead, e.g.:

    py experiment.py --algo PPO --env LunarLanderContinuous-v3 --seed 0
"""
import itertools
import os
import subprocess
import sys

import config

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment.py")


def main():
    runs = list(itertools.product(config.ALGOS, config.ENV_CONFIG.keys(),
                                  config.TRAIN_SEEDS))
    print(f"Total runs: {len(runs)}", flush=True)
    for i, (algo, env_id, seed) in enumerate(runs, 1):
        print(f"\n=== [{i}/{len(runs)}] {algo} on {env_id} seed={seed} ===", flush=True)
        subprocess.run([sys.executable, SCRIPT,
                        "--algo", algo, "--env", env_id, "--seed", str(seed)],
                       check=True)


if __name__ == "__main__":
    main()
