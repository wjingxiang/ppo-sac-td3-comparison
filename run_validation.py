"""Launch the 5-seed re-validation in parallel: exactly one process per seed."""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE, "logs", "LunarLanderContinuous-v3")
os.makedirs(LOG_DIR, exist_ok=True)

procs = []
for seed in range(5):
    log_path = os.path.join(LOG_DIR, f"validate_seed{seed}.log")
    log = open(log_path, "w")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    p = subprocess.Popen(
        [sys.executable, os.path.join(BASE, "validate_tuned.py"), "--seed", str(seed)],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )
    procs.append(p)
    print(f"[launched] seed={seed} pid={p.pid}", flush=True)

print(f"[launcher] {len(procs)} processes running, waiting...", flush=True)
for p in procs:
    p.wait()
print("ALL_VALIDATION_DONE", flush=True)
