"""Global configuration for the DRL comparison project.

Edit the values here instead of touching the experiment scripts.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")       # all CSVs + plots go here
MODEL_DIR = os.path.join(BASE_DIR, "models")   # saved SB3 models go here

# ---------------------------------------------------------------------------
# Algorithms (all run with SB3 DEFAULT hyperparameters for the baseline)
# ---------------------------------------------------------------------------
ALGOS = ["PPO", "SAC", "TD3"]

# ---------------------------------------------------------------------------
# Environments + training budget
# ---------------------------------------------------------------------------
ENV_CONFIG = {
    "LunarLanderContinuous-v3": {
        "total_timesteps": 1_000_000,
        # "Solved" bar for LunarLander is a mean return of 200 (standard).
        "threshold": 200.0,
    },
    "HalfCheetah-v4": {
        "total_timesteps": 3_000_000,
        "threshold": None,
    },
}

# ---------------------------------------------------------------------------
# Seeds (Section 3.2 of the proposal)
# ---------------------------------------------------------------------------
TRAIN_SEEDS = [0, 1, 2, 3, 4]                    # 5 training seeds
EVAL_SEED = 0                                    # fixed seed for periodic evaluation
FINAL_EVAL_SEEDS = [1001, 1002, 1003, 1004, 1005]  # final validation seeds

# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
CHECKPOINT_RATIO = 0.10   # evaluate every 10% of the training budget
N_EVAL_EPISODES = 5       # episodes per periodic evaluation
N_FINAL_EPISODES_PER_SEED = 20  # episodes per final-validation seed (5 x 20 = 100)
