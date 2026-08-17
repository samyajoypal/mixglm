"""Dataset-agnostic entry point for the final leaderboard and bootstrap analysis."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.real_data.run_rand_final_inference import main


if __name__ == "__main__":
    main()
