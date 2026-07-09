# src/mixglm/utils/repro.py
from __future__ import annotations

import os
import random
import numpy as np

Array = np.ndarray


def set_global_seed(seed: int) -> np.random.Generator:
    """
    Best-effort reproducibility across Python + NumPy.

    Returns a NumPy Generator which you should pass around explicitly
    in simulations/experiments.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    return np.random.default_rng(seed)
