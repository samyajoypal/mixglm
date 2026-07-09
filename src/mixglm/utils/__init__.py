# src/mixglm/utils/__init__.py
from .numerics import (
    logsumexp,
    softmax_from_log,
    clip_exp,
    safe_log,
    normalize_simplex,
    weighted_mean,
    weighted_var,
    check_finite,
    ensure_1d,
    ensure_2d,
)

from .checks import (
    check_1d,
    check_2d,
    check_same_n,
    check_prob_simplex,
    check_responsibilities,
    check_intercept,
    check_no_degenerate_columns,
    check_support_against_families,
)

from .logging import Logger
from .parallel import ParallelConfig, parallel_map
from .repro import set_global_seed

__all__ = [
    # numerics
    "logsumexp",
    "softmax_from_log",
    "clip_exp",
    "safe_log",
    "normalize_simplex",
    "weighted_mean",
    "weighted_var",
    "check_finite",
    "ensure_1d",
    "ensure_2d",
    # checks
    "check_1d",
    "check_2d",
    "check_same_n",
    "check_prob_simplex",
    "check_responsibilities",
    "check_intercept",
    "check_no_degenerate_columns",
    "check_support_against_families",
    # logging/parallel
    "Logger",
    "ParallelConfig",
    "parallel_map",
    # repro
    "set_global_seed",
]
