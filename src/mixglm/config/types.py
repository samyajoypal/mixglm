# src/mixglm/config/types.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EMConfig:
    max_iter: int = 200
    tol: float = 1e-6
    n_starts: int = 5
    inner_mstep_iter: int = 2
    min_pi: float = 1e-6
    verbose: bool = False
    init: str = "quantile"  # "quantile" | "random" | "kmeans_y"


@dataclass(frozen=True)
class SelectionConfig:
    criterion: str = "bic"   # "aic" | "bic" | "icl"
    K_max: int = 5
    beam_width: int = 10
    max_evals: int = 200


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 123
    n_jobs: int = 1
