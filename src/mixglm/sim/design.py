# src/mixglm/sim/design.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class DesignConfig:
    n: int
    p: int
    intercept: bool = True
    x_dist: str = "normal"     # "normal" | "uniform"
    x_scale: float = 1.0
    rho: float = 0.0           # AR(1)-style correlation between columns (excluding intercept)
    center: bool = False
    standardize: bool = False


def _ar1_cov(p: int, rho: float) -> Array:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def make_design(cfg: DesignConfig, rng: np.random.Generator) -> Array:
    """
    Generate a design matrix X of shape (n, p).

    If intercept=True, X[:,0]=1 and the remaining p-1 columns follow the chosen distribution.
    If rho != 0, the non-intercept columns have AR(1) correlation with parameter rho.
    """
    n, p = int(cfg.n), int(cfg.p)
    if p < 1:
        raise ValueError("p must be >= 1")
    if not (-0.999 < cfg.rho < 0.999):
        raise ValueError("rho must be in (-0.999, 0.999)")

    if cfg.intercept:
        if p == 1:
            X = np.ones((n, 1), dtype=float)
            return X
        q = p - 1
    else:
        q = p

    # base draws
    if cfg.x_dist == "normal":
        Z = rng.normal(size=(n, q))
    elif cfg.x_dist == "uniform":
        Z = rng.uniform(low=-1.0, high=1.0, size=(n, q))
    else:
        raise ValueError("x_dist must be 'normal' or 'uniform'")

    # impose correlation on columns if requested
    if q > 1 and abs(cfg.rho) > 0:
        C = _ar1_cov(q, cfg.rho)
        L = np.linalg.cholesky(C)
        Z = Z @ L.T

    Z = cfg.x_scale * Z

    if cfg.center:
        Z = Z - Z.mean(axis=0, keepdims=True)

    if cfg.standardize:
        s = Z.std(axis=0, keepdims=True)
        s = np.where(s <= 0, 1.0, s)
        Z = Z / s

    if cfg.intercept:
        X = np.column_stack([np.ones(n, dtype=float), Z])
    else:
        X = Z

    return X
