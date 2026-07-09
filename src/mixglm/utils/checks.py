# src/mixglm/utils/checks.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np

Array = np.ndarray


def check_1d(y: Array, *, name: str = "y", finite: bool = True) -> Array:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of shape (n,).")
    if finite and np.any(~np.isfinite(y)):
        raise ValueError(f"{name} contains non-finite values.")
    return y


def check_2d(X: Array, *, name: str = "X", finite: bool = True) -> Array:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n, p).")
    if finite and np.any(~np.isfinite(X)):
        raise ValueError(f"{name} contains non-finite values.")
    return X


def check_same_n(y: Array, X: Array) -> Tuple[Array, Array]:
    y = check_1d(y, name="y")
    X = check_2d(X, name="X")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of rows.")
    return y, X


def check_prob_simplex(pi: Array, *, name: str = "pi", tol: float = 1e-8) -> Array:
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if np.any(pi < -tol):
        raise ValueError(f"{name} contains negative values.")
    s = float(np.sum(pi))
    if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1 (got {s}).")
    return pi


def check_responsibilities(tau: Array, *, name: str = "tau", tol: float = 1e-8) -> Array:
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 2:
        raise ValueError(f"{name} must be 2D array of shape (n, K).")
    if np.any(tau < -tol):
        raise ValueError(f"{name} has negative entries.")
    row_sums = np.sum(tau, axis=1)
    if np.any(~np.isfinite(row_sums)) or np.any(np.abs(row_sums - 1.0) > 1e-6):
        raise ValueError(f"Rows of {name} must sum to 1.")
    return tau


def check_intercept(X: Array, *, atol: float = 1e-12, allow_missing: bool = True) -> None:
    """
    Check whether X includes an intercept column of ones.
    If allow_missing=True, this is advisory and does not raise by default.
    """
    X = check_2d(X, name="X")
    has_intercept = np.any(np.all(np.abs(X - 1.0) <= atol, axis=0))
    if (not has_intercept) and (not allow_missing):
        raise ValueError("X does not appear to contain an intercept column of ones.")


def check_no_degenerate_columns(X: Array, *, tol: float = 1e-12) -> None:
    """
    Warn/raise on near-constant columns (can harm optimization).
    Here we raise to keep it strict; you can relax later.
    """
    X = check_2d(X, name="X")
    col_var = np.var(X, axis=0)
    if np.any(col_var <= tol):
        idx = np.where(col_var <= tol)[0].tolist()
        raise ValueError(f"X has near-constant columns at indices: {idx}")


def check_support_against_families(y: Array, families: Sequence, *, name: str = "y") -> None:
    """
    Validate y against each family's support.
    Families are expected to have `.support.validate_y(y)`.
    """
    y = check_1d(y, name=name)
    for fam in families:
        fam.support.validate_y(y)
