# src/mixglm/utils/numerics.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

Array = np.ndarray


def logsumexp(A: Array, axis: int = -1, keepdims: bool = False) -> Array:
    """
    Numerically stable log-sum-exp.

    Parameters
    ----------
    A : array
    axis : int
    keepdims : bool

    Returns
    -------
    array
        log(sum(exp(A))) computed stably
    """
    A = np.asarray(A, dtype=float)
    m = np.max(A, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def softmax_from_log(logw: Array, axis: int = -1) -> Array:
    """
    Stable softmax when inputs are already in log-scale.

    Returns exp(logw) / sum(exp(logw)) along axis.
    """
    logw = np.asarray(logw, dtype=float)
    m = np.max(logw, axis=axis, keepdims=True)
    ex = np.exp(logw - m)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def clip_exp(x: Array, lo: float = -700.0, hi: float = 700.0) -> Array:
    """
    Safe exp by clipping input to avoid overflow.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(np.clip(x, lo, hi))


def safe_log(x: Array, eps: float = 1e-15) -> Array:
    """
    Safe log with lower clipping.
    """
    x = np.asarray(x, dtype=float)
    return np.log(np.clip(x, eps, None))


def normalize_simplex(v: Array, min_val: float = 0.0, eps: float = 1e-15) -> Array:
    """
    Project a vector onto the probability simplex in a simple, safe way:
    - clip to at least min_val
    - renormalize to sum to 1
    """
    v = np.asarray(v, dtype=float).copy()
    v = np.maximum(v, min_val)
    s = float(np.sum(v))
    if s <= eps:
        # fallback: uniform
        v[:] = 1.0 / v.size
        return v
    return v / s


def weighted_mean(y: Array, w: Array, eps: float = 1e-15) -> float:
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    ws = float(np.sum(w))
    if ws <= eps:
        return float(np.mean(y))
    return float(np.sum(w * y) / ws)


def weighted_var(y: Array, w: Array, ddof: int = 0, eps: float = 1e-15) -> float:
    """
    Weighted variance (not unbiased unless you choose ddof appropriately).
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    ws = float(np.sum(w))
    if ws <= eps:
        return float(np.var(y, ddof=ddof))
    m = np.sum(w * y) / ws
    v = np.sum(w * (y - m) ** 2) / max(ws - ddof, eps)
    return float(v)


def check_finite(name: str, x: Array) -> None:
    x = np.asarray(x)
    if np.any(~np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def ensure_1d(name: str, x: Array) -> Array:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    return x


def ensure_2d(name: str, x: Array) -> Array:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    return x
