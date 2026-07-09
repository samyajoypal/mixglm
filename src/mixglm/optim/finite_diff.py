# src/mixglm/optim/finite_diff.py
from __future__ import annotations

from typing import Callable
import numpy as np

Array = np.ndarray


def finite_diff_grad(f: Callable[[Array], float], x: Array, eps: float = 1e-6) -> Array:
    """
    Central finite-difference gradient for a scalar function f(x).

    Parameters
    ----------
    f : callable
        Scalar function of a vector argument.
    x : array (p,)
        Point to evaluate the gradient at.
    eps : float
        Base step size. Actual step uses: h_j = eps * (1 + |x_j|).

    Returns
    -------
    grad : array (p,)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite.")
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values.")

    g = np.zeros_like(x)
    for j in range(x.size):
        h = eps * (1.0 + abs(float(x[j])))
        x1 = x.copy()
        x2 = x.copy()
        x1[j] += h
        x2[j] -= h
        f1 = float(f(x1))
        f2 = float(f(x2))
        g[j] = (f1 - f2) / (2.0 * h)
    return g
