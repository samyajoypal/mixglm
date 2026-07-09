# src/mixglm/optim/scipy_opt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Any
import numpy as np

Array = np.ndarray


@dataclass
class SciPyOptResult:
    x: Array
    fun: float
    n_iter: int
    success: bool
    message: str


def scipy_minimize_box(
    *,
    fun: Callable[[Array], float],
    x0: Array,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    max_iter: int = 200,
    method: str = "L-BFGS-B",
) -> SciPyOptResult:
    """
    Thin wrapper around scipy.optimize.minimize for box-constrained problems.
    Falls back to returning x0 unchanged if SciPy is unavailable.

    Parameters
    ----------
    fun : objective function
    x0 : initial point
    bounds : list of (lo, hi) bounds, length p
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array.")
    if np.any(~np.isfinite(x0)):
        raise ValueError("x0 contains non-finite values.")

    try:
        from scipy.optimize import minimize
    except Exception as e:
        val = float(fun(x0))
        return SciPyOptResult(x=x0.copy(), fun=val, n_iter=0, success=False, message=f"SciPy not available: {e}")

    res = minimize(fun, x0=x0, method=method, bounds=bounds, options={"maxiter": int(max_iter)})
    xhat = np.asarray(res.x, dtype=float)
    return SciPyOptResult(
        x=xhat,
        fun=float(res.fun),
        n_iter=int(getattr(res, "nit", 0)),
        success=bool(res.success),
        message=str(res.message),
    )
