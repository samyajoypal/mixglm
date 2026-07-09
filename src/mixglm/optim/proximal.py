# src/mixglm/optim/proximal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

from mixglm.optim.finite_diff import finite_diff_grad

Array = np.ndarray


@dataclass
class ProxGradResult:
    x: Array
    n_iter: int
    converged: bool
    obj: float
    step: float


def prox_grad(
    *,
    x0: Array,
    smooth_obj: Callable[[Array], float],
    penalty_value: Callable[[Array], float],
    prox: Callable[[Array, float], Array],
    grad: Optional[Callable[[Array], Array]] = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    step0: float = 1e-2,
    backtrack_max: int = 25,
    fd_eps: float = 1e-6,
) -> ProxGradResult:
    """
    Proximal gradient descent with simple backtracking line search for composite objective:
        minimize F(x) = f(x) + P(x)
    where:
        f: smooth (we can provide grad or estimate via finite differences)
        P: possibly non-smooth, handled via proximal operator.

    Parameters
    ----------
    x0 : starting point (p,)
    smooth_obj : f(x)
    penalty_value : P(x)
    prox : proximal operator prox_{step*P}
    grad : optional gradient of f(x). If None, uses finite differences.
    """
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("x0 must be 1D.")
    if np.any(~np.isfinite(x)):
        raise ValueError("x0 contains non-finite values.")

    step = float(step0)
    if step <= 0:
        raise ValueError("step0 must be positive.")

    def F(z: Array) -> float:
        return float(smooth_obj(z) + penalty_value(z))

    obj_old = F(x)
    converged = False

    for it in range(1, max_iter + 1):
        g = grad(x) if grad is not None else finite_diff_grad(smooth_obj, x, eps=fd_eps)

        x_try = prox(x - step * g, step)
        obj_try = F(x_try)

        bt = 0
        while obj_try > obj_old and bt < backtrack_max:
            step *= 0.5
            x_try = prox(x - step * g, step)
            obj_try = F(x_try)
            bt += 1

        rel = abs(obj_old - obj_try) / (1.0 + abs(obj_old))
        x = x_try
        obj_old = obj_try

        if rel < tol:
            converged = True
            return ProxGradResult(x=x, n_iter=it, converged=True, obj=obj_old, step=step)

    return ProxGradResult(x=x, n_iter=max_iter, converged=converged, obj=obj_old, step=step)
