# src/mixglm/em/mstep.py
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

from mixglm.model.mixture_glm import ComponentSpec
from mixglm.optim.proximal import prox_grad
from mixglm.optim.scipy_opt import scipy_minimize_box

Array = np.ndarray


# def optimize_beta(
    # *,
    # X: Array,
    # y: Array,
    # tau_k: Array,
    # comp: ComponentSpec,
    # beta0: Array,
    # extra_k: Dict[str, Any],
    # max_iter: int = 200,
    # tol: float = 1e-6,
    # step0: float = 1e-2,
    # fd_eps: float = 1e-6,
# ) -> Array:
    # """
    # Optimize beta_k via proximal gradient:
      # minimize f(beta) + P(beta)
    # where f(beta) is the weighted component negative log-likelihood.
    # """
    # penalty = comp.penalty
    # family = comp.family
    # link = comp.link

    # def smooth_obj(b: Array) -> float:
        # mu = link.inverse(X @ b)
        # return family.component_nll(y=y, mu=mu, extra=extra_k, weights=tau_k)

    # res = prox_grad(
        # x0=np.asarray(beta0, dtype=float),
        # smooth_obj=smooth_obj,
        # penalty_value=penalty.value,
        # prox=penalty.prox,
        # grad=None,  # finite-diff inside prox_grad for now
        # max_iter=max_iter,
        # tol=tol,
        # step0=step0,
        # fd_eps=fd_eps,
    # )
    # return res.x

def optimize_beta(
    *,
    X: Array,
    y: Array,
    tau_k: Array,
    comp: ComponentSpec,
    beta0: Array,
    extra_k: Dict[str, Any],
    offset: Array | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    step0: float = 1e-2,
    fd_eps: float = 1e-6,
) -> Array:
    """
    Optimize beta_k via proximal gradient:
      minimize f(beta) + P(beta)

    IMPORTANT: We do NOT penalize the intercept (beta[0]).
    """
    penalty = comp.penalty
    family = comp.family
    link = comp.link
    offset_use = np.zeros(np.asarray(y).shape[0], dtype=float) if offset is None else np.asarray(offset, dtype=float)
    coef_mask = None
    if comp.coef_mask is not None:
        coef_mask = np.asarray(comp.coef_mask, dtype=bool)
        if coef_mask.shape != np.asarray(beta0).shape:
            raise ValueError(
                f"coef_mask must have shape {np.asarray(beta0).shape}; got {coef_mask.shape}."
            )

    def smooth_obj(b: Array) -> float:
        mu = link.inverse(X @ b + offset_use)
        return family.component_nll(y=y, mu=mu, extra=extra_k, weights=tau_k)

    # ---- penalty wrapper: exclude intercept ----
    def penalty_value_no_intercept(b: Array) -> float:
        b = np.asarray(b, dtype=float)
        if b.size <= 1:
            return 0.0
        return float(penalty.value(b[1:]))

    def prox_no_intercept(b: Array, step: float) -> Array:
        b = np.asarray(b, dtype=float).copy()
        if b.size > 1:
            b[1:] = penalty.prox(b[1:], step)
        if coef_mask is not None:
            b[~coef_mask] = 0.0
        return b

    beta_start = np.asarray(beta0, dtype=float).copy()
    if coef_mask is not None:
        beta_start[~coef_mask] = 0.0
    res = prox_grad(
        x0=beta_start,
        smooth_obj=smooth_obj,
        penalty_value=penalty_value_no_intercept,
        prox=prox_no_intercept,
        grad=None,  # finite-diff inside prox_grad for now
        max_iter=max_iter,
        tol=tol,
        step0=step0,
        fd_eps=fd_eps,
    )
    return res.x


def optimize_extra(
    *,
    X: Array,
    y: Array,
    tau_k: Array,
    comp: ComponentSpec,
    beta_k: Array,
    extra0: Dict[str, Any],
    offset: Array | None = None,
    max_iter: int = 200,
) -> Dict[str, Any]:
    """
    Optimize nuisance parameters for a component using SciPy if available.
    If the family has no nuisance params, returns {}.
    """
    family = comp.family
    link = comp.link
    names = family.extra_param_names
    offset_use = np.zeros(np.asarray(y).shape[0], dtype=float) if offset is None else np.asarray(offset, dtype=float)

    if len(names) == 0:
        return {}

    extra_t0 = family.transform_extra(extra0)
    x0 = np.array([float(extra_t0[n]) for n in names], dtype=float)

    bounds_dict = family.bounds_extra()
    bounds = [bounds_dict.get(n, (None, None)) for n in names]

    # def obj(xvec: Array) -> float:
        # extra_t = {n: float(xvec[j]) for j, n in enumerate(names)}
        # extra = family.inverse_transform_extra(extra_t)
        # family.validate_extra(extra)
        # mu = link.inverse(X @ beta_k)
        # return family.component_nll(y=y, mu=mu, extra=extra, weights=tau_k)
    def obj(xvec: Array) -> float:
        try:
            extra_t = {n: float(xvec[j]) for j, n in enumerate(names)}
            extra = family.inverse_transform_extra(extra_t)

            # If invalid, punish instead of raising
            family.validate_extra(extra)

            mu = link.inverse(X @ beta_k + offset_use)
            val = family.component_nll(y=y, mu=mu, extra=extra, weights=tau_k)

            # If numeric issues, punish
            if not np.isfinite(val):
                return 1e300
            return float(val)

        except Exception:
            return 1e300


    res = scipy_minimize_box(fun=obj, x0=x0, bounds=bounds, max_iter=max_iter, method="L-BFGS-B")

    # extra_t_hat = {n: float(res.x[j]) for j, n in enumerate(names)}
    # extra_hat = family.inverse_transform_extra(extra_t_hat)
    # family.validate_extra(extra_hat)
    # return extra_hat
    extra_t_hat = {n: float(res.x[j]) for j, n in enumerate(names)}
    extra_hat = family.inverse_transform_extra(extra_t_hat)
    try:
        family.validate_extra(extra_hat)
    except Exception:
        # fallback: keep previous valid extra0
        return dict(extra0)
    return extra_hat



def mstep_component(
    *,
    X: Array,
    y: Array,
    tau_k: Array,
    comp: ComponentSpec,
    beta_k: Array,
    extra_k: Dict[str, Any],
    offset: Array | None = None,
    inner_iter: int = 2,
) -> Tuple[Array, Dict[str, Any]]:
    """
    Alternating (GEM) updates for (beta_k, extra_k).
    """
    beta_new = np.asarray(beta_k, dtype=float)
    extra_new = dict(extra_k)

    for _ in range(inner_iter):
        beta_new = optimize_beta(
            X=X, y=y, tau_k=tau_k, comp=comp,
            beta0=beta_new, extra_k=extra_new, offset=offset,
            max_iter=200, tol=1e-6, step0=1e-2, fd_eps=1e-6,
        )
        extra_new = optimize_extra(
            X=X, y=y, tau_k=tau_k, comp=comp,
            beta_k=beta_new, extra0=extra_new, offset=offset,
            max_iter=200,
        )

    return beta_new, extra_new
