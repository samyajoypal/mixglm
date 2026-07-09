# src/mixglm/inference/numeric_se.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mixglm.model.mixture_glm import MixtureGLM, ComponentSpec
from mixglm.utils.numerics import logsumexp

Array = np.ndarray


@dataclass
class NumericSE:
    """
    Numerical Hessian-based standard errors for the unpenalized log-likelihood.

    Notes
    - Mixture models can have near-singular observed information (esp. with weak separation).
    - We return a pseudo-inverse covariance if needed.
    - This is intended as a baseline; Louis / analytic blocks can be added later.
    """
    theta_hat: Array
    cov: Array
    se: Array
    success: bool
    message: str
    param_slices: Dict[str, slice]


def _pack_theta(
    pi: Array,
    betas: Sequence[Array],
    extras: Sequence[Dict[str, Any]],
    components: Sequence[ComponentSpec],
) -> Tuple[Array, Dict[str, slice]]:
    """
    Pack parameters into a single vector:
      theta = [eta_pi (K-1 unconstrained), vec(betas), vec(extra_transformed)]

    We use:
      pi_k = softmax([eta_1,...,eta_{K-1}, 0])  (identifiability constraint)
    """
    K = len(components)
    p = betas[0].size

    # unconstrained mixing logits (K-1); last fixed to 0
    eta_pi = np.log(pi[:-1]) - np.log(pi[-1])

    parts = []
    slices: Dict[str, slice] = {}
    start = 0

    parts.append(eta_pi)
    slices["eta_pi"] = slice(start, start + (K - 1))
    start += (K - 1)

    bvec = np.concatenate([np.asarray(b).ravel() for b in betas])
    parts.append(bvec)
    slices["betas"] = slice(start, start + K * p)
    start += K * p

    # extras: family.transform_extra -> vector
    extra_list = []
    extra_slices: Dict[str, slice] = {}
    for k, comp in enumerate(components):
        names = comp.family.extra_param_names
        if len(names) == 0:
            continue
        extra_t = comp.family.transform_extra(extras[k])
        v = np.array([float(extra_t[n]) for n in names], dtype=float)
        extra_slices[f"extra_{k}"] = slice(start, start + v.size)
        start += v.size
        extra_list.append(v)

    if extra_list:
        parts.append(np.concatenate(extra_list))
    slices.update(extra_slices)

    theta = np.concatenate(parts) if parts else np.array([], dtype=float)
    return theta, slices


def _unpack_theta(
    theta: Array,
    *,
    components: Sequence[ComponentSpec],
    p: int,
) -> Tuple[Array, List[Array], List[Dict[str, Any]]]:
    """
    Inverse of _pack_theta.
    """
    K = len(components)
    theta = np.asarray(theta, dtype=float)

    pos = 0
    eta_pi = theta[pos: pos + (K - 1)]
    pos += (K - 1)

    # softmax with last logit = 0
    logits = np.concatenate([eta_pi, np.array([0.0])])
    m = np.max(logits)
    ex = np.exp(logits - m)
    pi = ex / np.sum(ex)

    betas = []
    for k in range(K):
        b = theta[pos: pos + p]
        pos += p
        betas.append(b.copy())

    extras: List[Dict[str, Any]] = []
    for k, comp in enumerate(components):
        names = comp.family.extra_param_names
        if len(names) == 0:
            extras.append({})
            continue
        v = theta[pos: pos + len(names)]
        pos += len(names)
        extra_t = {n: float(v[j]) for j, n in enumerate(names)}
        extra = comp.family.inverse_transform_extra(extra_t)
        comp.family.validate_extra(extra)
        extras.append(extra)

    return pi, betas, extras


def _loglik(
    y: Array,
    X: Array,
    components: Sequence[ComponentSpec],
    pi: Array,
    betas: Sequence[Array],
    extras: Sequence[Dict[str, Any]],
    offset: Array | None = None,
) -> float:
    """
    Observed-data log-likelihood (unpenalized).
    """
    y = np.asarray(y)
    X = np.asarray(X)
    n = y.shape[0]
    K = len(components)
    offset_use = np.zeros(n, dtype=float) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
    if offset_use.shape != (n,):
        raise ValueError(f"offset must have shape ({n},); got {offset_use.shape}.")

    log_terms = np.empty((n, K), dtype=float)
    for k, comp in enumerate(components):
        mu = comp.link.inverse(X @ betas[k] + offset_use)
        ll = comp.family.loglik_component(y=y, mu=mu, extra=extras[k])
        log_terms[:, k] = np.log(pi[k]) + ll

    return float(np.sum(logsumexp(log_terms, axis=1)))


def _numeric_hessian(f, x: Array, eps: float = 1e-5) -> Array:
    """
    Central finite difference Hessian for scalar f(x).
    O(p^2) evaluations, intended for small to moderate p.
    """
    x = np.asarray(x, dtype=float)
    p = x.size
    H = np.zeros((p, p), dtype=float)

    fx = float(f(x))
    for i in range(p):
        hi = eps * (1.0 + abs(float(x[i])))
        ei = np.zeros(p); ei[i] = 1.0

        for j in range(i, p):
            hj = eps * (1.0 + abs(float(x[j])))
            ej = np.zeros(p); ej[j] = 1.0

            if i == j:
                f1 = float(f(x + hi * ei))
                f2 = float(f(x - hi * ei))
                H[i, i] = (f1 - 2.0 * fx + f2) / (hi ** 2)
            else:
                fpp = float(f(x + hi * ei + hj * ej))
                fpm = float(f(x + hi * ei - hj * ej))
                fmp = float(f(x - hi * ei + hj * ej))
                fmm = float(f(x - hi * ei - hj * ej))
                val = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj)
                H[i, j] = val
                H[j, i] = val

    return H


def numeric_hessian_se(
    *,
    model: MixtureGLM,
    y: Array,
    X: Array,
    eps: float = 1e-5,
    use_pinv: bool = True,
    rcond: float = 1e-10,
    offset: Array | None = None,
) -> NumericSE:
    """
    Compute numerical Hessian-based SE for the fitted model (unpenalized likelihood).

    Important:
    - For penalized fits, these SEs are not formally valid.
      Use bootstrap for penalized inference.
    """
    if model.result_ is None:
        raise ValueError("Model must be fitted before calling numeric_hessian_se().")

    y = np.asarray(y)
    X = np.asarray(X)
    n, p = X.shape
    offset_use = np.zeros(n, dtype=float) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
    if offset_use.shape != (n,):
        raise ValueError(f"offset must have shape ({n},); got {offset_use.shape}.")
    components = model.components
    res = model.result_

    theta_hat, slices = _pack_theta(res.pi, res.betas, res.extras, components)

    def neg_ll(theta: Array) -> float:
        pi, betas, extras = _unpack_theta(theta, components=components, p=p)
        return -_loglik(y, X, components, pi, betas, extras, offset=offset_use)

    try:
        H = _numeric_hessian(neg_ll, theta_hat, eps=eps)
        # observed information is Hessian of negative loglik
        info = H

        if use_pinv:
            cov = np.linalg.pinv(info, rcond=rcond)
            msg = "Used pseudo-inverse of observed information."
            success = True
        else:
            cov = np.linalg.inv(info)
            msg = "Used inverse of observed information."
            success = True

        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        return NumericSE(
            theta_hat=theta_hat,
            cov=cov,
            se=se,
            success=success,
            message=msg,
            param_slices=slices,
        )
    except Exception as e:
        pdim = theta_hat.size
        cov = np.full((pdim, pdim), np.nan, dtype=float)
        se = np.full((pdim,), np.nan, dtype=float)
        return NumericSE(
            theta_hat=theta_hat,
            cov=cov,
            se=se,
            success=False,
            message=f"Failed to compute numeric SE: {e}",
            param_slices=slices,
        )
