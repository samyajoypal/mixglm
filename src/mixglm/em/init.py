# src/mixglm/em/init.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np

from mixglm.model.mixture_glm import ComponentSpec
from mixglm.utils.numerics import normalize_simplex, weighted_mean, weighted_var

Array = np.ndarray


@dataclass
class InitState:
    """
    Initialization container for EM.
    """
    pi: Array
    betas: List[Array]
    extras: List[Dict[str, Any]]
    tau: Array


def init_tau_quantile(y: Array, K: int) -> Array:
    """
    Quantile-based hard-ish initialization on y (1D).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    qs = np.quantile(y, np.linspace(0, 1, K + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12

    z = np.zeros(n, dtype=int)
    for k in range(K):
        mask = (y > qs[k]) & (y <= qs[k + 1])
        z[mask] = k

    tau = np.full((n, K), 0.1 / max(K - 1, 1), dtype=float)
    tau[np.arange(n), z] = 0.9
    tau /= tau.sum(axis=1, keepdims=True)
    return tau


def init_tau_random(n: int, K: int, rng: np.random.Generator) -> Array:
    """
    Random responsibilities from a symmetric Dirichlet-like draw.
    """
    A = rng.gamma(shape=1.0, scale=1.0, size=(n, K))
    return A / A.sum(axis=1, keepdims=True)


def _kmeans_1d(y: Array, K: int, rng: np.random.Generator, n_iter: int = 25) -> Tuple[Array, Array]:
    y = np.asarray(y, dtype=float)
    n = y.size
    centers = rng.choice(y, size=K, replace=False) if n >= K else np.linspace(y.min(), y.max(), K)
    labels = np.zeros(n, dtype=int)

    for _ in range(n_iter):
        d2 = (y[:, None] - centers[None, :]) ** 2
        new_labels = np.argmin(d2, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                centers[k] = y[mask].mean()
            else:
                centers[k] = rng.choice(y)
    return centers, labels


def init_tau_kmeans_y(y: Array, K: int, rng: np.random.Generator) -> Array:
    """
    Simple 1D k-means on y to initialize responsibilities.
    """
    _, labels = _kmeans_1d(y, K, rng=rng)
    n = y.size
    tau = np.full((n, K), 0.1 / max(K - 1, 1), dtype=float)
    tau[np.arange(n), labels] = 0.9
    tau /= tau.sum(axis=1, keepdims=True)
    return tau


def _cluster_scale_y(y: Array, components: Sequence[ComponentSpec]) -> Array:
    """
    Stabilized one-dimensional scale for y-only clustering.

    This is only an initialization device. Count and positive continuous responses
    are clustered on a log-like scale so a few extreme observations do not dominate
    the initial centers; unit-interval responses are clustered on a logit scale.
    """
    y = np.asarray(y, dtype=float)
    eps = 1e-8

    if np.all((y >= -eps) & (y <= 1.0 + eps)):
        yy = np.clip(y, eps, 1.0 - eps)
        return np.log(yy / (1.0 - yy))

    all_count = all(getattr(c.family.support, "kind", "") == "nonnegative_int" for c in components)
    if all_count or (np.all(y >= -eps) and np.all(np.abs(y - np.round(y)) <= eps)):
        return np.log1p(np.clip(y, 0.0, None))

    if np.all(y > 0.0):
        return np.log(np.clip(y, eps, None))

    return y


def init_tau_kmeans_y_scaled(
    y: Array,
    K: int,
    rng: np.random.Generator,
    components: Sequence[ComponentSpec],
) -> Array:
    """
    1D k-means initialization on a support-aware transformation of y.
    """
    z = _cluster_scale_y(y, components)
    _, labels = _kmeans_1d(z, K, rng=rng)
    n = np.asarray(y).size
    tau = np.full((n, K), 0.1 / max(K - 1, 1), dtype=float)
    tau[np.arange(n), labels] = 0.9
    tau /= tau.sum(axis=1, keepdims=True)
    return tau


def init_parameters(
    *,
    y: Array,
    X: Array,
    components: Sequence[ComponentSpec],
    init: str,
    rng: np.random.Generator,
    min_pi: float = 1e-6,
    offset: Array | None = None,
) -> InitState:
    """
    Initialize (tau, pi, betas, extras) for EM.

    betas are initialized via a simple weighted least squares on a link-adjusted target:
    - identity: y
    - log: log(y clipped)
    - otherwise: zeros

    extras are initialized using family.initialize_extra(y, tau_k).
    """
    y = np.asarray(y)
    X = np.asarray(X)
    n, p = X.shape
    K = len(components)
    offset_use = np.zeros(n, dtype=float) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
    if offset_use.shape != (n,):
        raise ValueError(f"offset must have shape ({n},); got {offset_use.shape}.")

    do_component_glm_init = False
    init_base = str(init).lower()
    if init_base == "quantile":
        tau = init_tau_quantile(y, K)
    elif init_base == "random":
        tau = init_tau_random(n, K, rng)
    elif init_base == "kmeans_y":
        tau = init_tau_kmeans_y(y, K, rng)
    elif init_base == "kmeans_glm":
        tau = init_tau_kmeans_y_scaled(y, K, rng, components)
        do_component_glm_init = True
    elif init_base == "quantile_glm":
        tau = init_tau_quantile(y, K)
        do_component_glm_init = True
    else:
        raise ValueError(
            "init must be one of: 'quantile', 'random', 'kmeans_y', "
            "'kmeans_glm', 'quantile_glm'."
        )

    pi = normalize_simplex(np.clip(tau.mean(axis=0), min_pi, 1.0), min_val=min_pi)

    betas: List[Array] = []
    extras: List[Dict[str, Any]] = []

    for k, comp in enumerate(components):
        w = tau[:, k].astype(float)

        # crude link-based target for initialization
        lname = comp.link.name.lower()
        if lname == "identity":
            y_tgt = y.astype(float) - offset_use
        elif lname == "log":
            y_tgt = np.log(np.clip(y.astype(float), 1e-8, None)) - offset_use
        else:
            y_tgt = np.zeros_like(y, dtype=float)

        # Weighted least squares (with tiny ridge to avoid singularity)
        sw = np.sqrt(np.clip(w, 0.0, None))
        Xw = X * sw[:, None]
        yw = y_tgt * sw
        ridge = 1e-8
        Xw_aug = np.vstack([Xw, np.sqrt(ridge) * np.eye(p)])
        yw_aug = np.concatenate([yw, np.zeros(p)])
        b, *_ = np.linalg.lstsq(Xw_aug, yw_aug, rcond=None)
        if comp.coef_mask is not None:
            mask = np.asarray(comp.coef_mask, dtype=bool)
            if mask.shape != (p,):
                raise ValueError(f"coef_mask must have shape ({p},); got {mask.shape}.")
            b[~mask] = 0.0
        betas.append(b.astype(float))

        extra0 = comp.family.initialize_extra(y=y, tau_k=w)
        if comp.family.num_extra_params() > 0:
            comp.family.validate_extra(extra0)
        extras.append(extra0)

    if do_component_glm_init:
        try:
            from mixglm.em.mstep import mstep_component

            for k, comp in enumerate(components):
                w = tau[:, k].astype(float)
                if float(np.sum(w)) <= max(2.0, 1e-8 * n):
                    continue
                beta_k, extra_k = mstep_component(
                    X=X,
                    y=y,
                    tau_k=w,
                    comp=comp,
                    beta_k=betas[k],
                    extra_k=extras[k],
                    offset=offset_use,
                    inner_iter=1,
                )
                if np.all(np.isfinite(beta_k)):
                    if comp.coef_mask is not None:
                        beta_k = np.asarray(beta_k, dtype=float).copy()
                        beta_k[~np.asarray(comp.coef_mask, dtype=bool)] = 0.0
                    betas[k] = beta_k.astype(float)
                    extras[k] = dict(extra_k)
        except Exception:
            # Initialization must never prevent EM from trying this start.
            # If a family-specific component fit fails, keep the WLS seed.
            pass

    return InitState(pi=pi, betas=betas, extras=extras, tau=tau)
