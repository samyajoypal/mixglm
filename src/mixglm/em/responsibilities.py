# src/mixglm/em/responsibilities.py
from __future__ import annotations

from typing import List, Dict, Any, Sequence, Tuple
import numpy as np

from mixglm.model.mixture_glm import ComponentSpec
from mixglm.utils.numerics import logsumexp, softmax_from_log, safe_log

Array = np.ndarray


def log_component_terms(
    *,
    y: Array,
    X: Array,
    components: Sequence[ComponentSpec],
    pi: Array,
    betas: Sequence[Array],
    extras: Sequence[Dict[str, Any]],
    offset: Array | None = None,
) -> Array:
    """
    Compute log terms:
        log_terms[i,k] = log(pi_k) + log f_k(y_i | x_i; beta_k, extra_k)

    Returns
    -------
    log_terms : array (n, K)
    """
    y = np.asarray(y)
    X = np.asarray(X)
    n = y.shape[0]
    K = len(components)
    if offset is None:
        offset_use = np.zeros(n, dtype=float)
    else:
        offset_use = np.asarray(offset, dtype=float).reshape(-1)
        if offset_use.shape != (n,):
            raise ValueError(f"offset must have shape ({n},); got {offset_use.shape}.")
        if not np.all(np.isfinite(offset_use)):
            raise ValueError("offset must contain only finite values.")
    log_terms = np.empty((n, K), dtype=float)

    for k, comp in enumerate(components):
        mu = comp.link.inverse(X @ betas[k] + offset_use)
        ll = comp.family.loglik_component(y=y, mu=mu, extra=extras[k])
        log_terms[:, k] = np.log(pi[k]) + ll

    return log_terms


def responsibilities_from_log_terms(log_terms: Array) -> Array:
    """
    Convert log_terms (n,K) into responsibilities tau (n,K) stably.
    """
    return softmax_from_log(log_terms, axis=1)


def observed_loglik_from_log_terms(log_terms: Array) -> float:
    """
    Observed-data log-likelihood:
        sum_i log sum_k exp(log_terms[i,k])
    """
    return float(np.sum(logsumexp(log_terms, axis=1)))


def compute_responsibilities(
    *,
    y: Array,
    X: Array,
    components: Sequence[ComponentSpec],
    pi: Array,
    betas: Sequence[Array],
    extras: Sequence[Dict[str, Any]],
    offset: Array | None = None,
) -> Tuple[Array, float]:
    """
    Convenience:
    - compute log_terms
    - compute tau
    - compute loglik

    Returns
    -------
    tau : (n,K)
    loglik : float
    """
    log_terms = log_component_terms(
        y=y, X=X,
        components=components,
        pi=pi, betas=betas, extras=extras, offset=offset
    )
    tau = responsibilities_from_log_terms(log_terms)
    ll = observed_loglik_from_log_terms(log_terms)
    return tau, ll
