# src/mixglm/selection/criteria.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple
import numpy as np

from mixglm.model.mixture_glm import ComponentSpec

Array = np.ndarray


@dataclass(frozen=True)
class InfoCriteria:
    """
    Information criteria values for a fitted model.
    """
    loglik: float
    n_params: int
    n_obs: int
    aic: float
    bic: float
    icl: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        out = {"loglik": float(self.loglik), "aic": float(self.aic), "bic": float(self.bic)}
        if self.icl is not None:
            out["icl"] = float(self.icl)
        out["n_params"] = float(self.n_params)
        return out


def count_parameters(K: int, p: int, components: Sequence[ComponentSpec]) -> int:
    """
    Parameter count:
      (K-1) mixing proportions
    + K * p regression coefficients
    + sum_k (# nuisance params of family_k)

    Notes:
    - This is the standard count used in BIC/AIC for mixture models.
    - If you later add covariate-dependent nuisance parameters, update this count accordingly.
    """
    extra = sum(comp.family.num_extra_params() for comp in components)
    return (K - 1) + K * p + extra


def compute_aic_bic(loglik: float, n_params: int, n_obs: int) -> Tuple[float, float]:
    """
    AIC = -2 loglik + 2 p
    BIC = -2 loglik + p log(n)
    """
    aic = -2.0 * float(loglik) + 2.0 * int(n_params)
    bic = -2.0 * float(loglik) + float(np.log(n_obs)) * int(n_params)
    return float(aic), float(bic)


def compute_icl(bic: float, tau: Array, eps: float = 1e-15) -> float:
    """
    ICL = BIC - 2 * sum_{i,k} tau_{ik} log tau_{ik}
    """
    tau = np.asarray(tau, dtype=float)
    ent = float(np.sum(tau * np.log(np.clip(tau, eps, 1.0))))
    return float(bic - 2.0 * ent)


def evaluate_criteria(
    *,
    loglik: float,
    X: Array,
    components: Sequence[ComponentSpec],
    responsibilities: Optional[Array] = None,
    compute_icl_flag: bool = True,
) -> InfoCriteria:
    """
    Convenience wrapper to compute IC values given a fitted model.

    Parameters
    ----------
    loglik : float
        Observed-data log-likelihood (unpenalized).
    X : array (n,p)
        Design matrix.
    components : list of ComponentSpec
        Component specifications.
    responsibilities : array (n,K), optional
        Posterior responsibilities (for ICL).
    compute_icl_flag : bool
        If True and responsibilities provided, compute ICL.

    Returns
    -------
    InfoCriteria
    """
    X = np.asarray(X)
    n, p = X.shape
    K = len(components)

    n_params = count_parameters(K=K, p=p, components=components)
    aic, bic = compute_aic_bic(loglik=loglik, n_params=n_params, n_obs=n)

    icl = None
    if compute_icl_flag and responsibilities is not None:
        icl = compute_icl(bic=bic, tau=responsibilities)

    return InfoCriteria(
        loglik=float(loglik),
        n_params=int(n_params),
        n_obs=int(n),
        aic=float(aic),
        bic=float(bic),
        icl=float(icl) if icl is not None else None,
    )
