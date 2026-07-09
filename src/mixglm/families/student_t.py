# src/mixglm/families/student_t.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class StudentTFamily(UnivariateFamily):
    """
    Student-t family with identity link by default.

    Model:
        Y | Z=k ~ t_{nu_k}(loc=mu_k, scale=sigma_k)

    Parameters:
        mu        : location, comes from GLM link
        sigma     : scale > 0 (component-specific)
        nu        : degrees of freedom > 2 (component-specific, for finite variance)

    Internal parameterization (for optimization stability):
        log_sigma : unconstrained
        log_nu_m2 : unconstrained, with nu = 2 + exp(log_nu_m2)

    Notes:
    - This parameterization enforces sigma>0 and nu>2.
    - We provide a stable logpdf using scipy.special.gammaln if available,
      otherwise fallback to a less stable approximation.
    """

    name: str = "student_t"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="real")

    @property
    def default_link_name(self) -> str:
        return "identity"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("log_sigma", "log_nu_m2")

    # ------------------------- initialization -------------------------

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Initialize sigma via weighted MAD / std and nu to a moderate value (e.g. 10).
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(tau_k, dtype=float)
        w_sum = np.sum(w)

        if w_sum <= 0:
            s = np.std(y) if y.size > 1 else 1.0
        else:
            m = np.sum(w * y) / w_sum
            # weighted robust scale: sqrt(weighted variance) with lower bound
            s = np.sqrt(np.sum(w * (y - m) ** 2) / max(w_sum, 1.0))

        s = float(max(s, 1e-6))
        nu0 = 10.0  # moderate tails
        return {"log_sigma": float(np.log(s)), "log_nu_m2": float(np.log(max(nu0 - 2.0, 1e-6)))}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        # bounds in transformed space
        return {
            "log_sigma": (np.log(1e-8), np.log(1e8)),
            "log_nu_m2": (np.log(1e-8), np.log(1e6)),  # nu up to ~ 2 + 1e6
        }

    # ------------------------- transforms -------------------------

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        # already in unconstrained form (log_sigma, log_nu_m2)
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def _sigma_nu(self, extra: Dict[str, Any]) -> Tuple[float, float]:
        if "log_sigma" not in extra or "log_nu_m2" not in extra:
            raise ValueError("StudentTFamily requires 'log_sigma' and 'log_nu_m2' in extra params.")
        log_sigma = float(extra["log_sigma"])
        log_nu_m2 = float(extra["log_nu_m2"])
        if (not np.isfinite(log_sigma)) or (not np.isfinite(log_nu_m2)):
            raise ValueError("log_sigma and log_nu_m2 must be finite.")
        sigma = float(np.exp(log_sigma))
        nu = float(2.0 + np.exp(log_nu_m2))
        sigma = max(sigma, 1e-12)
        nu = max(nu, 2.0 + 1e-12)
        return sigma, nu

    # ------------------------- core likelihood -------------------------

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        log t_{nu}( (y-mu)/sigma ) - log sigma

        logpdf:
            log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(νπ) - log σ
            - (ν+1)/2 * log(1 + (1/ν) * ((y-μ)/σ)^2)
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(params.mu, dtype=float)
        sigma, nu = self._sigma_nu(params.extra)

        z = (y - mu) / sigma
        z2 = z * z

        # Use gammaln if available for stability
        try:
            from scipy.special import gammaln
            c = (
                gammaln((nu + 1.0) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * np.log(nu * np.pi)
                - np.log(sigma)
            )
        except Exception:
            # Fallback (less stable). This is OK for early dev; SciPy is recommended.
            from math import lgamma
            c = (
                lgamma((nu + 1.0) / 2.0)
                - lgamma(nu / 2.0)
                - 0.5 * np.log(nu * np.pi)
                - np.log(sigma)
            )

        return c - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)

    # ------------------------- validation -------------------------

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        _ = self._sigma_nu(extra)  # checks finiteness and constraints
