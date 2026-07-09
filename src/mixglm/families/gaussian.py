# src/mixglm/families/gaussian.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class GaussianFamily(UnivariateFamily):
    """
    Gaussian (Normal) family with identity link by default.

    Model:
        Y | Z=k ~ N(mu_k, sigma_k^2)

    Parameters:
        mu     : location (mean), comes from GLM link
        sigma2 : variance (component-specific, constant across observations)

    Notes:
    - sigma2 is estimated from data.
    - We parameterize via log(sigma2) internally to enforce positivity.
    """

    name: str = "gaussian"
    is_discrete: bool = False

    # ------------------------- metadata -------------------------

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="real")

    @property
    def default_link_name(self) -> str:
        return "identity"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        # store variance, not sd
        return ("log_sigma2",)

    # ------------------------- initialization -------------------------

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Initialize log(sigma^2) using weighted variance.
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(tau_k, dtype=float)
        w_sum = np.sum(w)

        if w_sum <= 0:
            # fallback
            var = np.var(y) if y.size > 1 else 1.0
        else:
            mu_hat = np.sum(w * y) / w_sum
            var = np.sum(w * (y - mu_hat) ** 2) / max(w_sum, 1.0)

        var = max(var, 1e-8)
        return {"log_sigma2": float(np.log(var))}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        # bounds in transformed space (log_sigma2)
        return {"log_sigma2": (np.log(1e-8), np.log(1e8))}

    # ------------------------- transforms -------------------------

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        # already unconstrained (log scale)
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        # already unconstrained (log scale)
        return dict(extra_t)

    # ------------------------- core likelihood -------------------------

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        log N(y | mu, sigma^2)
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(params.mu, dtype=float)

        if "log_sigma2" not in params.extra:
            raise ValueError("GaussianFamily requires 'log_sigma2' in extra params.")

        log_sigma2 = float(params.extra["log_sigma2"])
        sigma2 = np.exp(log_sigma2)

        # numerical safety
        sigma2 = max(sigma2, 1e-12)

        # log-density
        return -0.5 * (
            np.log(2.0 * np.pi * sigma2)
            + (y - mu) ** 2 / sigma2
        )

    # ------------------------- validation -------------------------

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        if "log_sigma2" not in extra:
            raise ValueError("Missing 'log_sigma2' for GaussianFamily.")
        log_sigma2 = float(extra["log_sigma2"])
        if not np.isfinite(log_sigma2):
            raise ValueError("log_sigma2 must be finite.")
