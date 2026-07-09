# src/mixglm/families/poisson.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class PoissonFamily(UnivariateFamily):
    """
    Poisson family for count data (nonnegative integers).

    Model:
        Y | Z=k ~ Poisson(lambda_k)
        lambda_k = mu_k  (mu produced by link inverse, typically log link)

    Parameters:
        mu : mean/rate (lambda) > 0
    Nuisance parameters:
        none
    """

    name: str = "poisson"
    is_discrete: bool = True

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="nonnegative_int")

    @property
    def default_link_name(self) -> str:
        # canonical GLM link for Poisson is log
        return "log"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ()

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        return {}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        return {}

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        return

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        log pmf:
            y * log(mu) - mu - log(y!)
        """
        y = np.asarray(y)
        mu = np.asarray(params.mu, dtype=float)

        # mu must be positive
        mu = np.clip(mu, 1e-12, None)

        # use gammaln for log-factorial if available
        try:
            from scipy.special import gammaln
            log_fact = gammaln(y + 1.0)
        except Exception:
            # fallback (less stable for large y)
            from math import lgamma
            log_fact = np.vectorize(lambda t: lgamma(float(t) + 1.0))(y)

        return y * np.log(mu) - mu - log_fact
