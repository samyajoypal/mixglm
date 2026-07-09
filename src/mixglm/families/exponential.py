# src/mixglm/families/exponential.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class ExponentialFamily(UnivariateFamily):
    """
    Exponential family for positive continuous data.

    Parameterization:
        mean mu > 0 (modeled via link inverse, typically log)
        rate = 1/mu

    pdf:
        f(y) = (1/mu) * exp(-y/mu), y>0

    Nuisance parameters:
        none
    """

    name: str = "exponential"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="positive")

    @property
    def default_link_name(self) -> str:
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
        log pdf:
          -log(mu) - y/mu   for y>0
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(params.mu, dtype=float)

        y = np.clip(y, 1e-300, None)
        mu = np.clip(mu, 1e-12, None)

        return -np.log(mu) - (y / mu)
