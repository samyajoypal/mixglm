from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple
from mixglm.families.base import UnivariateFamily, FamilySupport, FamilyParams, Array

class GeometricFamily(UnivariateFamily):
    name: str = "geometric"
    is_discrete: bool = True

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="nonnegative_int")

    @property
    def default_link_name(self) -> str:
        return "log"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ()

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        y = np.asarray(y)
        mu = np.clip(np.asarray(params.mu, dtype=float), 1e-12, None)
        return y * np.log(mu) - (y + 1.0) * np.log(1.0 + mu)
