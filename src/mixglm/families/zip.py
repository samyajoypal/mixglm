from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy.special import gammaln
from mixglm.families.base import UnivariateFamily, FamilySupport, FamilyParams, Array

class ZIPoissonFamily(UnivariateFamily):
    name: str = "zip"
    is_discrete: bool = True

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="nonnegative_int")

    @property
    def default_link_name(self) -> str:
        return "log"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("logit_theta",)

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        w = np.asarray(tau_k, dtype=float)
        ws = float(np.sum(w)) + 1e-8
        frac = float(np.sum((y == 0) * w) / ws)
        theta = max(1e-3, min(frac * 0.5, 0.99))
        return {"logit_theta": float(np.log(theta / (1.0 - theta)))}

    def bounds_extra(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return {"logit_theta": (None, None)}

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        y = np.asarray(y)
        mu = np.clip(np.asarray(params.mu, dtype=float), 1e-12, None)
        lt = float(params.extra.get("logit_theta", -20.0))
        theta = 1.0 / (1.0 + np.exp(-lt))
        theta = np.clip(theta, 1e-12, 1.0 - 1e-12)

        ll = np.zeros_like(y, dtype=float)
        mask_zero = (y == 0)
        ll[mask_zero] = np.log(theta + (1.0 - theta) * np.exp(-mu[mask_zero]))

        mask_pos = ~mask_zero
        yp = y[mask_pos]
        mup = mu[mask_pos]

        ll[mask_pos] = np.log(1.0 - theta) - mup + yp * np.log(mup) - gammaln(yp + 1.0)
        return ll

    def mean_from_mu(self, mu: Array, extra: Dict[str, Any]) -> Array:
        mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
        lt = float(extra.get("logit_theta", -20.0))
        theta = 1.0 / (1.0 + np.exp(-lt))
        theta = np.clip(theta, 1e-12, 1.0 - 1e-12)
        return (1.0 - theta) * mu
