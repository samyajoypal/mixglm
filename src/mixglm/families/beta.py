# src/mixglm/families/beta.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class BetaFamily(UnivariateFamily):
    """
    Beta family for data in (0,1).

    Mean-precision parameterization:
        E[Y] = mu in (0,1)        (modeled via link inverse, typically logit)
        Precision phi > 0        (nuisance, constant per component)

    Convert to shape parameters:
        a = mu * phi
        b = (1-mu) * phi

    Nuisance parameter:
        log_phi (phi = exp(log_phi))

    log pdf:
        log Γ(a+b) - log Γ(a) - log Γ(b) + (a-1) log y + (b-1) log(1-y)
    """

    name: str = "beta"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="unit_interval")

    @property
    def default_link_name(self) -> str:
        return "logit"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("log_phi",)

    # ------------------------- initialization / transforms -------------------------

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Initialize phi using weighted method-of-moments:
          Var(Y) = mu(1-mu)/(phi+1)  ->  phi ≈ mu(1-mu)/Var - 1
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(tau_k, dtype=float)

        eps = 1e-8
        y = np.clip(y, eps, 1.0 - eps)

        ws = float(np.sum(w))
        if ws <= 0:
            m = float(np.mean(y))
            v = float(np.var(y))
        else:
            m = float(np.sum(w * y) / ws)
            v = float(np.sum(w * (y - m) ** 2) / max(ws, 1.0))

        m = float(np.clip(m, eps, 1.0 - eps))
        v = max(float(v), 1e-12)

        phi = max(m * (1.0 - m) / v - 1.0, 1e-6)
        return {"log_phi": float(np.log(phi))}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        return {"log_phi": (np.log(1e-12), np.log(1e8))}

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        if "log_phi" not in extra:
            raise ValueError("Beta requires 'log_phi' in extra params.")
        lp = float(extra["log_phi"])
        if not np.isfinite(lp):
            raise ValueError("log_phi must be finite.")
        phi = float(np.exp(lp))
        if not (phi > 0.0) or not np.isfinite(phi):
            raise ValueError("phi must be positive and finite.")

    def _phi(self, extra: Dict[str, Any]) -> float:
        self.validate_extra(extra)
        return float(np.exp(float(extra["log_phi"])))

    # ------------------------- log pdf -------------------------

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        y = np.asarray(y, dtype=float)
        mu = np.asarray(params.mu, dtype=float)

        eps = 1e-8
        y = np.clip(y, eps, 1.0 - eps)
        mu = np.clip(mu, eps, 1.0 - eps)

        phi = self._phi(params.extra)
        a = mu * phi
        b = (1.0 - mu) * phi

        try:
            from scipy.special import gammaln
            out = (
                gammaln(a + b)
                - gammaln(a)
                - gammaln(b)
                + (a - 1.0) * np.log(y)
                + (b - 1.0) * np.log(1.0 - y)
            )
        except Exception:
            from math import lgamma
            out = (
                (np.vectorize(lambda t: lgamma(float(t)))(a + b))
                - (np.vectorize(lambda t: lgamma(float(t)))(a))
                - (np.vectorize(lambda t: lgamma(float(t)))(b))
                + (a - 1.0) * np.log(y)
                + (b - 1.0) * np.log(1.0 - y)
            )

        return out
