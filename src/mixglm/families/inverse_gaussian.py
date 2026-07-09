# src/mixglm/families/inverse_gaussian.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class InverseGaussianFamily(UnivariateFamily):
    """
    Inverse Gaussian (Wald) family for positive continuous data.

    Parameterization:
      mean mu > 0 (modeled via link inverse)
      shape/precision lambda > 0 (nuisance)

    pdf:
      f(y) = sqrt(lambda / (2*pi*y^3)) * exp( -lambda * (y - mu)^2 / (2*mu^2*y) ), y>0

    Nuisance parameter:
      log_lambda  (lambda = exp(log_lambda))
    """

    name: str = "inverse_gaussian"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="positive")

    @property
    def default_link_name(self) -> str:
        # common choice for mean link is log
        return "log"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("log_lambda",)

    # ------------------------- initialization / transforms -------------------------

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Moment-based crude init for lambda using:
          Var(Y) = mu^3 / lambda  -> lambda ≈ mu^3 / Var(Y)
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(tau_k, dtype=float)

        ws = float(np.sum(w))
        if ws <= 0:
            m = float(np.mean(y)) if y.size else 1.0
            v = float(np.var(y)) if y.size else 1.0
        else:
            m = float(np.sum(w * y) / ws)
            v = float(np.sum(w * (y - m) ** 2) / max(ws, 1.0))

        m = max(m, 1e-8)
        v = max(v, 1e-12)
        lam = max((m ** 3) / v, 1e-6)
        return {"log_lambda": float(np.log(lam))}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        return {"log_lambda": (np.log(1e-12), np.log(1e12))}

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        if "log_lambda" not in extra:
            raise ValueError("InverseGaussian requires 'log_lambda' in extra params.")
        ll = float(extra["log_lambda"])
        if not np.isfinite(ll):
            raise ValueError("log_lambda must be finite.")
        lam = float(np.exp(ll))
        if not (lam > 0.0) or not np.isfinite(lam):
            raise ValueError("lambda must be positive and finite.")

    def _lambda(self, extra: Dict[str, Any]) -> float:
        self.validate_extra(extra)
        return float(np.exp(float(extra["log_lambda"])))

    # ------------------------- log pdf -------------------------

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        log pdf:
          0.5*(log lambda - log(2*pi) - 3 log y)
          - lambda*(y-mu)^2 / (2*mu^2*y)
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(params.mu, dtype=float)

        y = np.clip(y, 1e-300, None)
        mu = np.clip(mu, 1e-12, None)

        lam = self._lambda(params.extra)

        out = (
            0.5 * (np.log(lam) - np.log(2.0 * np.pi) - 3.0 * np.log(y))
            - lam * (y - mu) ** 2 / (2.0 * (mu ** 2) * y)
        )
        return out
