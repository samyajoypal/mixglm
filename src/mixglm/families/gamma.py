# src/mixglm/families/gamma.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class GammaFamily(UnivariateFamily):
    """
    Gamma family for positive continuous data.

    Mean/dispersion parameterization:
        E[Y] = mu > 0
        Var(Y) = mu^2 / k   (shape k > 0)
    Equivalently, shape k and scale theta = mu / k.

    Nuisance parameter:
        log_shape  (unconstrained), shape = exp(log_shape)

    pdf:
        f(y) = y^{k-1} exp(-y/theta) / (Gamma(k) theta^k),  y>0
    """

    name: str = "gamma"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="positive")

    @property
    def default_link_name(self) -> str:
        # common mean link for Gamma is log
        return "log"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("log_shape",)

    # ------------------------- initialization / transforms -------------------------

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Initialize shape using crude method-of-moments on weighted data:
            k ≈ (mean^2) / var
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
        k = max((m * m) / v, 1e-6)
        return {"log_shape": float(np.log(k))}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        return {"log_shape": (np.log(1e-12), np.log(1e6))}

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        if "log_shape" not in extra:
            raise ValueError("Gamma requires 'log_shape' in extra params.")
        ls = float(extra["log_shape"])
        if not np.isfinite(ls):
            raise ValueError("log_shape must be finite.")
        k = float(np.exp(ls))
        if not (k > 0.0) or not np.isfinite(k):
            raise ValueError("shape must be positive and finite.")

    def _shape(self, extra: Dict[str, Any]) -> float:
        self.validate_extra(extra)
        return float(np.exp(float(extra["log_shape"])))

    # ------------------------- log pdf -------------------------

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        log pdf:
            (k-1) log y - y/theta - log Gamma(k) - k log theta
        where theta = mu/k
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(params.mu, dtype=float)

        y = np.clip(y, 1e-300, None)     # positive support
        mu = np.clip(mu, 1e-12, None)

        k = self._shape(params.extra)
        theta = mu / k
        theta = np.clip(theta, 1e-300, None)

        try:
            from scipy.special import gammaln
            out = (k - 1.0) * np.log(y) - (y / theta) - gammaln(k) - k * np.log(theta)
        except Exception:
            from math import lgamma
            out = (k - 1.0) * np.log(y) - (y / theta) - float(lgamma(k)) - k * np.log(theta)

        return out
