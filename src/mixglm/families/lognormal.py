# src/mixglm/families/lognormal.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class LogNormalFamily(UnivariateFamily):
    """
    Lognormal family for positive continuous data.

    Parameterization:
      log(Y) ~ Normal(m, s^2)

    We model the *location* parameter m via the GLM link:
        m_i = eta_i = x_i^T beta   (typically identity link)
    and have nuisance parameter:
        log_sigma  (sigma = exp(log_sigma) > 0)

    Important:
    - Here the component "mu" returned by the link inverse is interpreted as m (mean of log Y),
      not E[Y]. That is consistent with our generic family interface treating 'mu' as the
      modeled location parameter.
    """

    name: str = "lognormal"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="positive")

    @property
    def default_link_name(self) -> str:
        # because mu is m = E[log Y], identity is the natural link
        return "identity"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("log_sigma",)

    # ------------------------- initialization / transforms -------------------------

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Initialize sigma using weighted SD of log(y).
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(tau_k, dtype=float)

        y = np.clip(y, 1e-300, None)
        z = np.log(y)

        ws = float(np.sum(w))
        if ws <= 0:
            s = float(np.std(z))
        else:
            m = float(np.sum(w * z) / ws)
            v = float(np.sum(w * (z - m) ** 2) / max(ws, 1.0))
            s = float(np.sqrt(max(v, 1e-12)))

        s = max(s, 1e-6)
        return {"log_sigma": float(np.log(s))}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        return {"log_sigma": (np.log(1e-12), np.log(1e6))}

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        if "log_sigma" not in extra:
            raise ValueError("LogNormal requires 'log_sigma' in extra params.")
        ls = float(extra["log_sigma"])
        if not np.isfinite(ls):
            raise ValueError("log_sigma must be finite.")
        s = float(np.exp(ls))
        if not (s > 0.0) or not np.isfinite(s):
            raise ValueError("sigma must be positive and finite.")

    def _sigma(self, extra: Dict[str, Any]) -> float:
        self.validate_extra(extra)
        return float(np.exp(float(extra["log_sigma"])))

    # ------------------------- log pdf -------------------------

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        log pdf for y>0:
            -log(y) - log(s) - 0.5 log(2pi) - (log y - m)^2 / (2 s^2)
        where m = params.mu and s = sigma(extra)
        """
        y = np.asarray(y, dtype=float)
        m = np.asarray(params.mu, dtype=float)

        y = np.clip(y, 1e-300, None)
        z = np.log(y)
        s = self._sigma(params.extra)
        s2 = s * s

        out = (
            -np.log(y)
            - np.log(s)
            - 0.5 * np.log(2.0 * np.pi)
            - (z - m) ** 2 / (2.0 * s2)
        )
        return out

    def mean_from_mu(self, mu: Array, extra: Dict[str, Any]) -> Array:
        m = np.asarray(mu, dtype=float)
        s = self._sigma(extra)
        return np.exp(m + 0.5 * s * s)
