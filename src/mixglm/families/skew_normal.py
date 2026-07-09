# src/mixglm/families/skew_normal.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class SkewNormalFamily(UnivariateFamily):
    """
    Univariate Skew-Normal family (Azzalini).

    SciPy parameterization (scipy.stats.skewnorm):
        a     : shape (real)
        loc   : location (real)
        scale : scale > 0

    In our framework:
      - params.mu is interpreted as loc (location), typically modeled by identity link
      - nuisance parameters:
            log_scale (scale = exp(log_scale))
            shape     (a)

    Support: real line.
    """

    name: str = "skew_normal"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="real")

    @property
    def default_link_name(self) -> str:
        # since mu is loc, identity is natural
        return "identity"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("log_scale", "shape")

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Simple init: scale = weighted std(y), shape = 0 (reduces to normal).
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(tau_k, dtype=float)
        ws = float(np.sum(w))
        if ws <= 0:
            s = float(np.std(y))
        else:
            m = float(np.sum(w * y) / ws)
            v = float(np.sum(w * (y - m) ** 2) / max(ws, 1.0))
            s = float(np.sqrt(max(v, 1e-12)))

        s = max(s, 1e-6)
        return {"log_scale": float(np.log(s)), "shape": 0.0}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        # shape is unconstrained; scale positive via log_scale bounds
        return {"log_scale": (np.log(1e-12), np.log(1e6)), "shape": (None, None)}

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        if "log_scale" not in extra or "shape" not in extra:
            raise ValueError("SkewNormal requires 'log_scale' and 'shape' in extra params.")
        ls = float(extra["log_scale"])
        a = float(extra["shape"])
        if not np.isfinite(ls):
            raise ValueError("log_scale must be finite.")
        if not np.isfinite(a):
            raise ValueError("shape must be finite.")
        sc = float(np.exp(ls))
        if not (sc > 0.0) or not np.isfinite(sc):
            raise ValueError("scale must be positive and finite.")

    def _scale_shape(self, extra: Dict[str, Any]) -> Tuple[float, float]:
        self.validate_extra(extra)
        scale = float(np.exp(float(extra["log_scale"])))
        shape = float(extra["shape"])
        return scale, shape

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        Delegate to scipy.stats.skewnorm.logpdf if available.
        """
        y = np.asarray(y, dtype=float)
        loc = np.asarray(params.mu, dtype=float)
        scale, shape = self._scale_shape(params.extra)

        try:
            from scipy.stats import skewnorm
            return skewnorm.logpdf(y, a=shape, loc=loc, scale=scale)
        except Exception:
            # Minimal fallback: no SciPy -> raise with helpful message
            raise ImportError(
                "SkewNormalFamily requires scipy (scipy.stats.skewnorm). "
                "Install SciPy or remove this family from the model space."
            )

    def mean_from_mu(self, mu: Array, extra: Dict[str, Any]) -> Array:
        loc = np.asarray(mu, dtype=float)
        scale, shape = self._scale_shape(extra)
        delta = shape / np.sqrt(1.0 + shape * shape)
        return loc + scale * delta * np.sqrt(2.0 / np.pi)
