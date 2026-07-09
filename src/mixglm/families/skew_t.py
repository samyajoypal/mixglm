# src/mixglm/families/skew_t.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class SkewTFamily(UnivariateFamily):
    """
    Skew-t distribution based on SciPy's Jones and Faddy skew-t: scipy.stats.jf_skew_t.

    SciPy signature:
        jf_skew_t.logpdf(x, a, b, loc=0, scale=1)

    In our framework:
      - params.mu is interpreted as loc (location), typically modeled by identity link
      - nuisance parameters:
            log_a     (a = exp(log_a) > 0)
            log_b     (b = exp(log_b) > 0)
            log_scale (scale = exp(log_scale) > 0)

    Notes:
      - When a=b, this reduces to a symmetric t distribution with df = 2a (SciPy docs). :contentReference[oaicite:2]{index=2}
    """

    name: str = "jf_skew_t"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="real")

    @property
    def default_link_name(self) -> str:
        return "identity"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("log_a", "log_b", "log_scale")

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Conservative feasible init:
          a=b=5  (moderate tails)
          scale = weighted std(y)
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
        a0 = 5.0
        b0 = 5.0
        return {
            "log_a": float(np.log(a0)),
            "log_b": float(np.log(b0)),
            "log_scale": float(np.log(s)),
        }

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        return {
            "log_a": (np.log(1e-6), np.log(1e6)),
            "log_b": (np.log(1e-6), np.log(1e6)),
            "log_scale": (np.log(1e-12), np.log(1e6)),
        }

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        for k in ("log_a", "log_b", "log_scale"):
            if k not in extra:
                raise ValueError(f"SkewTFamily(jf_skew_t) requires '{k}' in extra params.")
        la = float(extra["log_a"])
        lb = float(extra["log_b"])
        ls = float(extra["log_scale"])
        if not (np.isfinite(la) and np.isfinite(lb) and np.isfinite(ls)):
            raise ValueError("jf_skew_t extra params must be finite.")
        a = float(np.exp(la))
        b = float(np.exp(lb))
        scale = float(np.exp(ls))
        if not (a > 0.0 and np.isfinite(a)):
            raise ValueError("jf_skew_t requires a>0.")
        if not (b > 0.0 and np.isfinite(b)):
            raise ValueError("jf_skew_t requires b>0.")
        if not (scale > 0.0 and np.isfinite(scale)):
            raise ValueError("jf_skew_t requires scale>0.")

    def _params(self, extra: Dict[str, Any]) -> Tuple[float, float, float]:
        self.validate_extra(extra)
        a = float(np.exp(float(extra["log_a"])))
        b = float(np.exp(float(extra["log_b"])))
        scale = float(np.exp(float(extra["log_scale"])))
        return a, b, scale

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        Delegate to scipy.stats.jf_skew_t.logpdf.
        """
        y = np.asarray(y, dtype=float)
        loc = np.asarray(params.mu, dtype=float)
        a, b, scale = self._params(params.extra)

        try:
            from scipy.stats import jf_skew_t
            return jf_skew_t.logpdf(y, a=a, b=b, loc=loc, scale=scale)
        except Exception as e:
            raise ImportError(
                "SkewTFamily(jf_skew_t) requires SciPy with scipy.stats.jf_skew_t. "
                f"Original error: {e}"
            )
