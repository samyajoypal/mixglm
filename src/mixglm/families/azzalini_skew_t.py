# src/mixglm/families/azzalini_skew_t.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class AzzaliniSkewTFamily(UnivariateFamily):
    """
    Azzalini skew-t distribution via the external package skewt-scipy.

    Package docs show:
        from skewt_scipy.skewt import skewt
        skewt.logpdf(x, a, df, loc=0, scale=1)

    In our framework:
      - params.mu is interpreted as loc (location), typically identity link
      - nuisance parameters:
            shape     (a, real)
            log_df    (df = exp(log_df) > 0)
            log_scale (scale = exp(log_scale) > 0)
    :contentReference[oaicite:3]{index=3}
    """

    name: str = "azzalini_skew_t"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="real")

    @property
    def default_link_name(self) -> str:
        return "identity"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("shape", "log_df", "log_scale")

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Conservative init:
          shape=0 (symmetric)
          df=10
          scale=weighted std(y)
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
        return {
            "shape": 0.0,
            "log_df": float(np.log(10.0)),
            "log_scale": float(np.log(s)),
        }

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        return {
            "shape": (None, None),
            "log_df": (np.log(1e-6), np.log(1e6)),
            "log_scale": (np.log(1e-12), np.log(1e6)),
        }

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        for k in ("shape", "log_df", "log_scale"):
            if k not in extra:
                raise ValueError(f"AzzaliniSkewTFamily requires '{k}' in extra params.")
        a = float(extra["shape"])
        ldf = float(extra["log_df"])
        ls = float(extra["log_scale"])
        if not (np.isfinite(a) and np.isfinite(ldf) and np.isfinite(ls)):
            raise ValueError("Azzalini skew-t extra params must be finite.")
        df = float(np.exp(ldf))
        scale = float(np.exp(ls))
        if not (df > 0.0 and np.isfinite(df)):
            raise ValueError("df must be positive and finite.")
        if not (scale > 0.0 and np.isfinite(scale)):
            raise ValueError("scale must be positive and finite.")

    def _params(self, extra: Dict[str, Any]) -> Tuple[float, float, float]:
        self.validate_extra(extra)
        shape = float(extra["shape"])
        df = float(np.exp(float(extra["log_df"])))
        scale = float(np.exp(float(extra["log_scale"])))
        return shape, df, scale

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        Delegate to skewt-scipy skewt.logpdf.
        """
        y = np.asarray(y, dtype=float)
        loc = np.asarray(params.mu, dtype=float)
        shape, df, scale = self._params(params.extra)

        try:
            from skewt_scipy.skewt import skewt
            return skewt.logpdf(x=y, a=shape, df=df, loc=loc, scale=scale)
        except Exception as e:
            raise ImportError(
                "AzzaliniSkewTFamily requires the external package 'skewt-scipy' "
                "(import: from skewt_scipy.skewt import skewt). "
                f"Original error: {e}"
            )
