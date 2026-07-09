# # src/mixglm/families/genhyperbolic.py
# from __future__ import annotations

# from typing import Dict, Tuple, Any
# import numpy as np

# from mixglm.families.base import (
    # UnivariateFamily,
    # FamilySupport,
    # FamilyParams,
    # Array,
# )


# class GeneralizedHyperbolicFamily(UnivariateFamily):
    # """
    # Univariate Generalized Hyperbolic (GH) distribution (SciPy-backed).

    # We rely on scipy.stats.genhyperbolic. SciPy uses parameters:
        # p : shape (lambda)
        # a : alpha  (a > |b|)
        # b : beta   (skewness, real, but must satisfy |b| < a)
        # loc   : location
        # scale : scale > 0

    # In our framework:
      # - params.mu is interpreted as loc (location), typically modeled by identity link
      # - nuisance (extra) parameters:
            # p        (lambda)
            # log_a    (a = exp(log_a) > 0)
            # b        (beta)
            # log_scale (scale = exp(log_scale) > 0)

    # Constraints:
      # - a > |b|
      # - scale > 0

    # Implementation note:
      # We enforce a > |b| by soft constraint in validate_extra:
        # if a <= |b| + eps: set invalid (optimizer will avoid) or raise.
      # Since our nuisance optimization uses box bounds, not nonlinear constraints,
      # we recommend keeping b within a moderate range and initializing near feasible.
    # """

    # name: str = "genhyperbolic"
    # is_discrete: bool = False

    # @property
    # def support(self) -> FamilySupport:
        # return FamilySupport(kind="real")

    # @property
    # def default_link_name(self) -> str:
        # # loc is modeled; identity is natural
        # return "identity"

    # @property
    # def extra_param_names(self) -> Tuple[str, ...]:
        # return ("p", "log_a", "b", "log_scale")

    # # ------------------------- initialization / transforms -------------------------

    # def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        # """
        # Conservative init that is feasible:
          # p = 1.0
          # a = 1.5
          # b = 0.0
          # scale = weighted std(y)
        # """
        # y = np.asarray(y, dtype=float)
        # w = np.asarray(tau_k, dtype=float)
        # ws = float(np.sum(w))

        # if ws <= 0:
            # s = float(np.std(y))
        # else:
            # m = float(np.sum(w * y) / ws)
            # v = float(np.sum(w * (y - m) ** 2) / max(ws, 1.0))
            # s = float(np.sqrt(max(v, 1e-12)))

        # s = max(s, 1e-6)
        # return {"p": 1.0, "log_a": float(np.log(1.5)), "b": 0.0, "log_scale": float(np.log(s))}

    # def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        # # loose but bounded to help stability
        # return {
            # "p": (-50.0, 50.0),
            # "log_a": (np.log(1e-6), np.log(1e6)),
            # "b": (-1e3, 1e3),
            # "log_scale": (np.log(1e-12), np.log(1e6)),
        # }

    # def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        # return dict(extra)

    # def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        # return dict(extra_t)

    # def validate_extra(self, extra: Dict[str, Any]) -> None:
        # for k in ("p", "log_a", "b", "log_scale"):
            # if k not in extra:
                # raise ValueError(f"GeneralizedHyperbolic requires '{k}' in extra params.")
        # p = float(extra["p"])
        # la = float(extra["log_a"])
        # b = float(extra["b"])
        # ls = float(extra["log_scale"])

        # if not (np.isfinite(p) and np.isfinite(la) and np.isfinite(b) and np.isfinite(ls)):
            # raise ValueError("GH extra params must be finite.")

        # a = float(np.exp(la))
        # scale = float(np.exp(ls))

        # if not (a > 0.0 and np.isfinite(a)):
            # raise ValueError("GH requires a>0.")
        # if not (scale > 0.0 and np.isfinite(scale)):
            # raise ValueError("GH requires scale>0.")

        # # SciPy requirement: a > |b|
        # eps = 1e-8
        # if not (a > abs(b) + eps):
            # raise ValueError(f"GH requires a > |b|. Got a={a}, b={b}.")

    # def _params(self, extra: Dict[str, Any]) -> Tuple[float, float, float, float]:
        # self.validate_extra(extra)
        # p = float(extra["p"])
        # a = float(np.exp(float(extra["log_a"])))
        # b = float(extra["b"])
        # scale = float(np.exp(float(extra["log_scale"])))
        # return p, a, b, scale

    # # ------------------------- log pdf -------------------------

    # def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        # """
        # Delegate to scipy.stats.genhyperbolic.logpdf if available.
        # """
        # y = np.asarray(y, dtype=float)
        # loc = np.asarray(params.mu, dtype=float)
        # p, a, b, scale = self._params(params.extra)

        # try:
            # from scipy.stats import genhyperbolic
            # return genhyperbolic.logpdf(y, p=p, a=a, b=b, loc=loc, scale=scale)
        # except Exception:
            # raise ImportError(
                # "GeneralizedHyperbolicFamily requires scipy (scipy.stats.genhyperbolic). "
                # "Install SciPy or remove this family from the model space."
            # )


# src/mixglm/families/genhyperbolic.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class GeneralizedHyperbolicFamily(UnivariateFamily):
    """
    Univariate Generalized Hyperbolic (GH) distribution (SciPy-backed).

    SciPy parameters:
        p : shape (lambda)
        a : alpha (requires a > |b|)
        b : beta  (skewness, real)
        loc   : location
        scale : scale > 0

    In our framework:
      - params.mu is interpreted as loc (location), typically modeled by identity link
      - nuisance (extra) parameters in *transformed space*:
            p           (real)
            b           (real)
            u           (real)   where a = sqrt(b^2 + exp(u))  => always a > |b|
            log_scale   (real)   scale = exp(log_scale) > 0

    This parameterization is robust for:
      - SciPy optimization (no nonlinear constraints needed)
      - Louis finite differences (small steps never violate a > |b|)
    """

    name: str = "genhyperbolic"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="real")

    @property
    def default_link_name(self) -> str:
        return "identity"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        # NOTE: these are TRANSFORMED names used by optimizer + Louis
        return ("p", "b", "u", "log_scale")

    # ------------------------- initialization / transforms -------------------------

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Conservative, feasible init:
          p = 1.0
          b = 0.0
          u = log(delta^2), with delta ~ 1.5 => u=log(1.5^2)
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

        delta = 1.5
        return {
            "p": 1.0,
            "b": 0.0,
            "u": float(np.log(delta * delta)),      # exp(u) = delta^2
            "log_scale": float(np.log(s)),
        }

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        # keep loose bounds but avoid extreme exp(u) overflow
        return {
            "p": (-50.0, 50.0),
            "b": (-1e3, 1e3),
            "u": (np.log(1e-12), np.log(1e12)),          # exp(u) in [1e-12, 1e12]
            "log_scale": (np.log(1e-12), np.log(1e6)),
        }

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """
        External -> transformed.

        We accept either:
          A) new safe format: {"p","b","u","log_scale"}
          B) old format: {"p","log_a","b","log_scale"}  (backward compatible)

        Old -> new conversion:
          a = exp(log_a)
          u = log(a^2 - b^2)  (must be positive)
        """
        if all(k in extra for k in ("p", "b", "u", "log_scale")):
            return dict(extra)

        if all(k in extra for k in ("p", "log_a", "b", "log_scale")):
            p = float(extra["p"])
            b = float(extra["b"])
            log_a = float(extra["log_a"])
            log_scale = float(extra["log_scale"])

            a = float(np.exp(log_a))
            # ensure positive gap
            gap = a * a - b * b
            if gap <= 0:
                # fall back to a minimal feasible gap
                gap = 1e-12
            u = float(np.log(gap))
            return {"p": p, "b": b, "u": u, "log_scale": log_scale}

        # otherwise, just pass through (validate_extra will raise cleanly)
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformed -> external. We keep the transformed keys as the stored keys,
        because they are unconstrained and stable.

        If you *want* to expose {"log_a"} again, you can add it as a derived value,
        but internally we should keep ("p","b","u","log_scale").
        """
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        """
        Validate transformed extras.
        """
        for k in ("p", "b", "u", "log_scale"):
            if k not in extra:
                raise ValueError(f"GeneralizedHyperbolic requires '{k}' in extra params.")

        p = float(extra["p"])
        b = float(extra["b"])
        u = float(extra["u"])
        ls = float(extra["log_scale"])

        if not (np.isfinite(p) and np.isfinite(b) and np.isfinite(u) and np.isfinite(ls)):
            raise ValueError("GH extra params must be finite.")

        scale = float(np.exp(ls))
        if not (scale > 0.0 and np.isfinite(scale)):
            raise ValueError("GH requires scale>0.")

        # a is always > |b| by construction, but still guard overflow/NaN
        gap = float(np.exp(u))
        if not (gap > 0.0 and np.isfinite(gap)):
            raise ValueError("GH requires exp(u)>0 and finite.")
        a = float(np.sqrt(b * b + gap))
        if not (np.isfinite(a) and a > 0.0):
            raise ValueError("GH requires a>0 and finite.")

    def _params(self, extra: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        Return (p, a, b, scale) in SciPy's parameterization.
        """
        extra_t = self.transform_extra(extra)
        self.validate_extra(extra_t)

        p = float(extra_t["p"])
        b = float(extra_t["b"])
        u = float(extra_t["u"])
        scale = float(np.exp(float(extra_t["log_scale"])))

        # enforce a > |b| via a = sqrt(b^2 + exp(u))
        a = float(np.sqrt(b * b + float(np.exp(u))))
        return p, a, b, scale

    # ------------------------- log pdf -------------------------

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        Delegate to scipy.stats.genhyperbolic.logpdf.
        """
        y = np.asarray(y, dtype=float)
        loc = np.asarray(params.mu, dtype=float)
        p, a, b, scale = self._params(params.extra)

        try:
            from scipy.stats import genhyperbolic
            return genhyperbolic.logpdf(y, p=p, a=a, b=b, loc=loc, scale=scale)
        except Exception:
            raise ImportError(
                "GeneralizedHyperbolicFamily requires scipy (scipy.stats.genhyperbolic). "
                "Install SciPy or remove this family from the model space."
            )
