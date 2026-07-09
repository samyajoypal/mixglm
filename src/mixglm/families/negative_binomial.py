# src/mixglm/families/negative_binomial.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np

from mixglm.families.base import (
    UnivariateFamily,
    FamilySupport,
    FamilyParams,
    Array,
)


class NegativeBinomial2Family(UnivariateFamily):
    """
    Negative Binomial (NB2) family for count data with overdispersion.

    Parameterization (NB2):
        Var(Y) = mu + alpha * mu^2,  alpha > 0

    One convenient pmf uses "size" r = 1/alpha and p = r/(r+mu):
        P(Y=y) = C(y+r-1, y) * (1-p)^y * p^r

    Internal parameterization:
        log_alpha : unconstrained, alpha = exp(log_alpha)

    Notes:
    - Default link: log (canonical-ish for counts)
    - This family is useful as a building block; a ZINB family can be added later.
    """

    name: str = "nb2"
    is_discrete: bool = True

    @property
    def support(self) -> FamilySupport:
        return FamilySupport(kind="nonnegative_int")

    @property
    def default_link_name(self) -> str:
        return "log"

    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        return ("log_alpha",)

    # ------------------------- initialization / transforms -------------------------

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Initialize alpha via a crude moment estimate on weighted data:
            alpha ≈ max( (var - mean) / mean^2, small )
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
        alpha = max((v - m) / (m * m), 1e-6)
        return {"log_alpha": float(np.log(alpha))}

    def bounds_extra(self) -> Dict[str, Tuple[Any, Any]]:
        return {"log_alpha": (np.log(1e-12), np.log(1e6))}

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        return dict(extra_t)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        if "log_alpha" not in extra:
            raise ValueError("NB2 requires 'log_alpha' in extra params.")
        la = float(extra["log_alpha"])
        if not np.isfinite(la):
            raise ValueError("log_alpha must be finite.")
        alpha = float(np.exp(la))
        if not (alpha > 0.0) or not np.isfinite(alpha):
            raise ValueError("alpha must be positive and finite.")

    def _alpha(self, extra: Dict[str, Any]) -> float:
        self.validate_extra(extra)
        return float(np.exp(float(extra["log_alpha"])))

    # ------------------------- log pmf -------------------------

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        log pmf for NB2 using r=1/alpha, p=r/(r+mu):

            log Γ(y+r) - log Γ(r) - log Γ(y+1)
          + r log p + y log(1-p)

        where p = r/(r+mu), 1-p = mu/(r+mu)
        """
        y = np.asarray(y)
        mu = np.asarray(params.mu, dtype=float)
        mu = np.clip(mu, 1e-12, None)

        alpha = self._alpha(params.extra)
        r = 1.0 / alpha
        p = r / (r + mu)
        one_minus_p = mu / (r + mu)

        try:
            from scipy.special import gammaln
            out = (
                gammaln(y + r)
                - gammaln(r)
                - gammaln(y + 1.0)
                + r * np.log(p)
                + y * np.log(one_minus_p)
            )
        except Exception:
            from math import lgamma
            # vectorized fallback
            lg = np.vectorize(lambda t: lgamma(float(t)))
            out = (
                lg(y + r)
                - lg(np.array(r))
                - lg(y + 1.0)
                + r * np.log(p)
                + y * np.log(one_minus_p)
            )

        return out
