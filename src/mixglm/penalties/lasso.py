# src/mixglm/penalties/lasso.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mixglm.penalties.base import BasePenalty, Array
from mixglm.penalties.elastic_net import ElasticNetPenalty


@dataclass(frozen=True)
class LassoPenalty(BasePenalty):
    """
    Lasso penalty:
        P(beta) = lam * ||beta||_1

    Prox:
        prox_{step * P}(beta) = soft_threshold(beta, step * lam)
    """
    lam: float = 1.0

    @property
    def name(self) -> str:
        return "lasso"

    def value(self, beta: Array) -> float:
        b = self._as1d(beta)
        return float(self.lam * np.sum(np.abs(b)))

    def grad(self, beta: Array) -> Array:
        """
        Subgradient of the L1 norm (not unique at 0).
        Prefer prox() in optimization.
        """
        b = self._as1d(beta)
        return self.lam * np.sign(b)

    @staticmethod
    def _soft_threshold(v: Array, t: float) -> Array:
        return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

    def prox(self, beta: Array, step: float) -> Array:
        v = self._as1d(beta)
        if step < 0:
            raise ValueError("step must be nonnegative.")
        if self.lam == 0.0 or step == 0.0:
            return v.copy()
        return self._soft_threshold(v, step * self.lam)

    def is_smooth(self) -> bool:
        return False


def as_elastic_net(p: "LassoPenalty") -> ElasticNetPenalty:
    """
    Convenience converter: represent lasso as elastic net with l1_ratio=1.
    Useful if you want to unify solver code paths.
    """
    return ElasticNetPenalty(lam=p.lam, l1_ratio=1.0)
