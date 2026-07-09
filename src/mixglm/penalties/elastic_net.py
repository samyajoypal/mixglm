# src/mixglm/penalties/elastic_net.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mixglm.penalties.base import BasePenalty, Array


@dataclass(frozen=True)
class ElasticNetPenalty(BasePenalty):
    """
    Elastic Net penalty:
        P(beta) = lam * ( l1_ratio * ||beta||_1 + (1 - l1_ratio)/2 * ||beta||_2^2 )

    where l1_ratio in [0, 1].

    Prox operator (closed form):
        prox_{step*P}(v) = soft_threshold(v, step*lam*l1_ratio) / (1 + step*lam*(1-l1_ratio))
    """
    l1_ratio: float = 0.5  # in [0,1]

    @property
    def name(self) -> str:
        return "elastic_net"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 <= float(self.l1_ratio) <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1].")

    def value(self, beta: Array) -> float:
        b = self._as1d(beta)
        l1 = np.sum(np.abs(b))
        l2 = np.dot(b, b)
        return float(self.lam * (float(self.l1_ratio) * l1 + 0.5 * (1.0 - float(self.l1_ratio)) * l2))

    def grad(self, beta: Array) -> Array:
        """
        Subgradient when l1_ratio > 0. Prefer prox() in optimization.
        """
        b = self._as1d(beta)
        g_l2 = self.lam * (1.0 - float(self.l1_ratio)) * b
        g_l1 = self.lam * float(self.l1_ratio) * np.sign(b)
        return g_l2 + g_l1

    @staticmethod
    def _soft_threshold(v: Array, t: float) -> Array:
        return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

    def prox(self, beta: Array, step: float) -> Array:
        v = self._as1d(beta)
        if step < 0:
            raise ValueError("step must be nonnegative.")

        if self.lam == 0.0 or step == 0.0:
            return v.copy()

        t = float(step) * float(self.lam) * float(self.l1_ratio)
        u = self._soft_threshold(v, t)

        denom = 1.0 + float(step) * float(self.lam) * (1.0 - float(self.l1_ratio))
        return u / denom

    def is_smooth(self) -> bool:
        return float(self.l1_ratio) == 0.0
