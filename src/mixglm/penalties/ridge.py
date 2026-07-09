# src/mixglm/penalties/ridge.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mixglm.penalties.base import BasePenalty, Array


@dataclass(frozen=True)
class RidgePenalty(BasePenalty):
    """
    Ridge penalty:
        P(beta) = (lam/2) * ||beta||_2^2

    Prox:
        prox_{step * P}(beta) = beta / (1 + step * lam)
    """
    lam: float = 1.0

    @property
    def name(self) -> str:
        return "ridge"

    def value(self, beta: Array) -> float:
        b = self._as1d(beta)
        return float(0.5 * self.lam * np.dot(b, b))

    def grad(self, beta: Array) -> Array:
        b = self._as1d(beta)
        return self.lam * b

    def prox(self, beta: Array, step: float) -> Array:
        b = self._as1d(beta)
        if step < 0:
            raise ValueError("step must be nonnegative.")
        denom = 1.0 + step * self.lam
        return b / denom

    def is_smooth(self) -> bool:
        return True
