# src/mixglm/em/stopping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class StopState:
    """
    Tracks convergence for EM/GEM.
    """
    tol: float = 1e-6
    max_iter: int = 200
    min_iter: int = 1

    last_obj: Optional[float] = None
    n_iter: int = 0
    converged: bool = False

    def update(self, obj_value: float) -> bool:
        """
        Update with the new objective value and return True if we should stop.
        Uses relative improvement criterion.

        rel = |obj_t - obj_{t-1}| / (1 + |obj_{t-1}|)
        """
        self.n_iter += 1

        if self.last_obj is None:
            self.last_obj = float(obj_value)
            return False

        prev = float(self.last_obj)
        curr = float(obj_value)
        rel = abs(curr - prev) / (1.0 + abs(prev))

        self.last_obj = curr

        if self.n_iter < self.min_iter:
            return False

        if rel < self.tol:
            self.converged = True
            return True

        if self.n_iter >= self.max_iter:
            self.converged = False
            return True

        return False

    def info(self) -> Dict[str, Any]:
        return {
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "tol": float(self.tol),
            "max_iter": int(self.max_iter),
        }
