# # src/mixglm/penalties/base.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Protocol, Optional
# import numpy as np

# Array = np.ndarray


# class Penalty(Protocol):
    # """
    # Minimal penalty interface.

    # We support both smooth (ridge) and non-smooth (lasso/elastic-net) penalties.

    # Required:
    # - value(beta): penalty value P(beta)
    # - grad(beta): gradient (or subgradient) if available; may raise NotImplementedError
    # - prox(beta, step): proximal operator for step size 'step' (required for lasso/EN)

    # Notes:
    # - For ridge, prox is available in closed form.
    # - For pure smooth penalties, prox can still be implemented (identity if step=0).
    # - For lasso/EN, grad is not a true gradient everywhere; use prox in optimizers.
    # """
    # name: str
    # lam: float

    # def value(self, beta: Array) -> float: ...
    # def grad(self, beta: Array) -> Array: ...
    # def prox(self, beta: Array, step: float) -> Array: ...
    # def is_smooth(self) -> bool: ...


# @dataclass(frozen=True)
# class BasePenalty:
    # """
    # Base class with common validation and safe defaults.

    # Concrete penalties should override:
    # - value
    # - prox
    # - grad (optional, but recommended for smooth penalties)
    # - is_smooth
    # """
    # name: str
    # lam: float

    # def __post_init__(self) -> None:
        # if self.lam < 0:
            # raise ValueError("Penalty parameter lam must be nonnegative.")

    # @staticmethod
    # def _as1d(beta: Array) -> Array:
        # beta = np.asarray(beta, dtype=float)
        # if beta.ndim == 0:
            # beta = beta.reshape(1)
        # if beta.ndim != 1:
            # raise ValueError("beta must be a 1D array.")
        # if np.any(~np.isfinite(beta)):
            # raise ValueError("beta contains non-finite values.")
        # return beta

    # def value(self, beta: Array) -> float:
        # raise NotImplementedError

    # def grad(self, beta: Array) -> Array:
        # """
        # Gradient for smooth penalties.
        # For non-smooth penalties, optimizers should use prox instead of grad.
        # """
        # raise NotImplementedError

    # def prox(self, beta: Array, step: float) -> Array:
        # """
        # Proximal operator:
            # prox_{step * P}(beta) = argmin_u 0.5||u-beta||^2 + step * P(u)

        # Must be implemented for lasso/elastic-net (and ridge too, for uniformity).
        # """
        # raise NotImplementedError

    # def is_smooth(self) -> bool:
        # """Return True for differentiable penalties (e.g. ridge), False for lasso/EN."""
        # return False


# @dataclass(frozen=True)
# class NoPenalty(BasePenalty):
    # """
    # Convenience 'no penalty' option (lam ignored).
    # """
    # name: str = "none"
    # lam: float = 0.0

    # def value(self, beta: Array) -> float:
        # beta = self._as1d(beta)
        # return 0.0

    # def grad(self, beta: Array) -> Array:
        # beta = self._as1d(beta)
        # return np.zeros_like(beta)

    # def prox(self, beta: Array, step: float) -> Array:
        # beta = self._as1d(beta)
        # return beta.copy()

    # def is_smooth(self) -> bool:
        # return True

# src/mixglm/penalties/base.py
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class BasePenalty(ABC):
    """
    Base class for penalties P(beta).

    Concrete penalties must implement:
      - name (property)
      - value(beta)
      - prox(beta, step)

    Optional:
      - grad(beta) for smooth penalties
      - is_smooth()
    """
    lam: float = 0.0

    def __post_init__(self) -> None:
        lam = float(self.lam)
        if not np.isfinite(lam):
            raise ValueError("lam must be finite.")
        if lam < 0.0:
            raise ValueError("lam must be nonnegative.")
        object.__setattr__(self, "lam", lam)

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def _as1d(self, beta: Array) -> Array:
        b = np.asarray(beta, dtype=float)
        if b.ndim != 1:
            b = b.reshape(-1)
        return b

    @abstractmethod
    def value(self, beta: Array) -> float:
        raise NotImplementedError

    def grad(self, beta: Array) -> Array:
        raise NotImplementedError("grad not implemented for this penalty (use prox).")

    @abstractmethod
    def prox(self, beta: Array, step: float) -> Array:
        raise NotImplementedError

    def is_smooth(self) -> bool:
        return False


@dataclass(frozen=True)
class NoPenalty(BasePenalty):
    lam: float = 0.0

    @property
    def name(self) -> str:
        return "none"

    def value(self, beta: Array) -> float:
        return 0.0

    def grad(self, beta: Array) -> Array:
        b = self._as1d(beta)
        return np.zeros_like(b)

    def prox(self, beta: Array, step: float) -> Array:
        b = self._as1d(beta)
        return b.copy()

    def is_smooth(self) -> bool:
        return True
