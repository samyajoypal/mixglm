# src/mixglm/links/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

Array = np.ndarray


class Link(Protocol):
    """
    Minimal link interface used throughout the project.

    We only need:
    - link(mu):   eta
    - inverse(eta): mu
    - inverse_deriv(eta): d/deta inverse(eta)

    Notes:
    - inverse_deriv is used by gradient-based optimizers (optional but strongly recommended).
    - All methods must support vectorized numpy arrays.
    """
    name: str

    def link(self, mu: Array) -> Array: ...
    def inverse(self, eta: Array) -> Array: ...
    def inverse_deriv(self, eta: Array) -> Array: ...


@dataclass(frozen=True)
class BaseLink:
    """
    Convenience base class for links.
    Concrete links should override link(), inverse(), inverse_deriv().

    Design:
    - Avoid any statsmodels dependency.
    - Keep numerically stable implementations in concrete links.
    """
    name: str

    def link(self, mu: Array) -> Array:
        raise NotImplementedError

    def inverse(self, eta: Array) -> Array:
        raise NotImplementedError

    def inverse_deriv(self, eta: Array) -> Array:
        raise NotImplementedError

    # ---- helpers ----
    @staticmethod
    def _as1d(x: Array, name: str) -> Array:
        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1)
        if x.ndim != 1:
            raise ValueError(f"{name} must be a 1D array.")
        return x

    def validate_mu(self, mu: Array) -> Array:
        mu = self._as1d(mu, "mu")
        if np.any(~np.isfinite(mu)):
            raise ValueError("mu contains non-finite values.")
        return mu

    def validate_eta(self, eta: Array) -> Array:
        eta = self._as1d(eta, "eta")
        if np.any(~np.isfinite(eta)):
            raise ValueError("eta contains non-finite values.")
        return eta
