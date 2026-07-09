# src/mixglm/links/log.py
from __future__ import annotations

import numpy as np
from mixglm.links.base import BaseLink, Array


class LogLink(BaseLink):
    """
    Log link:
        eta = log(mu)
        mu  = exp(eta)

    Used for positive mean parameters (mu > 0).
    We clip inputs for numerical stability.
    """
    def __init__(self, *, eps: float = 1e-12) -> None:
        super().__init__(name="log")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.eps = float(eps)

    def link(self, mu: Array) -> Array:
        mu = self.validate_mu(mu)
        if np.any(mu <= 0):
            raise ValueError("LogLink requires mu > 0.")
        return np.log(np.clip(mu, self.eps, None))

    def inverse(self, eta: Array) -> Array:
        eta = self.validate_eta(eta)
        # avoid overflow in exp for very large eta
        eta_clip = np.clip(eta, -700.0, 700.0)
        return np.exp(eta_clip)

    def inverse_deriv(self, eta: Array) -> Array:
        eta = self.validate_eta(eta)
        eta_clip = np.clip(eta, -700.0, 700.0)
        return np.exp(eta_clip)
