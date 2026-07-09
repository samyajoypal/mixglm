# src/mixglm/links/identity.py
from __future__ import annotations

import numpy as np
from mixglm.links.base import BaseLink, Array


class IdentityLink(BaseLink):
    """
    Identity link:
        eta = mu
        mu = eta
    """
    def __init__(self) -> None:
        super().__init__(name="identity")

    def link(self, mu: Array) -> Array:
        mu = self.validate_mu(mu)
        return mu

    def inverse(self, eta: Array) -> Array:
        eta = self.validate_eta(eta)
        return eta

    def inverse_deriv(self, eta: Array) -> Array:
        eta = self.validate_eta(eta)
        return np.ones_like(eta, dtype=float)
