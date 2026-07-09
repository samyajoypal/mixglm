from __future__ import annotations
import numpy as np
from mixglm.links.base import BaseLink, Array

class LogitLink(BaseLink):
    """
    Logit link:
        eta = log(mu / (1 - mu))
        mu  = 1 / (1 + exp(-eta))
    Used for probabilities (mu in (0, 1)).
    """
    def __init__(self, *, eps: float = 1e-12) -> None:
        super().__init__(name="logit")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.eps = float(eps)

    def link(self, mu: Array) -> Array:
        mu = self.validate_mu(mu)
        mu = np.clip(mu, self.eps, 1.0 - self.eps)
        return np.log(mu / (1.0 - mu))

    def inverse(self, eta: Array) -> Array:
        eta = self.validate_eta(eta)
        eta_clip = np.clip(eta, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-eta_clip))

    def inverse_deriv(self, eta: Array) -> Array:
        eta = self.validate_eta(eta)
        eta_clip = np.clip(eta, -700.0, 700.0)
        p = 1.0 / (1.0 + np.exp(-eta_clip))
        return p * (1.0 - p)
