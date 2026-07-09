# src/mixglm/families/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Protocol, Any
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class FamilySupport:
    """Support constraints for the response y."""
    kind: str  # "real", "positive", "nonnegative_int", "unit_interval", "bounded"
    lower: Optional[float] = None
    upper: Optional[float] = None

    def validate_y(self, y: Array) -> None:
        y = np.asarray(y)
        if self.kind == "real":
            return
        if self.kind == "positive":
            if np.any(y <= 0):
                raise ValueError("Family support is (0, inf) but y contains non-positive values.")
            return
        if self.kind == "nonnegative_int":
            if np.any(y < 0) or np.any(np.floor(y) != y):
                raise ValueError("Family support is {0,1,2,...} but y contains negatives or non-integers.")
            return
        if self.kind == "unit_interval":
            if np.any(y < 0) or np.any(y > 1):
                raise ValueError("Family support is [0,1] but y is outside [0,1].")
            return
        if self.kind == "bounded":
            if self.lower is None or self.upper is None:
                raise ValueError("Bounded support requires lower and upper.")
            if np.any(y < self.lower) or np.any(y > self.upper):
                raise ValueError(f"Family support is [{self.lower},{self.upper}] but y is outside bounds.")
            return
        raise ValueError(f"Unknown support kind: {self.kind}")


class LinkLike(Protocol):
    """Minimal link interface expected by families."""
    name: str

    def inverse(self, eta: Array) -> Array: ...
    def inverse_deriv(self, eta: Array) -> Array: ...


@dataclass
class FamilyParams:
    """
    Container for parameters used by a family at evaluation time.

    mu: location/mean-like parameter after link inverse.
    extra: nuisance parameters (component-specific), independent of x in the main paper.
    """
    mu: Array
    extra: Dict[str, Any]


class UnivariateFamily:
    """
    Abstract base class for a univariate response distribution family.

    Design goals:
    - Stable log-density/pmf evaluation with broadcasting.
    - Explicit support checks for screening and safer fitting.
    - Family-specific nuisance parameters handled via a dictionary.
    - Optional parameter transforms to enforce constraints (positive scale, df > 2, etc).

    Each concrete family should implement:
      - support
      - default_link_name
      - num_extra_params / extra_param_names
      - logpdf_or_logpmf
      - initialize_extra (reasonable starting values from y and responsibilities)
      - bounds_extra (box constraints in unconstrained or constrained space)
      - transform_extra / inverse_transform_extra (optional, defaults identity)
    """

    # ---------- required metadata ----------
    name: str = "base"
    is_discrete: bool = False

    @property
    def support(self) -> FamilySupport:
        raise NotImplementedError

    @property
    def default_link_name(self) -> str:
        raise NotImplementedError

    # ---------- nuisance parameter interface ----------
    @property
    def extra_param_names(self) -> Tuple[str, ...]:
        """Names of nuisance parameters in extra dict, fixed across observations."""
        return tuple()

    def num_extra_params(self) -> int:
        return len(self.extra_param_names)

    def initialize_extra(self, y: Array, tau_k: Array) -> Dict[str, Any]:
        """
        Initialize nuisance parameters for a component using data y and responsibilities tau_k.

        tau_k is shape (n,). This method should return a dict keyed by extra_param_names.
        """
        return {}

    def bounds_extra(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """
        Box bounds for nuisance parameters in the *transformed* space (see transform_extra).

        Use (None, None) for unbounded. Prefer weak but safe bounds to prevent degeneracy.
        """
        return {name: (None, None) for name in self.extra_param_names}

    def transform_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map constrained nuisance parameters to an unconstrained representation for optimizers.
        Default is identity (parameters are already unconstrained).
        """
        return dict(extra)

    def inverse_transform_extra(self, extra_t: Dict[str, Any]) -> Dict[str, Any]:
        """Inverse of transform_extra."""
        return dict(extra_t)

    # ---------- core evaluation ----------
    def _validate_inputs(self, y: Array, mu: Array) -> Tuple[Array, Array]:
        y = np.asarray(y)
        mu = np.asarray(mu)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n,).")
        if mu.ndim != 1:
            raise ValueError("mu must be a 1D array of shape (n,).")
        if y.shape[0] != mu.shape[0]:
            raise ValueError("y and mu must have the same length.")
        self.support.validate_y(y)
        return y, mu

    def logpdf_or_logpmf(self, y: Array, params: FamilyParams) -> Array:
        """
        Return log f(y_i; mu_i, extra) for i=1..n as a 1D array.

        Concrete families must implement this.
        """
        raise NotImplementedError

    def loglik_component(self, y: Array, mu: Array, extra: Dict[str, Any]) -> Array:
        """
        Convenience wrapper for log-density/pmf evaluation.
        """
        y, mu = self._validate_inputs(y, mu)
        params = FamilyParams(mu=mu, extra=extra)
        ll = self.logpdf_or_logpmf(y, params)
        ll = np.asarray(ll)
        if ll.shape != y.shape:
            raise ValueError("logpdf_or_logpmf must return a 1D array of shape (n,).")
        return ll

    def mean_from_mu(self, mu: Array, extra: Dict[str, Any]) -> Array:
        """
        Conditional component mean E[Y | mu, extra].

        For most implemented families, the GLM location parameter ``mu`` is already
        the conditional mean. Families whose GLM location is not the mean, such as
        lognormal or zero-inflated counts, override this method.
        """
        return np.asarray(mu, dtype=float)

    # ---------- utilities used by EM and selection ----------
    def component_nll(
        self,
        y: Array,
        mu: Array,
        extra: Dict[str, Any],
        weights: Optional[Array] = None,
    ) -> float:
        """
        Weighted negative log-likelihood for a single component:
            -sum_i w_i log f(y_i; mu_i, extra)
        """
        ll = self.loglik_component(y, mu, extra)
        if weights is None:
            return float(-np.sum(ll))
        w = np.asarray(weights)
        if w.shape != ll.shape:
            raise ValueError("weights must be shape (n,).")
        return float(-np.sum(w * ll))

    def safe_logsumexp(self, A: Array, axis: int = 1) -> Array:
        """
        Numerically stable log-sum-exp along axis.
        A is typically shape (n, K) containing log(pi_k) + log f_k.
        """
        A = np.asarray(A)
        m = np.max(A, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)

    def validate_extra(self, extra: Dict[str, Any]) -> None:
        """
        Sanity check for nuisance parameters.
        Concrete families can override to enforce constraints.
        """
        for name in self.extra_param_names:
            if name not in extra:
                raise ValueError(f"Missing nuisance parameter '{name}' for family '{self.name}'.")

    def describe(self) -> str:
        """Human-readable summary for logging and model selection traces."""
        if self.num_extra_params() == 0:
            return f"{self.name} (no extra params)"
        return f"{self.name} (extra: {', '.join(self.extra_param_names)})"
