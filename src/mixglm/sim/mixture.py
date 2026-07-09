# src/mixglm/sim/mixture.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from mixglm.links.base import Link

Array = np.ndarray


@dataclass(frozen=True)
class SimComponent:
    """
    Defines how to simulate one component:
      eta_k = X @ beta_k
      loc_k = link_k.inverse(eta_k)
      y_i | z_i=k ~ component_sampler(loc_k[i], extra_k)
    """
    name: str
    beta: Array
    link: Link
    extra: Dict[str, Any]
    sampler: Any  # callable(loc: Array, rng: Generator, extra: dict) -> Array


@dataclass(frozen=True)
class MixtureSimResult:
    y: Array
    X: Array
    z: Array
    pi: Array
    comp_loc: List[Array]        # list of loc_k (n,)
    meta: Dict[str, Any]


def normalize_pi(pi: Array, min_pi: float = 1e-12) -> Array:
    pi = np.asarray(pi, dtype=float).copy()
    pi = np.clip(pi, min_pi, None)
    pi = pi / pi.sum()
    return pi


def sample_mixture(
    *,
    X: Array,
    components: Sequence[SimComponent],
    pi: Array,
    rng: np.random.Generator,
    meta: Optional[Dict[str, Any]] = None,
) -> MixtureSimResult:
    """
    Generic mixture sampler for y|X with component-specific regression on location.
    Mixing proportions pi are constant (for now).

    Returns y, z, and per-component locations.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    K = len(components)
    if K < 1:
        raise ValueError("Need at least 1 component")

    pi = normalize_pi(pi)
    if pi.shape != (K,):
        raise ValueError(f"pi must have shape ({K},)")

    # sample component labels
    z = rng.choice(K, size=n, p=pi)

    # compute locations for all components
    locs: List[Array] = []
    for comp in components:
        beta = np.asarray(comp.beta, dtype=float)
        if beta.shape[0] != X.shape[1]:
            raise ValueError(f"beta length {beta.shape[0]} does not match X columns {X.shape[1]}")
        eta = X @ beta
        loc = comp.link.inverse(eta)
        locs.append(np.asarray(loc, dtype=float))

    # sample y conditional on z
    y = np.empty(n, dtype=float)
    for k, comp in enumerate(components):
        idx = np.where(z == k)[0]
        if idx.size == 0:
            continue
        loc_k = locs[k][idx]
        y[idx] = comp.sampler(loc_k, rng=rng, extra=comp.extra)

    return MixtureSimResult(
        y=y,
        X=X,
        z=z,
        pi=pi,
        comp_loc=locs,
        meta=dict(meta or {}),
    )
