# src/mixglm/penalties/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, List

from mixglm.penalties.base import BasePenalty


@dataclass
class PenaltyRegistry:
    """
    Simple registry for penalties.

    Allows:
    - registering a penalty factory under a name
    - creating penalties by name with a given lambda (and optional params)
    """
    _factories: Dict[str, Callable[..., BasePenalty]]

    def __init__(self) -> None:
        self._factories = {}

    def register(self, name: str, factory: Callable[..., BasePenalty], *, overwrite: bool = False) -> None:
        key = str(name).lower()
        if (key in self._factories) and (not overwrite):
            raise KeyError(f"Penalty '{name}' already registered. Use overwrite=True to replace.")
        self._factories[key] = factory

    def create(self, name: str, **kwargs) -> BasePenalty:
        """
        Example:
            create("ridge", lam=1.0)
            create("elastic_net", lam=1.0, l1_ratio=0.5)
        """
        key = str(name).lower()
        if key not in self._factories:
            raise KeyError(f"Penalty '{name}' not registered. Available: {sorted(self._factories.keys())}")
        return self._factories[key](**kwargs)

    def available(self) -> List[str]:
        return sorted(self._factories.keys())


# Global registry instance
PENALTIES = PenaltyRegistry()


def register_defaults() -> None:
    """
    Register built-in penalties.
    Call once at startup (e.g. in selection routines).
    """
    from mixglm.penalties.base import NoPenalty
    from mixglm.penalties.ridge import RidgePenalty
    from mixglm.penalties.lasso import LassoPenalty
    from mixglm.penalties.elastic_net import ElasticNetPenalty

    PENALTIES.register("none", lambda lam=0.0: NoPenalty(lam=lam), overwrite=True)
    PENALTIES.register("ridge", lambda lam: RidgePenalty(lam=lam), overwrite=True)
    PENALTIES.register("lasso", lambda lam: LassoPenalty(lam=lam), overwrite=True)
    PENALTIES.register("elastic_net", lambda lam, l1_ratio=0.5: ElasticNetPenalty(lam=lam, l1_ratio=l1_ratio), overwrite=True)
