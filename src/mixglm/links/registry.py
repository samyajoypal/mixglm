# src/mixglm/links/registry.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Type

from mixglm.links.base import Link
from mixglm.links.identity import IdentityLink
from mixglm.links.log import LogLink
from mixglm.links.logit import LogitLink


class _LinkRegistry:
    def __init__(self) -> None:
        self._constructors: Dict[str, Callable[[], Link]] = {}

    def register(self, name: str, ctor: Callable[[], Link]) -> None:
        key = name.lower()
        self._constructors[key] = ctor

    def create(self, name: str) -> Link:
        key = name.lower()
        if key not in self._constructors:
            raise KeyError(f"Unknown link '{name}'. Available: {sorted(self._constructors.keys())}")
        return self._constructors[key]()

    def available(self) -> List[str]:
        return sorted(self._constructors.keys())


LINKS = _LinkRegistry()


def register_defaults() -> None:
    """
    Register built-in links. Safe to call multiple times.
    """
    LINKS.register("identity", lambda: IdentityLink())
    LINKS.register("log", lambda: LogLink())
    LINKS.register("logit", lambda: LogitLink())
