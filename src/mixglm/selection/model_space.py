# src/mixglm/selection/model_space.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Dict
import numpy as np

from mixglm.families.registry import FAMILIES, register_defaults as register_family_defaults

Array = np.ndarray


@dataclass(frozen=True)
class ModelSpace:
    """
    Defines a candidate model space for selection.

    Typically built by:
    - restricting families by response support (real, positive, count, etc.)
    - optionally restricting to a shortlist from screening
    """
    family_names: List[str]

    def validate_registered(self) -> None:
        register_family_defaults()
        available = set(FAMILIES.available())
        missing = [f for f in self.family_names if f.lower() not in available]
        if missing:
            raise KeyError(f"Families not registered: {missing}. Available: {sorted(available)}")


def infer_support_kind(y: Array, tol: float = 1e-12) -> str:
    """
    Infer a coarse support kind from y:
      - "nonnegative_int" if y are integers and >= 0
      - "positive" if y > 0
      - "unit_interval" if y in [0,1]
      - otherwise "real"
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1D.")

    if np.all(y >= -tol) and np.all(np.abs(y - np.round(y)) <= tol):
        return "nonnegative_int"
    if np.all(y >= -tol) and np.all(y <= 1.0 + tol):
        return "unit_interval"
    if np.all(y > 0):
        return "positive"
    return "real"


def families_by_support(
    y: Array,
    *,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> ModelSpace:
    """
    Construct a ModelSpace by filtering registered families by their support.kind.
    """
    register_family_defaults()
    include = [s.lower() for s in include] if include is not None else None
    exclude_set = set(s.lower() for s in exclude) if exclude is not None else set()

    kind = infer_support_kind(y)
    fams = []
    for name in FAMILIES.available():
        if include is not None and name not in include:
            continue
        if name in exclude_set:
            continue
        fam = FAMILIES.create(name)
        if fam.support.kind == kind:
            fams.append(name)

    return ModelSpace(family_names=fams)
