# src/mixglm/model/component.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from mixglm.families.base import UnivariateFamily
from mixglm.links.base import Link
from mixglm.penalties.base import BasePenalty, NoPenalty


@dataclass(frozen=True)
class ComponentSpec:
    """
    A single mixture component specification:
      - response family (distribution)
      - link mapping eta -> mu
      - penalty applied to regression coefficients for this component
    """
    family: UnivariateFamily
    link: Link
    penalty: BasePenalty = NoPenalty()
    coef_mask: Optional[Tuple[bool, ...]] = None

    def __post_init__(self) -> None:
        if self.family is None:
            raise ValueError("ComponentSpec.family must not be None.")
        if self.link is None:
            raise ValueError("ComponentSpec.link must not be None.")
        if self.penalty is None:
            raise ValueError("ComponentSpec.penalty must not be None.")
        if self.coef_mask is not None:
            mask = tuple(bool(x) for x in self.coef_mask)
            if not mask:
                raise ValueError("ComponentSpec.coef_mask must not be empty.")
            if not mask[0]:
                raise ValueError("ComponentSpec.coef_mask must keep the intercept free.")
            object.__setattr__(self, "coef_mask", mask)

    @property
    def name(self) -> str:
        return f"{self.family.name}|{self.link.name}|{self.penalty.name}"
