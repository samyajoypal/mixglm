# src/mixglm/selection/spaces.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

from mixglm.families.registry import FAMILIES, register_defaults


@dataclass(frozen=True)
class FamilySpace:
    """
    A named family space used for model selection.
    The 'names' are keys in the global family registry.
    """
    name: str
    names: List[str]


def model_space(kind: str, *, include: Optional[Sequence[str]] = None, exclude: Optional[Sequence[str]] = None) -> FamilySpace:
    """
    Build a coherent candidate-family list.

    kind:
      - "continuous": real-valued y
      - "counts": nonnegative integer-valued y (stored as floats in our code, but conceptually counts)

    include/exclude allow you to customize without rewriting code.
    """
    register_defaults()  # safe due to overwrite=True

    k = str(kind).lower().strip()
    if k in ("continuous", "real"):
        base = [
            "gaussian",
            "student_t",
            "lognormal",
            "gamma",
            "inverse_gaussian",
            "exponential",
            "skew_normal",
            "genhyperbolic",
            "jf_skew_t",
            "azzalini_skew_t",
        ]
    elif k in ("counts", "count", "discrete"):
        base = [
            "poisson",
            "nb2",
            "bernoulli",
            "geometric",
            "zip",
            "zinb",
        ]
    else:
        raise ValueError("kind must be one of: 'continuous', 'counts'.")

    # apply exclude/include
    names = list(dict.fromkeys(base))  # unique, preserve order

    if exclude:
        ex = {str(x).lower() for x in exclude}
        names = [n for n in names if n.lower() not in ex]

    if include:
        inc = [str(x).lower() for x in include]
        # add only if registered
        avail = set(FAMILIES.available())
        for n in inc:
            if n not in avail:
                raise KeyError(f"Requested include family '{n}' is not registered. Available: {sorted(avail)}")
        names += [n for n in inc if n not in names]

    # final sanity: all must be registered
    avail = set(FAMILIES.available())
    missing = [n for n in names if n not in avail]
    if missing:
        raise KeyError(f"Some families in model space are not registered: {missing}. Registered: {sorted(avail)}")

    return FamilySpace(name=k, names=names)
