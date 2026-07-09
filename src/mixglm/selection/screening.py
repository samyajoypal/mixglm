# src/mixglm/selection/screening.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np

from mixglm.model.mixture_glm import MixtureGLM, ComponentSpec
from mixglm.families.registry import FAMILIES, register_defaults as register_family_defaults
from mixglm.links.registry import LINKS, register_defaults as register_link_defaults
from mixglm.penalties.registry import PENALTIES, register_defaults as register_penalty_defaults
from mixglm.penalties.base import NoPenalty
from mixglm.selection.criteria import evaluate_criteria, InfoCriteria
from mixglm.utils.logging import Logger

Array = np.ndarray


@dataclass
class ScreeningResult:
    """
    Result of screening a set of candidate families with K=1.
    """
    family_name: str
    link_name: str
    criterion: str
    ic: InfoCriteria
    model: MixtureGLM  # fitted K=1 model


def screen_families_k1(
    *,
    y: Array,
    X: Array,
    family_names: Sequence[str],
    link_name: Optional[str] = None,
    penalty_name: str = "none",
    penalty_kwargs: Optional[Dict[str, Any]] = None,
    criterion: str = "bic",
    top_m: int = 5,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_starts: int = 3,
    seed: Optional[int] = None,
    init: str = "quantile",
    verbose: bool = False,
) -> List[ScreeningResult]:
    """
    Screen candidate families by fitting K=1 models and ranking by an information criterion.

    Parameters
    ----------
    family_names : list of names registered in FAMILIES
    link_name : if None, uses each family's default_link_name
    penalty_name : "none" recommended for screening; but you can also use ridge, etc.
    criterion : "bic" or "aic" (or "icl", though ICL is not meaningful for K=1)
    top_m : return top M families
    """
    # ensure defaults registered (safe if called multiple times with overwrite=True)
    register_family_defaults()
    register_link_defaults()
    register_penalty_defaults()

    logger = Logger(verbose=verbose)

    y = np.asarray(y)
    X = np.asarray(X)

    penalty_kwargs = {} if penalty_kwargs is None else dict(penalty_kwargs)

    results: List[ScreeningResult] = []
    logger.section("K=1 screening")

    for fname in family_names:
        fam = FAMILIES.create(fname)
        lnk_name = link_name or fam.default_link_name
        link = LINKS.create(lnk_name)

        pen = PENALTIES.create(penalty_name, **({"lam": 0.0} | penalty_kwargs)) if penalty_name != "none" else NoPenalty()

        comp = ComponentSpec(family=fam, link=link, penalty=pen)
        mdl = MixtureGLM([comp]).fit(
            y, X,
            max_iter=max_iter,
            tol=tol,
            n_starts=n_starts,
            seed=seed,
            init=init,
            verbose=False,  # keep internal EM quiet; we log summary here
        )

        ic = evaluate_criteria(
            loglik=mdl.result_.loglik,  # type: ignore[union-attr]
            X=X,
            components=mdl.components,
            responsibilities=mdl.result_.responsibilities,  # type: ignore[union-attr]
            compute_icl_flag=False,
        )

        key = criterion.lower()
        if key not in {"bic", "aic"}:
            raise ValueError("criterion must be one of: 'bic', 'aic' for K=1 screening.")

        logger.log(f"family={fname:>12s} link={lnk_name:>8s} loglik={ic.loglik:.3f} aic={ic.aic:.3f} bic={ic.bic:.3f}")

        results.append(
            ScreeningResult(
                family_name=fname,
                link_name=lnk_name,
                criterion=key,
                ic=ic,
                model=mdl,
            )
        )

    # rank
    if criterion.lower() == "bic":
        results.sort(key=lambda r: r.ic.bic)
    else:
        results.sort(key=lambda r: r.ic.aic)

    return results[: min(top_m, len(results))]
