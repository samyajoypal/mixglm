# # src/mixglm/selection/full_pipeline.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Sequence, Tuple

# import numpy as np

# from mixglm.selection.beam_search import beam_search_models, CandidateModel
# from mixglm.selection.spaces import model_space
# from mixglm.selection.tuning import tune_lambda_ic, TuneRow
# from mixglm.model.component import ComponentSpec
# from mixglm.penalties.lasso import LassoPenalty
# from mixglm.penalties.ridge import RidgePenalty
# from mixglm.penalties.elastic_net import ElasticNetPenalty

# Array = np.ndarray

# def filter_families_by_y_support(candidate_families, y):
    # from mixglm.families.registry import FAMILIES
    # ok = []
    # bad = []
    # for name in candidate_families:
        # fam = FAMILIES.create(name)
        # try:
            # fam.support.validate_y(np.asarray(y))
            # ok.append(name)
        # except Exception as e:
            # bad.append((name, str(e)))
    # return ok, bad


# @dataclass
# class PipelineBest:
    # penalty: str
    # K: int
    # families: Tuple[str, ...]
    # criterion: str
    # score: float
    # model: Any  # MixtureGLM
    # tune_rows: Optional[List[TuneRow]] = None


# def _criterion_value(res, crit: str) -> float:
    # v = getattr(res, crit.lower())
    # if v is None:
        # raise RuntimeError(f"Result has {crit}=None. Ensure compute_icl=True if you select by ICL.")
    # return float(v)


# def full_beam_pipeline(
    # *,
    # y: Array,
    # X: Array,
    # kind: str = "continuous",              # "continuous" | "counts"
    # K_max: int = 3,
    # beam_width: int = 5,
    # criterion: str = "bic",                # "bic" | "icl" | "aic"
    # # penalties to consider
    # do_none: bool = True,
    # ridge_grid: Optional[Sequence[float]] = None,
    # lasso_grid: Optional[Sequence[float]] = None,
    # enet_grid: Optional[Sequence[Tuple[float, float]]] = None,  # (lam, l1_ratio)
    # # fit controls
    # em_kwargs: Optional[Dict[str, Any]] = None,
    # seed: int = 123,
    # init: str = "quantile",
    # compute_icl: bool = True,
    # standardize: bool = True,
    # verbose: bool = False,
    # parallel: ParallelConfig | None = None,
    # show_progress: bool = True,
# ) -> Tuple[PipelineBest, List[PipelineBest]]:
    # """
    # Full pipeline:
      # - choose candidate families from model_space(kind)
      # - run beam search over K and families
      # - for each penalty type: pick best lambda by IC (grid)
      # - compare best models across penalty types

    # Returns (best_overall, best_per_penalty).
    # """
    # em_kwargs = dict(em_kwargs or {})
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mixglm.selection.beam_search import beam_search_models, CandidateModel
from mixglm.selection.spaces import model_space
from mixglm.selection.tuning import tune_lambda_ic, TuneRow
from mixglm.model.component import ComponentSpec
from mixglm.penalties.lasso import LassoPenalty
from mixglm.penalties.ridge import RidgePenalty
from mixglm.penalties.elastic_net import ElasticNetPenalty
from mixglm.utils.parallel import ParallelConfig

Array = np.ndarray


def filter_families_by_y_support(
    candidate_families: Sequence[str],
    y: Array
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Drop families whose support is incompatible with observed y.
    Returns (kept_names, dropped[(name, reason)]).
    """
    from mixglm.families.registry import FAMILIES

    ok: List[str] = []
    bad: List[Tuple[str, str]] = []
    y = np.asarray(y)

    for name in candidate_families:
        fam = FAMILIES.create(name)
        try:
            fam.support.validate_y(y)
            ok.append(name)
        except Exception as e:
            bad.append((name, str(e)))
    return ok, bad


@dataclass
class PipelineBest:
    penalty: str
    K: int
    families: Tuple[str, ...]
    criterion: str
    score: float
    model: Any  # MixtureGLM
    tune_rows: Optional[List[TuneRow]] = None


def _criterion_value(res: Any, crit: str) -> float:
    """
    Extract criterion value from MixtureGLMResult.
    """
    v = getattr(res, crit.lower())
    if v is None:
        raise RuntimeError(f"Result has {crit}=None. Ensure compute_icl=True if you select by ICL.")
    return float(v)


def full_beam_pipeline(
    *,
    y: Array,
    X: Array,
    kind: str = "continuous",              # "continuous" | "counts"
    candidate_families: Optional[Sequence[str]] = None,
    K_max: int = 3,
    beam_width: int = 5,
    criterion: str = "bic",                # "bic" | "icl" | "aic"
    # penalties to consider
    do_none: bool = True,
    ridge_grid: Optional[Sequence[float]] = None,
    lasso_grid: Optional[Sequence[float]] = None,
    enet_grid: Optional[Sequence[Tuple[float, float]]] = None,  # (lam, l1_ratio)
    # fit controls
    em_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 123,
    init: str = "quantile",
    compute_icl: bool = True,
    standardize: bool = True,
    verbose: bool = False,
    parallel: Optional[ParallelConfig] = None,      # used for BEAM search
    show_progress: bool = True,
    tuning_n_jobs: int = 8,                         # used for tuning grids
) -> Tuple[PipelineBest, List[PipelineBest]]:
    """
    Full pipeline:
      1) Choose candidate families from model_space(kind) and screen by y-support
      2) Beam search over K and family tuples (structure search; by default uses no penalty)
      3) For each penalty type: tune lambda by IC (grid search) over a small set of structures
      4) Compare best models across penalty types

    Returns:
      (best_overall, best_per_penalty)
    """
    # ---------------- defaults ----------------
    em_kwargs = dict(em_kwargs or {})
    em_kwargs.setdefault("max_iter", 200)
    em_kwargs.setdefault("tol", 1e-6)
    em_kwargs.setdefault("n_starts", 5)
    em_kwargs.setdefault("init", init)
    em_kwargs.setdefault("compute_icl", compute_icl)

    y = np.asarray(y)
    X = np.asarray(X)

    # Parallel configs
    beam_parallel = parallel or ParallelConfig(n_jobs=1)
    tune_parallel = ParallelConfig(n_jobs=int(tuning_n_jobs), backend="loky", prefer="processes")

    # ---------------- candidate family space ----------------
    fam_space = model_space(kind)
    if candidate_families is None:
        candidate_families = list(fam_space.names)

    candidate_families, dropped = filter_families_by_y_support(candidate_families, y)

    if verbose:
        print(f"[support-screen] kept={len(candidate_families)}, dropped={len(dropped)}")
        if dropped:
            print("Dropped (first 10):")
            for n, msg in dropped[:10]:
                print(f"  - {n}: {msg}")

    if len(candidate_families) == 0:
        raise RuntimeError("No candidate families remain after y-support screening.")

    bests: List[PipelineBest] = []

    # Keep beam-search results (structure search) so we don't redo work
    cands_none: Optional[List[CandidateModel]] = None

    # ---------------- 1) No penalty ----------------
    if do_none:
        cands_none = beam_search_models(
            y=y, X=X,
            candidate_families=candidate_families,
            K_max=K_max,
            beam_width=beam_width,
            criterion=criterion,
            penalty_name="none",
            penalty_kwargs=None,
            seed=seed,
            init=init,
            compute_icl=compute_icl,
            standardize=standardize,
            verbose=verbose,
            max_iter=em_kwargs["max_iter"],
            tol=em_kwargs["tol"],
            n_starts=em_kwargs["n_starts"],
            parallel=beam_parallel,
            show_progress=show_progress,
        )
        if len(cands_none) == 0:
            raise RuntimeError("Beam search returned no candidates (none penalty).")

        best_c = cands_none[0]
        if best_c.result is None:
            raise RuntimeError("Best 'none' candidate has result=None. Something went wrong in beam search.")

        # Refit once as a proper MixtureGLM object (for unified downstream interface)
        from mixglm.model.mixture_glm import MixtureGLM
        from mixglm.families.registry import FAMILIES
        from mixglm.links.registry import LINKS
        from mixglm.penalties.base import NoPenalty

        comps: List[ComponentSpec] = []
        for fname in best_c.family_names:
            fam = FAMILIES.create(fname)
            link = LINKS.create(fam.default_link_name)
            comps.append(ComponentSpec(family=fam, link=link, penalty=NoPenalty()))

        mdl = MixtureGLM(comps).fit(
            y=y, X=X,
            seed=seed,
            standardize=standardize,
            **em_kwargs,
        )
        assert mdl.result_ is not None

        score = _criterion_value(mdl.result_, criterion)
        bests.append(PipelineBest(
            penalty="none",
            K=len(best_c.family_names),
            families=best_c.family_names,
            criterion=criterion,
            score=float(score),
            model=mdl,
            tune_rows=None,
        ))

    # ---------------- helper: beam search for structures ----------------
    def _beam_family_tuples() -> List[Tuple[str, ...]]:
        """
        Return up to `beam_width` distinct family-tuples (structures) to be used for penalty tuning.
        Prefer reusing the already-computed 'none' beam search if available.
        """
        nonlocal cands_none

        if cands_none is None:
            cands_none = beam_search_models(
                y=y, X=X,
                candidate_families=candidate_families,
                K_max=K_max,
                beam_width=beam_width,
                criterion=criterion,
                penalty_name="none",     # structure search without penalty
                penalty_kwargs=None,
                seed=seed,
                init=init,
                compute_icl=compute_icl,
                standardize=standardize,
                verbose=verbose,
                max_iter=em_kwargs["max_iter"],
                tol=em_kwargs["tol"],
                n_starts=em_kwargs["n_starts"],
                parallel=beam_parallel,
                show_progress=show_progress,
            )

        top_struct: List[Tuple[str, ...]] = []
        seen: set[Tuple[str, ...]] = set()

        for c in cands_none:
            if c.result is None:
                continue
            if c.family_names in seen:
                continue
            seen.add(c.family_names)
            top_struct.append(c.family_names)
            if len(top_struct) >= beam_width:
                break

        if len(top_struct) == 0:
            raise RuntimeError("No valid structure candidates found for tuning.")
        return top_struct

    struct_candidates = _beam_family_tuples()

    # ---------------- template builder (families/links fixed; penalty injected during tuning) ----------------
    def _make_template(fnames: Tuple[str, ...]) -> List[ComponentSpec]:
        from mixglm.families.registry import FAMILIES
        from mixglm.links.registry import LINKS
        from mixglm.penalties.base import NoPenalty

        out: List[ComponentSpec] = []
        for fname in fnames:
            fam = FAMILIES.create(fname)
            link = LINKS.create(fam.default_link_name)
            out.append(ComponentSpec(family=fam, link=link, penalty=NoPenalty()))
        return out

    # ---------------- 2) Ridge tuning ----------------
    if ridge_grid:
        best_model = None
        best_score = float("inf")
        best_info: Optional[Tuple[Tuple[str, ...], List[TuneRow]]] = None

        for fnames in struct_candidates:
            template = _make_template(fnames)
            mdl, rows = tune_lambda_ic(
                y=y, X=X,
                components_template=template,
                make_penalty=lambda lam: RidgePenalty(lam=float(lam)),
                lambda_grid=list(ridge_grid),
                criterion=criterion,
                seed=seed,
                em_kwargs=em_kwargs,
                standardize=standardize,
                parallel=tune_parallel,
                show_progress=show_progress,
                label=f"ridge K={len(fnames)}",
            )
            assert mdl.result_ is not None
            score = _criterion_value(mdl.result_, criterion)
            if score < best_score:
                best_score = score
                best_model = mdl
                best_info = (fnames, rows)

        assert best_model is not None and best_info is not None
        fnames, rows = best_info
        bests.append(PipelineBest(
            penalty="ridge",
            K=len(fnames),
            families=fnames,
            criterion=criterion,
            score=float(best_score),
            model=best_model,
            tune_rows=rows,
        ))

    # ---------------- 3) Lasso tuning ----------------
    if lasso_grid:
        best_model = None
        best_score = float("inf")
        best_info: Optional[Tuple[Tuple[str, ...], List[TuneRow]]] = None

        for fnames in struct_candidates:
            template = _make_template(fnames)
            mdl, rows = tune_lambda_ic(
                y=y, X=X,
                components_template=template,
                make_penalty=lambda lam: LassoPenalty(lam=float(lam)),
                lambda_grid=list(lasso_grid),
                criterion=criterion,
                seed=seed,
                em_kwargs=em_kwargs,
                standardize=standardize,
                parallel=tune_parallel,
                show_progress=show_progress,
                label=f"lasso K={len(fnames)}",
            )
            assert mdl.result_ is not None
            score = _criterion_value(mdl.result_, criterion)
            if score < best_score:
                best_score = score
                best_model = mdl
                best_info = (fnames, rows)

        assert best_model is not None and best_info is not None
        fnames, rows = best_info
        bests.append(PipelineBest(
            penalty="lasso",
            K=len(fnames),
            families=fnames,
            criterion=criterion,
            score=float(best_score),
            model=best_model,
            tune_rows=rows,
        ))

    # ---------------- 4) Elastic-net tuning ----------------
    if enet_grid:
        best_model = None
        best_score = float("inf")
        best_info: Optional[Tuple[Tuple[str, ...], List[TuneRow], float, float]] = None

        for fnames in struct_candidates:
            template = _make_template(fnames)

            # Each (lam, l1_ratio) is a point in the EN grid.
            for lam, l1_ratio in enet_grid:
                mdl, rows = tune_lambda_ic(
                    y=y, X=X,
                    components_template=template,
                    make_penalty=lambda _lam: ElasticNetPenalty(lam=float(lam), l1_ratio=float(l1_ratio)),
                    lambda_grid=[float(lam)],  # single point per (lam, l1_ratio)
                    criterion=criterion,
                    seed=seed,
                    em_kwargs=em_kwargs,
                    standardize=standardize,
                    parallel=ParallelConfig(n_jobs=1),  # single fit; no need to parallelize
                    show_progress=show_progress,
                    label=f"enet(lam={lam},r={l1_ratio}) K={len(fnames)}",
                )
                assert mdl.result_ is not None
                score = _criterion_value(mdl.result_, criterion)
                if score < best_score:
                    best_score = score
                    best_model = mdl
                    best_info = (fnames, rows, float(lam), float(l1_ratio))

        assert best_model is not None and best_info is not None
        fnames, rows, lam, l1_ratio = best_info
        bests.append(PipelineBest(
            penalty=f"enet(lam={lam},l1_ratio={l1_ratio})",
            K=len(fnames),
            families=fnames,
            criterion=criterion,
            score=float(best_score),
            model=best_model,
            tune_rows=rows,
        ))

    if len(bests) == 0:
        raise RuntimeError("Pipeline produced no models. Enable do_none or provide penalty grids.")

    # pick best overall (lower IC is better)
    bests.sort(key=lambda b: b.score)
    return bests[0], bests
