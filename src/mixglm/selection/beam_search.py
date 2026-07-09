# # src/mixglm/selection/beam_search.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import List, Optional, Sequence, Dict, Any, Tuple

# import numpy as np

# from mixglm.model.mixture_glm import MixtureGLM, ComponentSpec, MixtureGLMResult
# from mixglm.families.registry import FAMILIES, register_defaults as register_family_defaults
# from mixglm.links.registry import LINKS, register_defaults as register_link_defaults
# from mixglm.penalties.registry import PENALTIES, register_defaults as register_penalty_defaults
# from mixglm.selection.criteria import evaluate_criteria, InfoCriteria
# from mixglm.utils.logging import Logger
# from mixglm.penalties.base import NoPenalty

# Array = np.ndarray


# @dataclass
# class CandidateModel:
    # """
    # Represents a fitted candidate model during beam search.
    # """
    # K: int
    # family_names: Tuple[str, ...]
    # link_names: Tuple[str, ...]
    # penalty_names: Tuple[str, ...]
    # penalty_kwargs: Tuple[Tuple[Tuple[str, Any], ...], ...]  # per component, hashable
    # result: MixtureGLMResult
    # ic: InfoCriteria

    # def score(self, criterion: str) -> float:
        # c = criterion.lower()
        # if c == "bic":
            # return self.ic.bic
        # if c == "aic":
            # return self.ic.aic
        # if c == "icl":
            # if self.ic.icl is None:
                # raise ValueError("ICL not available for this candidate.")
            # return self.ic.icl
        # raise ValueError("criterion must be one of: 'bic', 'aic', 'icl'.")


# def _freeze_kwargs(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    # return tuple(sorted(d.items(), key=lambda kv: kv[0]))


# def beam_search_models(
    # *,
    # y: Array,
    # X: Array,
    # candidate_families: Sequence[str],
    # K_max: int,
    # beam_width: int = 5,
    # criterion: str = "bic",
    # # per-component default link choice: None -> use family's default
    # default_link_name: Optional[str] = None,
    # # penalties
    # penalty_name: str = "none",
    # penalty_kwargs: Optional[Dict[str, Any]] = None,
    # # fitting controls
    # max_iter: int = 200,
    # tol: float = 1e-6,
    # n_starts: int = 5,
    # seed: Optional[int] = None,
    # init: str = "quantile",
    # verbose: bool = False,
    # compute_icl: bool = True,
# ) -> List[CandidateModel]:
    # """
    # Constructive model search over (K, component families) using a beam search strategy.

    # Strategy:
    # - Start with all K=1 models from candidate_families, keep best `beam_width`.
    # - For K=2..K_max:
        # Expand each retained model by adding one component with each candidate family.
        # Fit each expanded model, score by criterion, keep best `beam_width`.

    # Notes:
    # - This searches ordered component tuples (label switching later). For practical use,
      # this is acceptable; you can post-process by sorting components or merging duplicates.
    # """
    # register_family_defaults()
    # register_link_defaults()
    # register_penalty_defaults()

    # logger = Logger(verbose=verbose)
    # penalty_kwargs = {} if penalty_kwargs is None else dict(penalty_kwargs)

    # y = np.asarray(y)
    # X = np.asarray(X)

    # if K_max < 1:
        # raise ValueError("K_max must be >= 1.")
    # if beam_width < 1:
        # raise ValueError("beam_width must be >= 1.")
    # if len(candidate_families) == 0:
        # raise ValueError("candidate_families must not be empty.")

    # all_levels: List[List[CandidateModel]] = []

    # def fit_model(fnames: Tuple[str, ...]) -> CandidateModel:
        # try:
            # comps: List[ComponentSpec] = []
            # link_names: List[str] = []
            # penalty_names: List[str] = []
            # pkws: List[Tuple[Tuple[str, Any], ...]] = []

            # for fname in fnames:
                # fam = FAMILIES.create(fname)
                # lnk_name = default_link_name or fam.default_link_name
                # link = LINKS.create(lnk_name)

                # if penalty_name.lower() == "none":
                    # pen = NoPenalty()
                    # pkw = _freeze_kwargs({"lam": 0.0})
                # else:
                    # # enforce lam in kwargs if not provided
                    # if "lam" not in penalty_kwargs:
                        # raise ValueError("penalty_kwargs must include 'lam' when penalty_name != 'none'.")
                    # pen = PENALTIES.create(penalty_name, **penalty_kwargs)
                    # pkw = _freeze_kwargs(penalty_kwargs)

                # comps.append(ComponentSpec(family=fam, link=link, penalty=pen))
                # link_names.append(lnk_name)
                # penalty_names.append(penalty_name.lower())
                # pkws.append(pkw)

            # mdl = MixtureGLM(comps).fit(
                # y, X,
                # max_iter=max_iter,
                # tol=tol,
                # n_starts=n_starts,
                # seed=seed,
                # init=init,
                # verbose=False,
            # )
            # res = mdl.result_
            # assert res is not None

            # ic = evaluate_criteria(
                # loglik=res.loglik,
                # X=X,
                # components=mdl.components,
                # responsibilities=res.responsibilities,
                # compute_icl_flag=compute_icl,
            # )

            # return CandidateModel(
                # K=len(fnames),
                # family_names=fnames,
                # link_names=tuple(link_names),
                # penalty_names=tuple(penalty_names),
                # penalty_kwargs=tuple(pkws),
                # result=res,
                # ic=ic,
            # )
        # except Exception as e:
            # # return a "failed" candidate (score=inf) so caller can drop it
            # return CandidateModel(
                # family_names=tuple(fnames),
                # criterion=criterion,
                # score=float("inf"),
                # converged=False,
                # result=None,
                # error=str(e)[:160],
            # )

    # # ----- Level K=1 -----
    # logger.section("Beam search: K=1")
    # level1: List[CandidateModel] = []
    # for fname in candidate_families:
        # cand = fit_model((fname,))
        # logger.log(f"K=1 families={cand.family_names} | loglik={cand.ic.loglik:.3f} | aic={cand.ic.aic:.3f} | bic={cand.ic.bic:.3f}" +
                   # (f" | icl={cand.ic.icl:.3f}" if cand.ic.icl is not None else ""))
        # level1.append(cand)

    # level1.sort(key=lambda c: c.score(criterion))
    # level1 = level1[:beam_width]
    # all_levels.append(level1)

    # # ----- Expand K=2..K_max -----
    # for K in range(2, K_max + 1):
        # logger.section(f"Beam search: K={K}")
        # prev = all_levels[-1]
        # expanded: List[CandidateModel] = []

        # for base in prev:
            # for fname in candidate_families:
                # fnames = base.family_names + (fname,)
                # cand = fit_model(fnames)
                # logger.log(
                    # f"K={K} families={cand.family_names} | score({criterion})={cand.score(criterion):.3f}"
                # )
                # expanded.append(cand)

        # expanded.sort(key=lambda c: c.score(criterion))
        # expanded = expanded[:beam_width]
        # all_levels.append(expanded)

    # # flatten all candidates across K for convenience
    # flat: List[CandidateModel] = [c for level in all_levels for c in level]
    # flat.sort(key=lambda c: c.score(criterion))
    # return flat





# src/mixglm/selection/beam_search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any, Tuple

import time
import numpy as np

from mixglm.model.mixture_glm import MixtureGLM, ComponentSpec, MixtureGLMResult
from mixglm.families.registry import FAMILIES, register_defaults as register_family_defaults
from mixglm.links.registry import LINKS, register_defaults as register_link_defaults
from mixglm.penalties.registry import PENALTIES, register_defaults as register_penalty_defaults
from mixglm.selection.criteria import evaluate_criteria, InfoCriteria
from mixglm.utils.logging import Logger
from mixglm.utils.parallel import ParallelConfig, parallel_map
from mixglm.penalties.base import NoPenalty

Array = np.ndarray


@dataclass
class CandidateModel:
    """
    Represents a fitted candidate model during beam search.

    Notes:
    - If success=False, result/ic may be None and score() returns +inf.
    """
    K: int
    family_names: Tuple[str, ...]
    link_names: Tuple[str, ...]
    penalty_names: Tuple[str, ...]
    penalty_kwargs: Tuple[Tuple[Tuple[str, Any], ...], ...]  # per component, hashable

    success: bool
    result: Optional[MixtureGLMResult]
    ic: Optional[InfoCriteria]
    error: Optional[str] = None
    model: Optional[Any] = None

    def score(self, criterion: str) -> float:
        if (not self.success) or (self.ic is None):
            return float("inf")
        c = criterion.lower()
        if c == "bic":
            return float(self.ic.bic)
        if c == "aic":
            return float(self.ic.aic)
        if c == "icl":
            if self.ic.icl is None:
                return float("inf")
            return float(self.ic.icl)
        raise ValueError("criterion must be one of: 'bic', 'aic', 'icl'.")


def _freeze_kwargs(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(d.items(), key=lambda kv: kv[0]))


def _stable_model_seed(seed: Optional[int], key: Tuple[Any, ...]) -> Optional[int]:
    """
    Create a deterministic per-model seed so parallel workers do not repeat the same init.
    """
    if seed is None:
        return None
    # stable hash -> 32-bit
    h = hash(key) & 0xFFFFFFFF
    return int((int(seed) + h) & 0xFFFFFFFF)


def beam_search_models(
    *,
    y: Array,
    X: Array,
    candidate_families: Sequence[str],
    K_max: int,
    beam_width: int = 5,
    criterion: str = "bic",
    # per-component default link choice: None -> use family's default
    default_link_name: Optional[str] = None,
    # penalties
    penalty_name: str = "none",
    penalty_kwargs: Optional[Dict[str, Any]] = None,
    # fitting controls
    max_iter: int = 200,
    tol: float = 1e-6,
    n_starts: int = 5,
    seed: Optional[int] = None,
    init: str = "quantile",
    verbose: bool = False,
    compute_icl: bool = True,
    standardize: bool = True,
    # parallel + progress
    parallel: Optional[ParallelConfig] = None,
    show_progress: bool = True,
) -> List[CandidateModel]:
    """
    Constructive model search over (K, component families) using a beam search strategy.

    Strategy:
    - Start with all K=1 models from candidate_families, keep best `beam_width`.
    - For K=2..K_max:
        Expand each retained model by adding one component with each candidate family.
        Fit each expanded model, score by criterion, keep best `beam_width`.

    Parallelization:
    - Each beam level fits a batch of candidate models; we run that batch in parallel via joblib
      if available and parallel.n_jobs != 1.

    Robustness:
    - Any candidate fit failure is captured and returned as success=False; beam continues.
    """
    register_family_defaults()
    register_link_defaults()
    register_penalty_defaults()

    logger = Logger(verbose=verbose)
    parallel = parallel or ParallelConfig(n_jobs=1)

    penalty_kwargs = {} if penalty_kwargs is None else dict(penalty_kwargs)

    y = np.asarray(y)
    X = np.asarray(X)

    if K_max < 1:
        raise ValueError("K_max must be >= 1.")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1.")
    if len(candidate_families) == 0:
        raise ValueError("candidate_families must not be empty.")

    def fit_model(fnames: Tuple[str, ...]) -> CandidateModel:
        comps: List[ComponentSpec] = []
        link_names: List[str] = []
        penalty_names: List[str] = []
        pkws: List[Tuple[Tuple[str, Any], ...]] = []

        try:
            for fname in fnames:
                fam = FAMILIES.create(fname)
                lnk_name = default_link_name or fam.default_link_name
                link = LINKS.create(lnk_name)

                if penalty_name.lower() == "none":
                    pen = NoPenalty()
                    pkw = _freeze_kwargs({"lam": 0.0})
                    pname = "none"
                else:
                    if "lam" not in penalty_kwargs:
                        raise ValueError("penalty_kwargs must include 'lam' when penalty_name != 'none'.")
                    pen = PENALTIES.create(penalty_name, **penalty_kwargs)
                    pkw = _freeze_kwargs(penalty_kwargs)
                    pname = str(penalty_name).lower()

                comps.append(ComponentSpec(family=fam, link=link, penalty=pen))
                link_names.append(str(lnk_name).lower())
                penalty_names.append(pname)
                pkws.append(pkw)

            # per-model seed for reproducibility without identical restarts across parallel jobs
            model_seed = _stable_model_seed(
                seed,
                key=(fnames, tuple(link_names), tuple(penalty_names), tuple(pkws), init, n_starts, max_iter, tol),
            )

            mdl = MixtureGLM(comps).fit(
                y, X,
                max_iter=max_iter,
                tol=tol,
                n_starts=n_starts,
                seed=model_seed,
                init=init,
                verbose=False,          # do not spam EM output
                compute_icl=compute_icl,
                standardize=standardize,
            )
            res = mdl.result_
            if res is None:
                raise RuntimeError("Fit returned no result_.")

            ic = evaluate_criteria(
                loglik=res.loglik,
                X=X,
                components=mdl.components,
                responsibilities=res.responsibilities,
                compute_icl_flag=compute_icl,
            )

            return CandidateModel(
                K=len(fnames),
                family_names=fnames,
                link_names=tuple(link_names),
                penalty_names=tuple(penalty_names),
                penalty_kwargs=tuple(pkws),
                success=True,
                result=res,
                ic=ic,
                error=None,
                model=mdl,
            )
        except Exception as e:
            return CandidateModel(
                K=len(fnames),
                family_names=fnames,
                link_names=tuple(link_names),
                penalty_names=tuple(penalty_names),
                penalty_kwargs=tuple(pkws),
                success=False,
                result=None,
                ic=None,
                error=str(e)[:200],
                model=None,
            )

    def _fit_batch(items: List[Tuple[str, ...]], *, stage_label: str) -> List[CandidateModel]:
        """
        Fit a batch of models, serial with detailed progress or parallel with stage messages.
        """
        total = len(items)
        if total == 0:
            return []

        n_jobs = parallel.effective_jobs()

        # Serial: can show fine progress
        if n_jobs == 1:
            out: List[CandidateModel] = []
            t0 = time.time()
            if show_progress:
                print(f"[beam {stage_label}] fitting {total} models (serial) ...")
            for i, fnames in enumerate(items, 1):
                cand = fit_model(fnames)
                out.append(cand)
                if show_progress and ((i % 10 == 0) or (i == total)):
                    dt = time.time() - t0
                    rate = i / max(dt, 1e-9)
                    ok = sum(1 for c in out if c.success)
                    print(f"[beam {stage_label}] {i}/{total} done | ok={ok} | {rate:.2f} fits/s")
            return out

        # Parallel: coarse progress (joblib has no cheap per-task callback without extra deps)
        if show_progress:
            print(f"[beam {stage_label}] fitting {total} models with n_jobs={n_jobs} ...")
        t0 = time.time()
        out = parallel_map(fit_model, items, cfg=parallel)
        if show_progress:
            ok = sum(1 for c in out if c.success)
            print(f"[beam {stage_label}] done in {time.time() - t0:.1f}s | ok={ok}/{total}")
        return out

    all_levels: List[List[CandidateModel]] = []

    # ----- Level K=1 -----
    logger.section("Beam search: K=1")
    items1 = [(fname,) for fname in candidate_families]
    level1_all = _fit_batch(items1, stage_label="K=1")

    # keep only successful finite-scored
    level1_ok = [c for c in level1_all if np.isfinite(c.score(criterion))]
    level1_ok.sort(key=lambda c: c.score(criterion))
    level1 = level1_ok[:beam_width]
    all_levels.append(level1)

    if verbose:
        for c in level1[: min(len(level1), 10)]:
            assert c.ic is not None
            logger.log(
                f"K=1 families={c.family_names} | loglik={c.ic.loglik:.3f} | aic={c.ic.aic:.3f} | bic={c.ic.bic:.3f}"
                + (f" | icl={c.ic.icl:.3f}" if c.ic.icl is not None else "")
            )

    # ----- Expand K=2..K_max -----
    for K in range(2, K_max + 1):
        logger.section(f"Beam search: K={K}")
        prev = all_levels[-1]
        if len(prev) == 0:
            if show_progress:
                print(f"[beam K={K}] no previous candidates survived; stopping early.")
            break

        items: List[Tuple[str, ...]] = []
        for base in prev:
            for fname in candidate_families:
                items.append(base.family_names + (fname,))

        expanded_all = _fit_batch(items, stage_label=f"K={K}")

        expanded_ok = [c for c in expanded_all if np.isfinite(c.score(criterion))]
        expanded_ok.sort(key=lambda c: c.score(criterion))
        expanded = expanded_ok[:beam_width]
        all_levels.append(expanded)

        if verbose and len(expanded) > 0:
            logger.log(f"Top {min(len(expanded), 5)} at K={K}:")
            for c in expanded[: min(len(expanded), 5)]:
                logger.log(f"  families={c.family_names} | score({criterion})={c.score(criterion):.3f}")

    # flatten all candidates across K for convenience
    flat: List[CandidateModel] = [c for level in all_levels for c in level]
    flat.sort(key=lambda c: c.score(criterion))
    return flat
