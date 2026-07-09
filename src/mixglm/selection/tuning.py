# # src/mixglm/selection/tuning.py

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
# import numpy as np

# from mixglm.model.mixture_glm import ComponentSpec, MixtureGLM

# Array = np.ndarray


# @dataclass(frozen=True)
# class TuneRow:
    # lam: float
    # criterion: str
    # score: float
    # loglik: float
    # bic: float
    # icl: Optional[float]
    # aic: float
    # pi: Array
    # nnz: List[int]   # per-component nnz excluding intercept


# def _nnz(beta: Array, thr: float = 1e-2) -> int:
    # b = np.asarray(beta, dtype=float)
    # if b.size <= 1:
        # return 0
    # return int(np.sum(np.abs(b[1:]) > thr))


# def tune_lambda_ic(
    # *,
    # y: Array,
    # X: Array,
    # components_template: Sequence[ComponentSpec],
    # make_penalty: Callable[[float], Any],  # e.g. lambda lam: LassoPenalty(lam=lam)
    # lambda_grid: Sequence[float],
    # criterion: str = "bic",                # "bic" | "icl" | "aic"
    # seed: int = 123,
    # em_kwargs: Optional[Dict[str, Any]] = None,
    # standardize: bool = True,
    # nnz_thr: float = 1e-2,
# ) -> Tuple[MixtureGLM, List[TuneRow]]:
    # """
    # Fit a grid of penalty strengths and select by IC (BIC/ICL/AIC).

    # Notes:
    # - Uses the model's own standardization (if enabled).
    # - Currently refits from scratch per lambda (robust). Warm-start can be added later.
    # """
    # em_kwargs = dict(em_kwargs or {})
    # crit = str(criterion).lower()
    # if crit not in ("bic", "icl", "aic"):
        # raise ValueError("criterion must be one of: 'bic', 'icl', 'aic'.")

    # rows: List[TuneRow] = []
    # best_model: Optional[MixtureGLM] = None
    # best_score = np.inf

    # for lam in lambda_grid:
        # # rebuild components with new penalties
        # comps: List[ComponentSpec] = []
        # for c in components_template:
            # comps.append(
                # ComponentSpec(
                    # family=c.family,
                    # link=c.link,
                    # penalty=make_penalty(float(lam)),
                # )
            # )

        # model = MixtureGLM(comps)
        # model.fit(y=y, X=X, seed=seed, standardize=standardize, **em_kwargs)
        # res = model.result_

        # score = getattr(res, crit)
        # if score is None:
            # # e.g. icl=None if compute_icl=False
            # raise RuntimeError(f"Requested criterion '{crit}' but result has {crit}=None. Set compute_icl=True.")

        # nnz = [_nnz(b, thr=nnz_thr) for b in res.betas]

        # row = TuneRow(
            # lam=float(lam),
            # criterion=crit,
            # score=float(score),
            # loglik=float(res.loglik),
            # bic=float(res.bic),
            # icl=float(res.icl) if res.icl is not None else None,
            # aic=float(res.aic),
            # pi=res.pi.copy(),
            # nnz=nnz,
        # )
        # rows.append(row)

        # if float(score) < best_score:
            # best_score = float(score)
            # best_model = model

    # if best_model is None:
        # raise RuntimeError("tune_lambda_ic failed to fit any model.")

    # return best_model, rows

# src/mixglm/selection/tuning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
import time
import numpy as np

from mixglm.model.mixture_glm import ComponentSpec, MixtureGLM
from mixglm.utils.parallel import ParallelConfig, parallel_map

Array = np.ndarray


@dataclass(frozen=True)
class TuneRow:
    lam: float
    criterion: str
    score: float
    loglik: float
    bic: float
    icl: Optional[float]
    aic: float
    pi: Array
    nnz: List[int]             # per-component nnz excluding intercept
    converged: bool = True
    error: Optional[str] = None


def _nnz(beta: Array, thr: float = 1e-2) -> int:
    b = np.asarray(beta, dtype=float)
    if b.size <= 1:
        return 0
    return int(np.sum(np.abs(b[1:]) > thr))


def tune_lambda_ic(
    *,
    y: Array,
    X: Array,
    components_template: Sequence[ComponentSpec],
    make_penalty: Callable[[float], Any],  # e.g. lambda lam: LassoPenalty(lam=lam)
    lambda_grid: Sequence[float],
    criterion: str = "bic",                # "bic" | "icl" | "aic"
    seed: int = 123,
    em_kwargs: Optional[Dict[str, Any]] = None,
    standardize: bool = True,
    nnz_thr: float = 1e-2,
    parallel: ParallelConfig | None = None,
    show_progress: bool = True,
    label: str = "tune",                   # label for progress prints
) -> Tuple[MixtureGLM, List[TuneRow]]:
    """
    Fit a grid of penalty strengths and select by IC (BIC/ICL/AIC).

    Improvements vs first version:
    - optional parallelization over lambda_grid (joblib if installed)
    - progress printing (elapsed, done/total)
    - robust to failed fits (records error, score=+inf)

    Notes:
    - Uses the model's own standardization (if enabled).
    - Refits per lambda (robust). Warm-start can be added later.
    """
    em_kwargs = dict(em_kwargs or {})
    crit = str(criterion).lower()
    if crit not in ("bic", "icl", "aic"):
        raise ValueError("criterion must be one of: 'bic', 'icl', 'aic'.")

    y = np.asarray(y)
    X = np.asarray(X)
    lambda_grid = list(lambda_grid)
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must not be empty.")

    cfg = parallel or ParallelConfig(n_jobs=1)
    n_jobs = cfg.effective_jobs()

    # Make a *copy* of template list (but reuse family/link objects, which are immutable here)
    template = list(components_template)

    t0 = time.time()
    total = len(lambda_grid)

    def _fit_one(lam: float) -> Tuple[Optional[MixtureGLM], TuneRow]:
        try:
            # rebuild components with new penalties
            comps: List[ComponentSpec] = []
            for c in template:
                comps.append(
                    ComponentSpec(
                        family=c.family,
                        link=c.link,
                        penalty=make_penalty(float(lam)),
                    )
                )

            model = MixtureGLM(comps)
            model.fit(y=y, X=X, seed=seed, standardize=standardize, **em_kwargs)
            res = model.result_
            if res is None:
                raise RuntimeError("Model fit produced result_=None.")

            score = getattr(res, crit)
            if score is None:
                raise RuntimeError(
                    f"Requested criterion '{crit}' but result has {crit}=None. "
                    f"Set compute_icl=True if you select by ICL."
                )

            nnz = [_nnz(b, thr=nnz_thr) for b in res.betas]

            row = TuneRow(
                lam=float(lam),
                criterion=crit,
                score=float(score),
                loglik=float(res.loglik),
                bic=float(res.bic),
                icl=float(res.icl) if res.icl is not None else None,
                aic=float(res.aic),
                pi=res.pi.copy(),
                nnz=nnz,
                converged=bool(res.converged),
                error=None,
            )
            return model, row

        except Exception as e:
            # failed fit: record and mark score as infinite
            row = TuneRow(
                lam=float(lam),
                criterion=crit,
                score=float("inf"),
                loglik=float("-inf"),
                bic=float("inf"),
                icl=None,
                aic=float("inf"),
                pi=np.array([], dtype=float),
                nnz=[0 for _ in template],
                converged=False,
                error=str(e)[:220],
            )
            return None, row

    # Parallel execution over lambdas
    # We also print progress as tasks finish, but with joblib we only get results at the end.
    # So: we print a single "started" line, and a single "done" line, plus a small summary.
    if show_progress:
        fams = tuple(c.family.name for c in template)
        print(f"[{label}] lambdas={total} n_jobs={n_jobs} families={fams}")

    results = parallel_map(_fit_one, lambda_grid, cfg=cfg)

    # Unpack
    models: List[Optional[MixtureGLM]] = []
    rows: List[TuneRow] = []
    for m, r in results:
        models.append(m)
        rows.append(r)

    # pick best
    best_idx = int(np.argmin([r.score for r in rows]))
    best_model = models[best_idx]
    if best_model is None:
        # if the best is None, it means all failed
        # show the first few errors for debugging
        errs = [r.error for r in rows if r.error]
        msg = "tune_lambda_ic: all lambda fits failed."
        if errs:
            msg += " Examples: " + " | ".join(errs[:3])
        raise RuntimeError(msg)

    elapsed = time.time() - t0
    if show_progress:
        ok = sum(np.isfinite(r.score) for r in rows)
        print(f"[{label}] done in {elapsed:.1f}s | ok={ok}/{total} | best lam={rows[best_idx].lam} score={rows[best_idx].score:.3f}")

    return best_model, rows
