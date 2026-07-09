# src/mixglm/inference/bootstrap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np

from mixglm.model.mixture_glm import MixtureGLM, MixtureGLMResult, ComponentSpec
from mixglm.utils.parallel import ParallelConfig, parallel_map
from mixglm.utils.logging import Logger

Array = np.ndarray


@dataclass
class BootstrapResult:
    """
    Bootstrap output for mixture GLM parameters.

    Notes
    - We store draws for pi, betas, and (optionally) extras.
    - Label switching can occur. For now we return raw draws.
      Later we can add component alignment (e.g., by sorting by mean mu at x_ref).
    """
    pi_draws: Array                  # (B, K)
    beta_draws: Array                # (B, K, p)
    extra_draws: Optional[List[List[Dict[str, Any]]]]  # length B, list of K dicts per draw
    failed: int
    seed: Optional[int]


def bootstrap_se(
    *,
    model: MixtureGLM,
    y: Array,
    X: Array,
    B: int = 200,
    seed: Optional[int] = None,
    n_jobs: int = 1,
    max_iter: Optional[int] = None,
    tol: Optional[float] = None,
    n_starts: Optional[int] = None,
    init: Optional[str] = None,
    verbose: bool = False,
) -> BootstrapResult:
    """
    Nonparametric bootstrap for mixture GLM parameters.

    Parameters
    ----------
    model : MixtureGLM
        A configured model instance (components set).
        This function will refit the same model structure on bootstrap samples.
    y, X : data
    B : number of bootstrap replications
    max_iter, tol, n_starts, init : optional overrides for fitting options
    n_jobs : parallel jobs (uses joblib if available)
    """
    y = np.asarray(y)
    X = np.asarray(X)
    n = y.shape[0]
    K = len(model.components)
    p = X.shape[1]

    rng = np.random.default_rng(seed)
    logger = Logger(verbose=verbose)

    # capture fit args from the original model if available
    # fall back to MixtureGLM.fit defaults if not
    fit_kwargs: Dict[str, Any] = {}
    if max_iter is not None:
        fit_kwargs["max_iter"] = int(max_iter)
    if tol is not None:
        fit_kwargs["tol"] = float(tol)
    if n_starts is not None:
        fit_kwargs["n_starts"] = int(n_starts)
    if init is not None:
        fit_kwargs["init"] = str(init)

    # Use component specs from provided model (do not mutate original)
    comps: List[ComponentSpec] = list(model.components)

    def one_boot(b: int) -> Optional[Tuple[Array, Array, List[Dict[str, Any]]]]:
        # independent seed per replicate for reproducibility under parallelism
        rrng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        idx = rrng.integers(0, n, size=n)
        yb = y[idx]
        Xb = X[idx, :]

        try:
            mdl_b = MixtureGLM(comps).fit(yb, Xb, **fit_kwargs)
            res = mdl_b.result_
            assert res is not None
            pi_b = res.pi.copy()
            beta_b = np.stack([bk.copy() for bk in res.betas], axis=0)  # (K,p)
            extra_b = [dict(ex) for ex in res.extras]
            return pi_b, beta_b, extra_b
        except Exception:
            return None

    cfg = ParallelConfig(n_jobs=n_jobs)

    logger.section("Bootstrap")
    outs = parallel_map(one_boot, range(B), cfg=cfg)

    pi_draws = np.full((B, K), np.nan, dtype=float)
    beta_draws = np.full((B, K, p), np.nan, dtype=float)
    extra_draws: List[List[Dict[str, Any]]] = []

    failed = 0
    for b, out in enumerate(outs):
        if out is None:
            failed += 1
            extra_draws.append([{} for _ in range(K)])
            continue
        pi_b, beta_b, extra_b = out
        pi_draws[b, :] = pi_b
        beta_draws[b, :, :] = beta_b
        extra_draws.append(extra_b)

    logger.log(f"Bootstrap finished: B={B}, failed={failed}")

    return BootstrapResult(
        pi_draws=pi_draws,
        beta_draws=beta_draws,
        extra_draws=extra_draws,
        failed=failed,
        seed=seed,
    )
