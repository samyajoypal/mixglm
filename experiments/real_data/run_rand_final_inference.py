from __future__ import annotations

import itertools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from mixglm.families.registry import FAMILIES, register_defaults as register_families
from mixglm.inference.louis import louis_observed_information
from mixglm.links.registry import LINKS, register_defaults as register_links
from mixglm.model.component import ComponentSpec
from mixglm.model.mixture_glm import MixtureGLM
from mixglm.penalties.base import NoPenalty
from mixglm.penalties.registry import register_defaults as register_penalties

from experiments.real_data.screen_real_data import (
    active_feature_summary,
    active_sets,
    active_summary,
    family_tuples,
    fit_one,
    load_dataset,
    make_components,
    screen_features,
)


def _parse_csv(value: str, cast=str) -> List[Any]:
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def _safe_id(parts: Sequence[Any]) -> str:
    return "_".join(str(x).replace("+", "-").replace("/", "-").replace(" ", "_") for x in parts)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_checkpoint(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(_jsonable(payload)))
    os.replace(temp_path, path)


def _register() -> None:
    register_families()
    register_links()
    register_penalties()


def _full_task(task: Dict[str, Any], cache_dir: Path) -> Dict[str, Any]:
    _register()
    path = cache_dir / f"{task['task_id']}.json"
    cached = _load_checkpoint(path)
    if cached is not None:
        return cached
    row = fit_one(
        fnames=tuple(task["families"]),
        lam=float(task["lambda"]),
        X_train=task["X"],
        y_train=task["y"],
        X_test=task["X"],
        y_test=task["y"],
        seed=int(task["seed"]),
        init=str(task["init"]),
        max_iter=int(task["max_iter"]),
        tol=float(task["tol"]),
        n_starts=int(task["n_starts"]),
        active_threshold=float(task["active_threshold"]),
        feature_names=task["feature_names"],
        refit_active=True,
        refit_max_iter=int(task["refit_max_iter"]),
        refit_n_starts=int(task["refit_n_starts"]),
        min_active_per_component=int(task["min_active_per_component"]),
    )
    row.update(
        {
            "dataset": task["dataset"],
            "init": task["init"],
            "seed": int(task["seed"]),
            "task_id": task["task_id"],
            "n": int(task["y"].shape[0]),
            "p": int(task["X"].shape[1]),
        }
    )
    _write_checkpoint(path, row)
    return row


def _fit_selection_path(
    *,
    y: np.ndarray,
    X: np.ndarray,
    families: Tuple[str, ...],
    lambdas: Sequence[float],
    inits: Sequence[str],
    max_iter: int,
    tol: float,
    n_starts: int,
    refit_n_starts: int,
    active_threshold: float,
    min_active_per_component: int,
    seed: int,
) -> Tuple[MixtureGLM, MixtureGLM, List[Tuple[bool, ...]], float, str]:
    best_penalized: MixtureGLM | None = None
    best_refit: MixtureGLM | None = None
    best_masks: List[Tuple[bool, ...]] | None = None
    best_bic = np.inf
    best_lambda = np.nan
    best_init = ""
    task_no = 0
    for lam in lambdas:
        for init in inits:
            task_no += 1
            try:
                model = MixtureGLM(make_components(families, float(lam)))
                model.fit(
                    y,
                    X,
                    max_iter=max_iter,
                    tol=tol,
                    n_starts=n_starts,
                    seed=seed + task_no,
                    init=init,
                    standardize=True,
                    compute_icl=False,
                    verbose=False,
                )
                result = model.result_
                if result is None or not result.converged or not np.isfinite(result.bic):
                    continue
                masks = _support_masks(model, active_threshold)
                active_counts = [int(sum(mask) - 1) for mask in masks]
                if min(active_counts) < int(min_active_per_component):
                    continue
                refit = _post_lasso_refit(
                    y=y,
                    X=X,
                    families=families,
                    masks=masks,
                    preferred_init=init,
                    max_iter=max_iter,
                    tol=tol,
                    n_starts=refit_n_starts,
                    seed=seed + 10000 + task_no,
                )
                refit_result = refit.result_
                if (
                    refit_result is None
                    or not refit_result.converged
                    or not np.isfinite(refit_result.bic)
                ):
                    continue
                if refit_result.bic < best_bic:
                    best_penalized = model
                    best_refit = refit
                    best_masks = masks
                    best_bic = float(refit_result.bic)
                    best_lambda = float(lam)
                    best_init = str(init)
            except Exception:
                continue
    if best_penalized is None or best_refit is None or best_masks is None:
        raise RuntimeError("No active-set refit passed the component gate along the selection path.")
    return best_penalized, best_refit, best_masks, best_lambda, best_init


def _support_masks(model: MixtureGLM, threshold: float) -> List[Tuple[bool, ...]]:
    if model.result_ is None:
        raise ValueError("Model must be fitted.")
    masks: List[Tuple[bool, ...]] = []
    for beta in model.result_.betas:
        keep = np.abs(np.asarray(beta, dtype=float)) > float(threshold)
        keep[0] = True
        masks.append(tuple(bool(x) for x in keep))
    return masks


def _masked_components(families: Sequence[str], masks: Sequence[Tuple[bool, ...]]) -> List[ComponentSpec]:
    components: List[ComponentSpec] = []
    for family_name, mask in zip(families, masks):
        family = FAMILIES.create(family_name)
        components.append(
            ComponentSpec(
                family=family,
                link=LINKS.create(family.default_link_name),
                penalty=NoPenalty(),
                coef_mask=tuple(mask),
            )
        )
    return components


def _post_lasso_refit(
    *,
    y: np.ndarray,
    X: np.ndarray,
    families: Tuple[str, ...],
    masks: Sequence[Tuple[bool, ...]],
    preferred_init: str,
    max_iter: int,
    tol: float,
    n_starts: int,
    seed: int,
) -> MixtureGLM:
    init_order = list(
        dict.fromkeys([preferred_init, "kmeans_glm", "quantile_glm", "random"])
    )
    best: MixtureGLM | None = None
    best_loglik = -np.inf
    for j, init in enumerate(init_order):
        try:
            model = MixtureGLM(_masked_components(families, masks))
            model.fit(
                y,
                X,
                max_iter=max_iter,
                tol=tol,
                n_starts=n_starts,
                seed=seed + 1000 + j,
                init=init,
                standardize=True,
                compute_icl=False,
                verbose=False,
            )
            result = model.result_
            if result is not None and result.converged and result.loglik > best_loglik:
                best = model
                best_loglik = float(result.loglik)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Component-specific post-lasso refit failed.")
    return best


def _masks_from_leaderboard_row(
    row: pd.Series,
    *,
    p: int,
    n_components: int,
) -> List[Tuple[bool, ...]]:
    active_sets_value = row["active_sets"]
    active_sets_parsed = (
        json.loads(active_sets_value)
        if isinstance(active_sets_value, str)
        else active_sets_value
    )
    if len(active_sets_parsed) != int(n_components):
        raise ValueError("Leaderboard active sets do not match the selected component count.")
    masks: List[Tuple[bool, ...]] = []
    for active in active_sets_parsed:
        mask = np.zeros(int(p), dtype=bool)
        mask[0] = True
        indices = np.asarray(active, dtype=int)
        if indices.size and (indices.min() < 1 or indices.max() >= int(p)):
            raise ValueError("Leaderboard active-set index is outside the fitted design.")
        mask[indices] = True
        masks.append(tuple(bool(value) for value in mask))
    return masks


def _fit_leaderboard_winner(
    *,
    row: pd.Series,
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int,
    tol: float,
    leaderboard_starts: int,
    refit_n_starts: int,
    refit_batches: int,
    active_threshold: float,
) -> Tuple[MixtureGLM, MixtureGLM, List[Tuple[bool, ...]], Tuple[str, ...], float, str]:
    families = tuple(str(row["families"]).split("+"))
    selected_lambda = float(row["lambda"])
    selected_init = str(row["init"])
    selected_seed = int(row["seed"])
    masks = _masks_from_leaderboard_row(
        row, p=X.shape[1], n_components=len(families)
    )

    penalized = MixtureGLM(make_components(families, selected_lambda))
    penalized.fit(
        y,
        X,
        max_iter=max_iter,
        tol=tol,
        n_starts=leaderboard_starts,
        seed=selected_seed,
        init=selected_init,
        standardize=True,
        compute_icl=False,
        verbose=False,
    )
    if penalized.result_ is None or not penalized.result_.converged:
        raise RuntimeError("The selected leaderboard penalized fit could not be reproduced.")
    reproduced_masks = _support_masks(penalized, active_threshold)
    if reproduced_masks != masks:
        raise RuntimeError("The selected leaderboard support could not be reproduced exactly.")

    best_refit: MixtureGLM | None = None
    best_loglik = -np.inf
    target_loglik = float(row["selection_loglik_train"])
    tolerance = max(1e-4, 1e-8 * abs(target_loglik))
    for batch in range(max(1, int(refit_batches))):
        try:
            refit = _post_lasso_refit(
                y=y,
                X=X,
                families=families,
                masks=masks,
                preferred_init=selected_init,
                max_iter=max_iter,
                tol=tol,
                n_starts=refit_n_starts,
                seed=selected_seed + 100000 * batch,
            )
            if refit.result_ is not None and refit.result_.loglik > best_loglik:
                best_refit = refit
                best_loglik = float(refit.result_.loglik)
            if np.isfinite(target_loglik) and best_loglik >= target_loglik - tolerance:
                break
        except Exception:
            continue
    if best_refit is None or best_refit.result_ is None:
        raise RuntimeError("The selected leaderboard active-set refit could not be reproduced.")

    if np.isfinite(target_loglik) and best_loglik < target_loglik - tolerance:
        raise RuntimeError(
            "Final active-set refit is inferior to the leaderboard winner: "
            f"{best_loglik:.6f} < {target_loglik:.6f}."
        )
    return (
        penalized,
        best_refit,
        masks,
        families,
        selected_lambda,
        selected_init,
    )


def _model_payload(
    model: MixtureGLM,
    *,
    families: Sequence[str],
    feature_names: Sequence[str],
    masks: Sequence[Tuple[bool, ...]] | None = None,
) -> Dict[str, Any]:
    if model.result_ is None:
        raise ValueError("Model must be fitted.")
    result = model.result_
    active = active_sets(model, threshold=1e-5)
    payload: Dict[str, Any] = {
        "families": list(families),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "loglik": float(result.loglik),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "pi": result.pi,
        "betas_fit_scale": result.betas,
        "betas_input_scale": model.betas_original_scale(),
        "extras": result.extras,
        "active_sets": active,
        "active_features": [
            [str(feature_names[j]) for j in idx] for idx in active
        ],
    }
    if masks is not None:
        payload["coef_masks"] = masks
    return _jsonable(payload)


def _conditional_louis(
    model: MixtureGLM,
    *,
    y: np.ndarray,
    X: np.ndarray,
    masks: Sequence[Tuple[bool, ...]],
    families: Sequence[str],
    feature_names: Sequence[str],
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if model.result_ is None:
        raise ValueError("Model must be fitted.")
    X_fit = model._X_fit_scale(X)
    result = model.result_
    louis = louis_observed_information(
        y=y,
        X=X_fit,
        components=model.components,
        pi=result.pi,
        betas=result.betas,
        extras=result.extras,
        tau=result.responsibilities,
        derivative_method="analytic",
        ridge=1e-8,
    )
    beta_pattern = re.compile(r"^beta\[(\d+)\]\[(\d+)\]$")
    free: List[int] = []
    for idx, name in enumerate(louis.param_names):
        match = beta_pattern.match(name)
        if match is None:
            free.append(idx)
            continue
        component = int(match.group(1))
        coefficient = int(match.group(2))
        if masks[component][coefficient]:
            free.append(idx)

    info = louis.info[np.ix_(free, free)]
    info_symmetric = 0.5 * (info + info.T)
    eigenvalues = np.linalg.eigvalsh(info_symmetric)
    rank = int(np.linalg.matrix_rank(info_symmetric))
    condition_number = float(np.linalg.cond(info_symmetric))
    information_valid = bool(
        np.all(np.isfinite(eigenvalues))
        and rank == int(info.shape[0])
        and float(np.min(eigenvalues)) > 0.0
    )
    estimates = louis.theta_hat[free]
    if information_valid:
        cov = np.linalg.pinv(info_symmetric, rcond=1e-10)
        se = np.sqrt(np.diag(cov))
    else:
        se = np.full(estimates.shape, np.nan, dtype=float)
    zcrit = float(norm.ppf(1.0 - alpha / 2.0))
    parameter_names = [louis.param_names[j] for j in free]

    def display_name(parameter: str) -> str:
        match = beta_pattern.match(parameter)
        if match is not None:
            component = int(match.group(1))
            coefficient = int(match.group(2))
            return f"{families[component]}: {feature_names[coefficient]}"
        if parameter.startswith("eta_pi"):
            return "mixing log-odds"
        return parameter

    table = pd.DataFrame(
        {
            "parameter": parameter_names,
            "label": [display_name(name) for name in parameter_names],
            "estimate": estimates,
            "se": se,
            "ci_lower": estimates - zcrit * se,
            "ci_upper": estimates + zcrit * se,
        }
    )
    diagnostics = {
        "dimension": int(info.shape[0]),
        "rank": rank,
        "condition_number": condition_number,
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "information_valid_for_wald": information_valid,
        "warning": (
            ""
            if information_valid
            else "Conditional Wald intervals withheld because observed information is not positive definite."
        ),
        "derivative_sources": louis.derivative_sources,
    }
    return table, _jsonable(diagnostics)


def _bootstrap_indices(
    rng: np.random.Generator,
    n: int,
    groups: np.ndarray | None,
) -> Tuple[np.ndarray, str, int]:
    if groups is None:
        return rng.integers(0, int(n), size=int(n)), "observation", int(n)
    groups = np.asarray(groups)
    if groups.shape != (int(n),):
        raise ValueError(f"groups must have shape ({int(n)},); got {groups.shape}.")
    unique_groups = np.unique(groups)
    sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
    blocks = [np.flatnonzero(groups == group) for group in sampled_groups]
    return np.concatenate(blocks), "participant_group", int(unique_groups.size)


def _aligned_component_order(
    *,
    fitted_eta: np.ndarray,
    fitted_pi: np.ndarray,
    families: Sequence[str],
    reference_eta: np.ndarray,
    reference_pi: np.ndarray,
) -> List[int]:
    fitted_eta = np.asarray(fitted_eta, dtype=float)
    fitted_pi = np.asarray(fitted_pi, dtype=float)
    reference_eta = np.asarray(reference_eta, dtype=float)
    reference_pi = np.asarray(reference_pi, dtype=float)
    if fitted_eta.shape != reference_eta.shape:
        raise ValueError("Reference and bootstrap component predictors have different shapes.")

    order = list(range(len(families)))
    eta_scale = max(float(np.std(reference_eta)), 1e-8)
    for family in dict.fromkeys(families):
        indices = [j for j, name in enumerate(families) if name == family]
        if len(indices) < 2:
            continue
        best_permutation = tuple(indices)
        best_cost = np.inf
        for permutation in itertools.permutations(indices):
            cost = 0.0
            for reference_index, fitted_index in zip(indices, permutation):
                eta_difference = fitted_eta[fitted_index] - reference_eta[reference_index]
                cost += float(np.mean(eta_difference ** 2)) / (eta_scale ** 2)
                cost += 0.05 * float(
                    (fitted_pi[fitted_index] - reference_pi[reference_index]) ** 2
                )
            if cost < best_cost:
                best_cost = cost
                best_permutation = permutation
        for reference_index, fitted_index in zip(indices, best_permutation):
            order[reference_index] = fitted_index
    return order


def _bootstrap_one(
    *,
    replicate: int,
    replicate_seed: int,
    X_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray | None,
    all_feature_names: Sequence[str],
    data_kind: str,
    reference_eta: np.ndarray,
    reference_pi: np.ndarray,
    families: Tuple[str, ...],
    lambdas: Sequence[float],
    inits: Sequence[str],
    p_screen: int,
    max_iter: int,
    tol: float,
    n_starts: int,
    refit_n_starts: int,
    active_threshold: float,
    min_active_per_component: int,
    cache_dir: Path,
) -> Dict[str, Any]:
    _register()
    path = cache_dir / f"bootstrap_{replicate:04d}.json"
    cached = _load_checkpoint(path)
    if cached is not None:
        return cached
    started = time.time()
    try:
        rng = np.random.default_rng(replicate_seed)
        indices, resampling_unit, n_sampled_units = _bootstrap_indices(
            rng, y_all.shape[0], groups_all
        )
        X_boot_raw = X_all[indices]
        y_boot = y_all[indices]
        X_boot, names_boot, keep_idx = screen_features(
            X_boot_raw,
            y_boot,
            all_feature_names,
            kind=data_kind,
            p_screen=p_screen,
        )
        penalized, refit, masks, selected_lambda, selected_init = _fit_selection_path(
            y=y_boot,
            X=X_boot,
            families=families,
            lambdas=lambdas,
            inits=inits,
            max_iter=max_iter,
            tol=tol,
            n_starts=n_starts,
            refit_n_starts=refit_n_starts,
            active_threshold=active_threshold,
            min_active_per_component=min_active_per_component,
            seed=replicate_seed + 100,
        )
        if penalized.result_ is None or refit.result_ is None:
            raise RuntimeError("Bootstrap fit returned no result.")

        name_to_original = {str(name): j for j, name in enumerate(all_feature_names)}
        beta_input_raw = refit.betas_original_scale()
        beta_full = []
        for beta in beta_input_raw:
            expanded = np.zeros(len(all_feature_names), dtype=float)
            for local_j, feature_name in enumerate(names_boot):
                expanded[name_to_original[str(feature_name)]] = float(beta[local_j])
            beta_full.append(expanded)
        fitted_eta = np.vstack([X_all @ beta for beta in beta_full])
        component_order = _aligned_component_order(
            fitted_eta=fitted_eta,
            fitted_pi=refit.result_.pi,
            families=families,
            reference_eta=reference_eta,
            reference_pi=reference_pi,
        )
        beta_input = [beta_input_raw[j] for j in component_order]
        ordered_masks = [masks[j] for j in component_order]
        ordered_pi = [refit.result_.pi[j] for j in component_order]
        ordered_extras = [refit.result_.extras[j] for j in component_order]
        row: Dict[str, Any] = {
            "replicate": int(replicate),
            "seed": int(replicate_seed),
            "success": True,
            "error": "",
            "resampling_unit": resampling_unit,
            "n_sampled_units": n_sampled_units,
            "n_bootstrap_observations": int(indices.size),
            "component_permutation": json.dumps(component_order),
            "selected_lambda": float(selected_lambda),
            "selected_init": selected_init,
            "penalized_bic": float(penalized.result_.bic),
            "refit_loglik": float(refit.result_.loglik),
            "refit_bic": float(refit.result_.bic),
            "screened_original_indices": json.dumps([int(x) for x in keep_idx]),
            "screened_features": json.dumps([str(x) for x in names_boot]),
            "active_original_indices": json.dumps(
                [
                    [name_to_original[str(names_boot[j])] for j, flag in enumerate(mask) if flag and j > 0]
                    for mask in ordered_masks
                ]
            ),
            "active_counts": json.dumps(
                [int(sum(mask) - 1) for mask in ordered_masks]
            ),
            "seconds": float(time.time() - started),
        }
        for k, weight in enumerate(ordered_pi):
            row[f"pi_{k}"] = float(weight)
        for k, beta in enumerate(beta_input):
            row[f"intercept_{k}"] = float(beta[0])
            for original_j in range(1, len(all_feature_names)):
                row[f"beta_{k}_{original_j}"] = 0.0
            for local_j in range(1, len(names_boot)):
                original_j = name_to_original[str(names_boot[local_j])]
                if ordered_masks[k][local_j]:
                    row[f"beta_{k}_{original_j}"] = float(beta[local_j])
        for k, extra in enumerate(ordered_extras):
            for name, value in extra.items():
                row[f"extra_{k}_{name}"] = float(value)
                if name == "log_alpha":
                    row[f"alpha_{k}"] = float(np.exp(float(value)))
        _write_checkpoint(path, row)
        return row
    except Exception as exc:
        row = {
            "replicate": int(replicate),
            "seed": int(replicate_seed),
            "success": False,
            "error": str(exc)[:500],
            "seconds": float(time.time() - started),
        }
        _write_checkpoint(path, row)
        return row


def _bootstrap_summaries(
    *,
    draws: pd.DataFrame,
    final_model: MixtureGLM,
    final_masks: Sequence[Tuple[bool, ...]],
    all_feature_names: Sequence[str],
    final_keep_idx: Sequence[int],
    out_dir: Path,
) -> None:
    draws.to_csv(out_dir / "bootstrap_draws.csv", index=False)
    successful = draws[draws["success"]].copy()
    selection_rows: List[Dict[str, Any]] = []
    active_lists = [json.loads(x) for x in successful["active_original_indices"]]
    screened_lists = [set(json.loads(x)) for x in successful["screened_original_indices"]]
    full_selected = [
        {
            int(final_keep_idx[local_j])
            for local_j, flag in enumerate(mask)
            if flag and local_j > 0
        }
        for mask in final_masks
    ]
    for k in range(len(final_masks)):
        for j in range(1, len(all_feature_names)):
            selected = [j in set(active[k]) for active in active_lists]
            screened = [j in values for values in screened_lists]
            selection_rows.append(
                {
                    "component": k,
                    "feature_index": j,
                    "feature": all_feature_names[j],
                    "screening_frequency": float(np.mean(screened)) if screened else np.nan,
                    "selection_frequency": float(np.mean(selected)) if selected else np.nan,
                    "selected_in_full_fit": j in full_selected[k],
                }
            )
    pd.DataFrame(selection_rows).to_csv(out_dir / "selection_frequencies.csv", index=False)

    if final_model.result_ is None:
        raise ValueError("Final model must be fitted.")
    final_beta = final_model.betas_original_scale()
    estimates: Dict[str, float] = {}
    for k, value in enumerate(final_model.result_.pi):
        estimates[f"pi_{k}"] = float(value)
    for k, beta in enumerate(final_beta):
        estimates[f"intercept_{k}"] = float(beta[0])
        for j in range(1, len(all_feature_names)):
            estimates[f"beta_{k}_{j}"] = 0.0
        for local_j in range(1, len(final_keep_idx)):
            original_j = int(final_keep_idx[local_j])
            if final_masks[k][local_j]:
                estimates[f"beta_{k}_{original_j}"] = float(beta[local_j])
    for k, extra in enumerate(final_model.result_.extras):
        for name, value in extra.items():
            estimates[f"extra_{k}_{name}"] = float(value)
            if name == "log_alpha":
                estimates[f"alpha_{k}"] = float(np.exp(float(value)))

    summary_rows: List[Dict[str, Any]] = []
    for parameter, estimate in estimates.items():
        if parameter not in successful:
            continue
        values = pd.to_numeric(successful[parameter], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        summary_rows.append(
            {
                "parameter": parameter,
                "full_data_estimate": estimate,
                "bootstrap_mean": float(np.mean(values)) if values.size else np.nan,
                "bootstrap_se": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                "ci_lower": float(np.quantile(values, 0.025)) if values.size else np.nan,
                "ci_upper": float(np.quantile(values, 0.975)) if values.size else np.nan,
                "n_valid": int(values.size),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "bootstrap_parameter_summary.csv", index=False)

    lambda_summary = (
        successful.groupby("selected_lambda", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    lambda_summary["frequency"] = lambda_summary["count"] / max(len(successful), 1)
    lambda_summary.to_csv(out_dir / "bootstrap_lambda_frequencies.csv", index=False)


def main() -> None:
    _register()
    out_dir = Path(
        os.environ.get(
            "MIXGLM_INFERENCE_OUTPUT_ROOT",
            "experiments/real_data/targeted_outputs/rand_final_inference_v1",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    full_cache = out_dir / "full_checkpoints"
    bootstrap_cache = out_dir / "bootstrap_checkpoints"
    full_cache.mkdir(exist_ok=True)
    bootstrap_cache.mkdir(exist_ok=True)

    dataset = os.environ.get("MIXGLM_INFERENCE_DATASET", "randhealth_notmdvis_baseline")
    p_screen = int(os.environ.get("MIXGLM_INFERENCE_P_SCREEN", "50"))
    lambdas = _parse_csv(
        os.environ.get("MIXGLM_INFERENCE_LAMBDAS", "0,0.1,0.25,0.5,1,2,5,10,20"),
        float,
    )
    inits = _parse_csv(os.environ.get("MIXGLM_INFERENCE_INITS", "kmeans_glm,quantile_glm"))
    max_iter = int(os.environ.get("MIXGLM_INFERENCE_MAX_ITER", "200"))
    tol = float(os.environ.get("MIXGLM_INFERENCE_TOL", "1e-3"))
    leaderboard_starts = int(os.environ.get("MIXGLM_INFERENCE_LEADERBOARD_STARTS", "2"))
    bootstrap_starts = int(os.environ.get("MIXGLM_INFERENCE_BOOTSTRAP_STARTS", "1"))
    refit_starts = int(os.environ.get("MIXGLM_INFERENCE_REFIT_STARTS", "2"))
    final_refit_starts = int(
        os.environ.get("MIXGLM_INFERENCE_FINAL_REFIT_STARTS", "5")
    )
    final_refit_batches = int(
        os.environ.get("MIXGLM_INFERENCE_FINAL_REFIT_BATCHES", "3")
    )
    active_threshold = float(os.environ.get("MIXGLM_INFERENCE_ACTIVE_THRESHOLD", "1e-5"))
    bootstrap_reps = int(os.environ.get("MIXGLM_INFERENCE_BOOTSTRAP_REPS", "500"))
    min_active_per_component = int(
        os.environ.get("MIXGLM_INFERENCE_MIN_ACTIVE_PER_COMPONENT", "1")
    )
    bootstrap_family_request = os.environ.get(
        "MIXGLM_INFERENCE_BOOTSTRAP_FAMILIES", "poisson,nb2"
    ).strip()
    bootstrap_families = (
        None
        if bootstrap_family_request.lower() == "auto"
        else tuple(_parse_csv(bootstrap_family_request))
    )
    seed = int(os.environ.get("MIXGLM_INFERENCE_SEED", "20260703"))
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("MIXGLM_INFERENCE_N_JOBS", "1")))

    spec = load_dataset(dataset)
    leaderboard_families = _parse_csv(
        os.environ.get("MIXGLM_INFERENCE_LEADERBOARD_FAMILIES", ",".join(spec.family_space))
    )
    leaderboard_k_max = int(os.environ.get("MIXGLM_INFERENCE_LEADERBOARD_K_MAX", "3"))
    X_full, feature_names, keep_idx = screen_features(
        spec.X,
        spec.y,
        spec.feature_names,
        kind=spec.kind,
        p_screen=p_screen,
    )
    y_full = np.asarray(spec.y, dtype=float)
    design_payload = {
        "X": X_full,
        "y": y_full,
        "feature_names": np.asarray(feature_names, dtype=object),
        "keep_idx": np.asarray(keep_idx, dtype=int),
    }
    if spec.groups is not None:
        design_payload["groups"] = np.asarray(spec.groups)
    np.savez_compressed(out_dir / "full_data_design.npz", **design_payload)

    metadata = {
        "dataset": dataset,
        "n": int(y_full.size),
        "p": int(X_full.shape[1]),
        "p_screen": p_screen,
        "lambdas": lambdas,
        "penalty_type": "lasso",
        "penalty_mixing_alpha": 1.0,
        "lambda_shared_across_components": True,
        "lambda_zero_is_unpenalized": True,
        "inits": inits,
        "max_iter": max_iter,
        "tol": tol,
        "leaderboard_starts": leaderboard_starts,
        "bootstrap_starts": bootstrap_starts,
        "refit_starts": refit_starts,
        "final_refit_starts": final_refit_starts,
        "final_refit_batches": final_refit_batches,
        "active_threshold": active_threshold,
        "bootstrap_reps": bootstrap_reps,
        "min_active_per_component": min_active_per_component,
        "selection_rule": "minimum active-set refit BIC subject to the active-component gate",
        "bootstrap_families_requested": bootstrap_family_request,
        "leaderboard_families": leaderboard_families,
        "leaderboard_k_max": leaderboard_k_max,
        "bootstrap_scope": "screening, lambda, and support reselected within every replicate",
        "bootstrap_resampling_unit": (
            "participant_group" if spec.groups is not None else "observation"
        ),
        "n_groups": int(np.unique(spec.groups).size) if spec.groups is not None else None,
        "seed": seed,
        "n_jobs": n_jobs,
    }
    combos = family_tuples(leaderboard_families, k_max=leaderboard_k_max, k_min=1)
    tasks: List[Dict[str, Any]] = []
    for families in combos:
        for init in inits:
            for lam in lambdas:
                task_id = _safe_id([dataset, "K" + str(len(families)), "+".join(families), "L" + str(lam), init])
                tasks.append(
                    {
                        "task_id": task_id,
                        "dataset": dataset,
                        "families": families,
                        "lambda": lam,
                        "init": init,
                        "seed": seed + len(tasks) + 1,
                        "max_iter": max_iter,
                        "tol": tol,
                        "n_starts": leaderboard_starts,
                        "active_threshold": active_threshold,
                        "refit_max_iter": max_iter,
                        "refit_n_starts": refit_starts,
                        "min_active_per_component": min_active_per_component,
                        "X": X_full,
                        "y": y_full,
                        "feature_names": feature_names,
                    }
                )
    print(f"Full-data leaderboard: {len(tasks)} fits", flush=True)
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_full_task)(task, full_cache) for task in tasks
    )
    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(out_dir / "full_data_leaderboard_raw.csv", index=False)
    usable = leaderboard[
        leaderboard["converged"]
        & np.isfinite(leaderboard["selection_bic"])
        & leaderboard["selection_prediction_stable"]
    ].sort_values("selection_bic")
    usable.head(50).to_csv(out_dir / "full_data_top50_bic.csv", index=False)
    if usable.empty:
        raise RuntimeError("No full-data leaderboard model converged.")
    overall_winner = usable.iloc[0]
    publication_usable = usable[
        usable["nonidentical"]
        & ~usable["has_intercept_only_component"]
        & usable["passes_active_component_gate"]
    ]
    publication_usable.head(50).to_csv(
        out_dir / "full_data_top50_publication_bic.csv", index=False
    )
    print(
        f"Overall full-data winner: {overall_winner['families']} lambda={overall_winner['lambda']} "
        f"refit BIC={overall_winner['selection_bic']:.3f}",
        flush=True,
    )
    publication_winner = None
    if not publication_usable.empty:
        publication_winner = publication_usable.iloc[0]
        print(
            f"Publication-eligible winner: {publication_winner['families']} "
            f"lambda={publication_winner['lambda']} "
            f"refit BIC={publication_winner['selection_bic']:.3f}",
            flush=True,
        )

    if bootstrap_families is None:
        overall_is_eligible = bool(
            overall_winner["nonidentical"]
            and not overall_winner["has_intercept_only_component"]
            and overall_winner["passes_active_component_gate"]
        )
        if not overall_is_eligible:
            raise RuntimeError(
                "The overall BIC winner is not a nonidentical model with active slopes "
                "in every component; automatic inference will not substitute a runner-up."
            )
        final_winner_row = overall_winner
        bootstrap_families = tuple(str(final_winner_row["families"]).split("+"))
    else:
        requested_family_label = "+".join(bootstrap_families)
        matching = usable[
            (usable["families"] == requested_family_label)
            & ~usable["has_intercept_only_component"]
            & usable["passes_active_component_gate"]
        ]
        if matching.empty:
            raise RuntimeError(
                f"No eligible leaderboard model matches {requested_family_label}."
            )
        final_winner_row = matching.iloc[0]
    metadata["bootstrap_families_selected"] = bootstrap_families
    metadata["selected_leaderboard_task"] = str(final_winner_row["task_id"])
    metadata["selected_leaderboard_bic"] = float(final_winner_row["selection_bic"])
    metadata["selected_leaderboard_loglik"] = float(
        final_winner_row["selection_loglik_train"]
    )
    (out_dir / "run_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2))
    (out_dir / "selected_leaderboard_row.json").write_text(
        json.dumps(_jsonable(final_winner_row.to_dict()), indent=2)
    )

    (
        final_penalized,
        final_refit,
        masks,
        bootstrap_families,
        selected_lambda,
        selected_init,
    ) = _fit_leaderboard_winner(
        row=final_winner_row,
        y=y_full,
        X=X_full,
        max_iter=max_iter,
        tol=tol,
        leaderboard_starts=leaderboard_starts,
        refit_n_starts=final_refit_starts,
        refit_batches=final_refit_batches,
        active_threshold=active_threshold,
    )
    (out_dir / "selected_penalized_model.json").write_text(
        json.dumps(
            _model_payload(
                final_penalized,
                families=bootstrap_families,
                feature_names=feature_names,
            ),
            indent=2,
        )
    )
    (out_dir / "post_lasso_model.json").write_text(
        json.dumps(
            _model_payload(
                final_refit,
                families=bootstrap_families,
                feature_names=feature_names,
                masks=masks,
            ),
            indent=2,
        )
    )
    louis_table, louis_diagnostics = _conditional_louis(
        final_refit,
        y=y_full,
        X=X_full,
        masks=masks,
        families=bootstrap_families,
        feature_names=feature_names,
    )
    louis_table.to_csv(out_dir / "post_lasso_conditional_louis.csv", index=False)
    (out_dir / "post_lasso_louis_diagnostics.json").write_text(
        json.dumps(louis_diagnostics, indent=2)
    )
    print(
        f"Selected {bootstrap_families} lambda={selected_lambda} init={selected_init}; "
        f"active={[int(sum(m) - 1) for m in masks]}; "
        f"refit BIC={final_refit.result_.bic:.3f}",
        flush=True,
    )

    reference_beta = final_refit.betas_original_scale()
    reference_beta_full = []
    for beta in reference_beta:
        expanded = np.zeros(len(spec.feature_names), dtype=float)
        for local_j, original_j in enumerate(keep_idx):
            expanded[int(original_j)] = float(beta[local_j])
        reference_beta_full.append(expanded)
    reference_eta = np.vstack([
        np.asarray(spec.X, dtype=float) @ beta for beta in reference_beta_full
    ])
    reference_pi = np.asarray(final_refit.result_.pi, dtype=float)
    seed_rng = np.random.default_rng(seed + 300000)
    replicate_seeds = seed_rng.integers(1, 2**31 - 1, size=bootstrap_reps)
    print(f"Selection-aware bootstrap: B={bootstrap_reps}", flush=True)
    bootstrap_rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_bootstrap_one)(
            replicate=b,
            replicate_seed=int(replicate_seeds[b]),
            X_all=np.asarray(spec.X, dtype=float),
            y_all=y_full,
            groups_all=None if spec.groups is None else np.asarray(spec.groups),
            all_feature_names=spec.feature_names,
            data_kind=spec.kind,
            reference_eta=reference_eta,
            reference_pi=reference_pi,
            families=bootstrap_families,
            lambdas=lambdas,
            inits=inits,
            p_screen=p_screen,
            max_iter=max_iter,
            tol=tol,
            n_starts=bootstrap_starts,
            refit_n_starts=refit_starts,
            active_threshold=active_threshold,
            min_active_per_component=min_active_per_component,
            cache_dir=bootstrap_cache,
        )
        for b in range(bootstrap_reps)
    )
    draws = pd.DataFrame(bootstrap_rows).sort_values("replicate")
    _bootstrap_summaries(
        draws=draws,
        final_model=final_refit,
        final_masks=masks,
        all_feature_names=spec.feature_names,
        final_keep_idx=keep_idx,
        out_dir=out_dir,
    )
    successes = int(draws["success"].sum())
    print(f"Bootstrap complete: {successes}/{bootstrap_reps} successful", flush=True)


if __name__ == "__main__":
    main()
