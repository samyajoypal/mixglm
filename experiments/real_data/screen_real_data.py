from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from mixglm.families.registry import FAMILIES, register_defaults as register_families
from mixglm.links.registry import LINKS, register_defaults as register_links
from mixglm.model.component import ComponentSpec
from mixglm.model.mixture_glm import MixtureGLM
from mixglm.penalties.base import NoPenalty
from mixglm.penalties.lasso import LassoPenalty
from mixglm.penalties.registry import register_defaults as register_penalties


COUNT_FAMILIES = ["poisson", "nb2", "zip", "zinb"]
POSITIVE_FAMILIES = ["gaussian", "student_t", "gamma", "lognormal", "skew_normal"]
REAL_FAMILIES = ["gaussian", "student_t", "skew_normal"]
BOUNDED_FAMILIES = ["beta", "gaussian", "student_t", "skew_normal", "lognormal"]
PREPARED_COUNT_DATASETS = {
    "anes_tvnews",
    "star98_above",
    "star98_below",
    "bike_hour",
    "online_news_shares",
    "biochem_articles",
    "recreation_trips",
    "doctor_visits",
    "doctor_nondoctor",
    "doctor_hospdays",
    "doctor_hospadmi",
    "nmes_visits",
    "nmes_nvisits",
    "nmes_emergency",
    "nmes_hospital",
    "badhealth_visits",
    "mdvis_visits",
    "rwm_docvis",
    "rwm5yr_docvis",
    "rwm5yr_hospvis",
    "vietnam_pharvis",
    "insurance_car_claims",
    "insurance_singapore_claims",
    "insurance_ohlsson_claims",
    "insurance_claims_long",
    "county_murders",
    "randhealth_notmdvis",
    "randhealth_notmdvis_baseline",
    "randhealth_mentvis",
    "randhealth_totadm",
    "webworms_count",
    "bird_counts",
    "crime1_arrests",
    "crime1_felony_arrests",
    "patents_1979",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    family_space: List[str]
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    groups: np.ndarray | None = None
    offset: np.ndarray | None = None


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y)
    n = int(X.shape[0])
    if n != int(y.shape[0]):
        raise ValueError("X and y must have the same number of rows.")
    if n < 2:
        raise ValueError("Need at least two observations for a train/test split.")
    n_test = min(max(1, int(test_size)), n - 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def split_train_test_indices(
    n: int,
    *,
    test_size: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_test = min(max(1, int(test_size)), int(n) - 1)
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(int(n))
        return perm[n_test:], perm[:n_test]
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    if unique.size < 2:
        raise ValueError("Grouped splitting requires at least two groups.")
    counts = np.bincount(inverse)
    order = rng.permutation(unique.size)
    cumulative = np.cumsum(counts[order])
    candidates = np.arange(1, unique.size)
    n_groups = int(candidates[np.argmin(np.abs(cumulative[candidates - 1] - n_test))])
    test_mask = np.isin(inverse, order[:n_groups])
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def _load_feature_names(stem: str, p: int) -> List[str]:
    path = os.path.join("data", f"{stem}_features.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            names = list(json.load(f))
        if len(names) == p:
            return names
    return ["Intercept"] + [f"x{j}" for j in range(1, p)]


def _load_xy(stem: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x_path = os.path.join("data", f"{stem}_X.npy")
    y_path = os.path.join("data", f"{stem}_y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Missing prepared data/{stem}_X.npy or data/{stem}_y.npy")
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y, _load_feature_names(stem, X.shape[1])


def _load_auxiliary(stem: str, n: int) -> Tuple[np.ndarray | None, np.ndarray | None]:
    groups_path = os.path.join("data", f"{stem}_groups.npy")
    offset_path = os.path.join("data", f"{stem}_offset.npy")
    groups = np.load(groups_path, allow_pickle=True) if os.path.exists(groups_path) else None
    offset = np.load(offset_path) if os.path.exists(offset_path) else None
    for label, values in (("groups", groups), ("offset", offset)):
        if values is not None and np.asarray(values).shape != (int(n),):
            raise ValueError(f"{stem}: {label} must have shape ({int(n)},).")
    return groups, offset


def load_dataset(name: str) -> DatasetSpec:
    key = name.lower().strip()
    if key == "rand":
        X, y, names = _load_xy("rand")
        return DatasetSpec(key, "count", COUNT_FAMILIES, X, y.astype(float), names)
    if key == "blog":
        X, y, names = _load_xy("blog")
        return DatasetSpec(key, "count", COUNT_FAMILIES, X, y.astype(float), names)
    if key in PREPARED_COUNT_DATASETS:
        X, y, names = _load_xy(key)
        groups, offset = _load_auxiliary(key, X.shape[0])
        return DatasetSpec(
            key, "count", COUNT_FAMILIES, X, y.astype(float), names,
            groups=groups, offset=offset,
        )
    if key == "cali_raw":
        X, y, names = _load_xy("cali")
        return DatasetSpec(key, "positive", POSITIVE_FAMILIES, X, y.astype(float), names)
    if key == "cali_log":
        X, y, names = _load_xy("cali")
        return DatasetSpec(key, "real", REAL_FAMILIES, X, np.log(y.astype(float)), names)
    if key == "crime_beta":
        X, y, names = _load_xy("crime")
        y = np.clip(y.astype(float), 1e-5, 1.0 - 1e-5)
        return DatasetSpec(key, "bounded", BOUNDED_FAMILIES, X, y, names)
    if key == "crime_logit":
        X, y, names = _load_xy("crime")
        y = np.clip(y.astype(float), 1e-5, 1.0 - 1e-5)
        y = np.log(y / (1.0 - y))
        return DatasetSpec(key, "real", REAL_FAMILIES, X, y, names)
    if key == "ames_raw":
        X, y, names = _load_xy("ames")
        return DatasetSpec(key, "positive", POSITIVE_FAMILIES, X, y.astype(float), names)
    if key == "ames_log":
        X, y, names = _load_xy("ames")
        return DatasetSpec(key, "real", REAL_FAMILIES, X, np.log(y.astype(float)), names)
    if key == "super_raw":
        X, y, names = _load_xy("super")
        return DatasetSpec(key, "positive", POSITIVE_FAMILIES, X, y.astype(float), names)
    if key == "super_log":
        X, y, names = _load_xy("super")
        return DatasetSpec(key, "real", REAL_FAMILIES, X, np.log(y.astype(float)), names)
    if key == "parkinsons_raw":
        X, y, names = _load_xy("parkinsons")
        return DatasetSpec(key, "positive", POSITIVE_FAMILIES, X, y.astype(float), names)
    if key == "parkinsons_log":
        X, y, names = _load_xy("parkinsons")
        return DatasetSpec(key, "real", REAL_FAMILIES, X, np.log(y.astype(float)), names)
    raise ValueError(f"Unknown dataset '{name}'.")


def _response_for_screening(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "count":
        return np.log1p(y)
    return y.astype(float)


def screen_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    kind: str,
    p_screen: int,
) -> Tuple[np.ndarray, List[str], List[int]]:
    X = np.asarray(X, dtype=float)
    y_work = _response_for_screening(np.asarray(y, dtype=float), kind)
    if X.shape[1] <= p_screen + 1:
        idx = list(range(X.shape[1]))
        return X, [feature_names[i] for i in idx], idx

    scores = []
    yc = y_work - float(np.mean(y_work))
    ysd = float(np.sqrt(np.sum(yc * yc)))
    for j in range(1, X.shape[1]):
        x = X[:, j]
        xc = x - float(np.mean(x))
        xsd = float(np.sqrt(np.sum(xc * xc)))
        if xsd <= 1e-12 or ysd <= 1e-12:
            score = 0.0
        else:
            score = abs(float(np.dot(xc, yc) / (xsd * ysd)))
        scores.append((score, j))
    keep = [0] + [j for _, j in sorted(scores, reverse=True)[:p_screen]]
    return X[:, keep], [feature_names[i] for i in keep], keep


def family_tuples(families: Sequence[str], k_max: int, k_min: int = 1) -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    for k in range(int(k_min), int(k_max) + 1):
        out.extend(tuple(c) for c in itertools.combinations_with_replacement(families, k))
    return out


def make_components(fnames: Sequence[str], lam: float) -> List[ComponentSpec]:
    comps: List[ComponentSpec] = []
    for fname in fnames:
        fam = FAMILIES.create(fname)
        link = LINKS.create(fam.default_link_name)
        penalty = NoPenalty() if float(lam) <= 0.0 else LassoPenalty(lam=float(lam))
        comps.append(ComponentSpec(family=fam, link=link, penalty=penalty))
    return comps


def make_masked_components(
    fnames: Sequence[str],
    masks: Sequence[Sequence[bool]],
) -> List[ComponentSpec]:
    comps: List[ComponentSpec] = []
    for fname, mask in zip(fnames, masks):
        fam = FAMILIES.create(fname)
        link = LINKS.create(fam.default_link_name)
        comps.append(
            ComponentSpec(
                family=fam,
                link=link,
                penalty=NoPenalty(),
                coef_mask=tuple(bool(x) for x in mask),
            )
        )
    return comps


def heldout_loglik(
    model: MixtureGLM,
    y: np.ndarray,
    X: np.ndarray,
    *,
    offset: np.ndarray | None = None,
) -> float:
    X_use = model._X_fit_scale(X)
    y = np.asarray(y, dtype=float)
    offset_use = np.zeros(y.shape[0], dtype=float) if offset is None else np.asarray(offset, dtype=float)
    K = len(model.components)
    log_terms = np.empty((y.shape[0], K), dtype=float)
    assert model.result_ is not None
    for k, comp in enumerate(model.components):
        mu = comp.link.inverse(X_use @ model.result_.betas[k] + offset_use)
        ll = comp.family.loglik_component(y=y, mu=mu, extra=model.result_.extras[k])
        log_terms[:, k] = np.log(np.clip(model.result_.pi[k], 1e-300, 1.0)) + ll
    m = np.max(log_terms, axis=1, keepdims=True)
    return float(np.sum(m + np.log(np.sum(np.exp(log_terms - m), axis=1, keepdims=True))))


def prediction_summary(
    model: MixtureGLM,
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_test: np.ndarray,
    offset_test: np.ndarray | None = None,
) -> Dict[str, Any]:
    y_test = np.asarray(y_test, dtype=float)
    y_pred = model.predict_mean(X_test, offset=offset_test)
    pred_finite = bool(np.all(np.isfinite(y_pred)))
    observed_max = max(float(np.max(y_train)), float(np.max(y_test)), 1.0)
    baseline_mean_rmse = float(np.sqrt(np.mean((y_test - float(np.mean(y_train))) ** 2)))
    baseline_median_mae = float(np.mean(np.abs(y_test - float(np.median(y_train)))))
    out: Dict[str, Any] = {
        "test_loglik": heldout_loglik(model, y_test, X_test, offset=offset_test),
        "test_rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        "test_mae": float(np.mean(np.abs(y_test - y_pred))),
        "prediction_finite": pred_finite,
        "prediction_max": float(np.max(y_pred)) if pred_finite else np.inf,
        "prediction_q99": float(np.quantile(y_pred, 0.99)) if pred_finite else np.inf,
        "baseline_mean_rmse": baseline_mean_rmse,
        "baseline_median_mae": baseline_median_mae,
    }
    out["prediction_stable"] = bool(pred_finite and out["prediction_max"] <= 10.0 * observed_max)
    out["rmse_skill"] = float(1.0 - out["test_rmse"] / max(baseline_mean_rmse, 1e-12))
    out["mae_skill"] = float(1.0 - out["test_mae"] / max(baseline_median_mae, 1e-12))
    return out


def active_sets(model: MixtureGLM, *, threshold: float) -> List[List[int]]:
    assert model.result_ is not None
    out: List[List[int]] = []
    for beta in model.result_.betas:
        b = np.asarray(beta, dtype=float)
        out.append([j for j in range(1, b.size) if abs(float(b[j])) > threshold])
    return out


def active_masks(model: MixtureGLM, *, threshold: float) -> List[Tuple[bool, ...]]:
    assert model.result_ is not None
    masks: List[Tuple[bool, ...]] = []
    for beta in model.result_.betas:
        b = np.asarray(beta, dtype=float)
        keep = np.zeros(b.size, dtype=bool)
        keep[0] = True
        keep[1:] = np.abs(b[1:]) > float(threshold)
        masks.append(tuple(bool(x) for x in keep))
    return masks


def active_gate_summary(
    active: Sequence[Sequence[int]],
    *,
    min_active_per_component: int,
) -> Dict[str, Any]:
    counts = [len(a) for a in active]
    min_count = min(counts) if counts else 0
    return {
        "min_active_count": int(min_count),
        "has_intercept_only_component": bool(any(c == 0 for c in counts)),
        "passes_active_component_gate": bool(min_count >= int(min_active_per_component)),
    }


def active_feature_summary(
    model: MixtureGLM,
    active: List[List[int]],
    feature_names: Sequence[str] | None,
) -> Dict[str, str]:
    names = list(feature_names) if feature_names is not None else []
    if model.result_ is None:
        return {
            "active_sets": json.dumps(active),
            "active_features": json.dumps([]),
            "active_coefficients": json.dumps([]),
            "component_intercepts": json.dumps([]),
        }

    try:
        betas_orig = model.betas_original_scale()
    except Exception:
        betas_orig = [np.asarray(b, dtype=float) for b in model.result_.betas]

    active_features: List[List[str]] = []
    active_coefficients: List[Dict[str, float]] = []
    component_intercepts: List[float] = []
    for k, idxs in enumerate(active):
        beta = np.asarray(betas_orig[k], dtype=float)
        component_intercepts.append(float(beta[0]) if beta.size else float("nan"))
        feats: List[str] = []
        coefs: Dict[str, float] = {}
        for j in idxs:
            name = names[j] if j < len(names) else f"x{j}"
            feats.append(str(name))
            coefs[str(name)] = float(beta[j])
        active_features.append(feats)
        active_coefficients.append(coefs)

    return {
        "active_sets": json.dumps(active),
        "active_features": json.dumps(active_features),
        "active_coefficients": json.dumps(active_coefficients),
        "component_intercepts": json.dumps(component_intercepts),
    }


def active_summary(active: List[List[int]]) -> Dict[str, Any]:
    counts = [len(a) for a in active]
    if len(active) < 2:
        return {"active_counts": counts, "shared_active": None, "symmetric_diff": None}
    sets = [set(a) for a in active]
    shared = len(set.intersection(*sets)) if sets else 0
    union = len(set.union(*sets)) if sets else 0
    sym = union - shared
    return {"active_counts": counts, "shared_active": shared, "symmetric_diff": sym}


def post_lasso_refit(
    *,
    y_train: np.ndarray,
    X_train: np.ndarray,
    fnames: Sequence[str],
    masks: Sequence[Sequence[bool]],
    preferred_init: str,
    seed: int,
    max_iter: int,
    tol: float,
    n_starts: int,
    offset_train: np.ndarray | None = None,
) -> MixtureGLM:
    init_order = list(dict.fromkeys([preferred_init, "kmeans_glm", "quantile_glm", "random"]))
    best: MixtureGLM | None = None
    best_loglik = -np.inf
    for j, init_name in enumerate(init_order):
        try:
            model = MixtureGLM(make_masked_components(fnames, masks))
            model.fit(
                y_train,
                X_train,
                max_iter=max_iter,
                tol=tol,
                n_starts=n_starts,
                seed=seed + 1000 + j,
                init=init_name,
                standardize=True,
                compute_icl=True,
                verbose=False,
                offset=offset_train,
            )
            res = model.result_
            if res is not None and res.converged and np.isfinite(res.loglik) and res.loglik > best_loglik:
                best = model
                best_loglik = float(res.loglik)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("active-set refit failed for all initialization strategies")
    return best


def fit_one(
    *,
    fnames: Tuple[str, ...],
    lam: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    init: str,
    max_iter: int,
    tol: float,
    n_starts: int,
    active_threshold: float,
    feature_names: Sequence[str] | None = None,
    offset_train: np.ndarray | None = None,
    offset_test: np.ndarray | None = None,
    refit_active: bool = False,
    refit_max_iter: int | None = None,
    refit_n_starts: int | None = None,
    min_active_per_component: int = 0,
) -> Dict[str, Any]:
    t0 = time.time()
    row: Dict[str, Any] = {
        "families": "+".join(fnames),
        "K": len(fnames),
        "lambda": float(lam),
        "nonidentical": len(set(fnames)) > 1,
        "converged": False,
        "error": "",
        "refit_attempted": False,
        "refit_converged": False,
        "refit_error": "",
        "selection_source": "penalized_fit",
    }
    try:
        model = MixtureGLM(make_components(fnames, lam))
        model.fit(
            y_train,
            X_train,
            max_iter=max_iter,
            tol=tol,
            n_starts=n_starts,
            seed=seed,
            init=init,
            standardize=True,
            compute_icl=True,
            verbose=False,
            offset=offset_train,
        )
        res = model.result_
        if res is None:
            raise RuntimeError("fit returned no result_")
        final_obj = float(res.history.get("obj", [np.nan])[-1]) if res.history else np.nan
        row.update(
            {
                "converged": bool(res.converged),
                "loglik_train": float(res.loglik),
                "penalized_objective_train": final_obj,
                "penalty_value_train": float(res.loglik - final_obj) if np.isfinite(final_obj) else np.nan,
                "bic": float(res.bic),
                "aic": float(res.aic),
                "icl": float(res.icl) if res.icl is not None else np.nan,
                "pi": json.dumps([float(x) for x in res.pi]),
                "extras": json.dumps(
                    [{str(k): float(v) for k, v in ex.items()} for ex in res.extras]
                ),
            }
        )
        row.update(
            prediction_summary(
                model,
                y_train=y_train,
                y_test=y_test,
                X_test=X_test,
                offset_test=offset_test,
            )
        )
        active = active_sets(model, threshold=active_threshold)
        row.update(active_gate_summary(active, min_active_per_component=min_active_per_component))
        row.update(active_summary(active))
        row.update(active_feature_summary(model, active, feature_names))
        row["active_counts"] = json.dumps(row["active_counts"])
        row["n_iter"] = int(res.n_iter)
        row["min_pi"] = float(np.min(res.pi))
        row["penalized_active_bic"] = float(row["bic"])
        row["penalized_active_aic"] = float(row["aic"])
        row["selection_bic"] = float(row["bic"])
        row["selection_aic"] = float(row["aic"])
        row["selection_icl"] = float(row["icl"])
        row["selection_loglik_train"] = float(row["loglik_train"])
        row["selection_test_loglik"] = float(row["test_loglik"])
        row["selection_test_rmse"] = float(row["test_rmse"])
        row["selection_test_mae"] = float(row["test_mae"])
        row["selection_prediction_finite"] = bool(row["prediction_finite"])
        row["selection_prediction_stable"] = bool(row["prediction_stable"])

        if bool(refit_active) and float(lam) > 0.0 and bool(res.converged):
            row["refit_attempted"] = True
            masks = active_masks(model, threshold=active_threshold)
            try:
                refit = post_lasso_refit(
                    y_train=y_train,
                    X_train=X_train,
                    fnames=fnames,
                    masks=masks,
                    preferred_init=init,
                    seed=seed,
                    max_iter=int(refit_max_iter or max_iter),
                    tol=tol,
                    n_starts=int(refit_n_starts or n_starts),
                    offset_train=offset_train,
                )
                refit_res = refit.result_
                if refit_res is None:
                    raise RuntimeError("refit returned no result_")
                refit_pred = prediction_summary(
                    refit,
                    y_train=y_train,
                    y_test=y_test,
                    X_test=X_test,
                    offset_test=offset_test,
                )
                row.update(
                    {
                        "refit_converged": bool(refit_res.converged),
                        "refit_loglik_train": float(refit_res.loglik),
                        "refit_bic": float(refit_res.bic),
                        "refit_aic": float(refit_res.aic),
                        "refit_icl": float(refit_res.icl) if refit_res.icl is not None else np.nan,
                        "refit_test_loglik": float(refit_pred["test_loglik"]),
                        "refit_test_rmse": float(refit_pred["test_rmse"]),
                        "refit_test_mae": float(refit_pred["test_mae"]),
                        "refit_prediction_finite": bool(refit_pred["prediction_finite"]),
                        "refit_prediction_stable": bool(refit_pred["prediction_stable"]),
                        "refit_pi": json.dumps([float(x) for x in refit_res.pi]),
                        "refit_extras": json.dumps(
                            [{str(k): float(v) for k, v in ex.items()} for ex in refit_res.extras]
                        ),
                        "selection_source": "active_set_refit",
                        "selection_bic": float(refit_res.bic),
                        "selection_aic": float(refit_res.aic),
                        "selection_icl": float(refit_res.icl) if refit_res.icl is not None else np.nan,
                        "selection_loglik_train": float(refit_res.loglik),
                        "selection_test_loglik": float(refit_pred["test_loglik"]),
                        "selection_test_rmse": float(refit_pred["test_rmse"]),
                        "selection_test_mae": float(refit_pred["test_mae"]),
                        "selection_prediction_finite": bool(refit_pred["prediction_finite"]),
                        "selection_prediction_stable": bool(refit_pred["prediction_stable"]),
                    }
                )
            except Exception as e:
                row["refit_error"] = str(e)[:260]
                row["selection_source"] = "refit_failed"
                row["selection_bic"] = np.inf
                row["selection_aic"] = np.inf
                row["selection_icl"] = np.inf
                row["selection_loglik_train"] = -np.inf
                row["selection_test_loglik"] = -np.inf
                row["selection_test_rmse"] = np.inf
                row["selection_test_mae"] = np.inf
                row["selection_prediction_finite"] = False
                row["selection_prediction_stable"] = False

        row["publication_candidate"] = bool(
            row["nonidentical"]
            and int(row["K"]) >= 2
            and bool(row["passes_active_component_gate"])
        )
    except Exception as e:
        row.update(
            {
                "loglik_train": -np.inf,
                "penalized_objective_train": -np.inf,
                "penalty_value_train": np.nan,
                "bic": np.inf,
                "aic": np.inf,
                "icl": np.inf,
                "test_loglik": -np.inf,
                "test_rmse": np.inf,
                "test_mae": np.inf,
                "pi": "[]",
                "extras": "[]",
                "prediction_finite": False,
                "prediction_stable": False,
                "prediction_max": np.inf,
                "prediction_q99": np.inf,
                "baseline_mean_rmse": np.nan,
                "baseline_median_mae": np.nan,
                "rmse_skill": -np.inf,
                "mae_skill": -np.inf,
                "active_counts": "[]",
                "active_sets": "[]",
                "active_features": "[]",
                "active_coefficients": "[]",
                "component_intercepts": "[]",
                "min_active_count": 0,
                "has_intercept_only_component": True,
                "passes_active_component_gate": False,
                "publication_candidate": False,
                "penalized_active_bic": np.inf,
                "penalized_active_aic": np.inf,
                "selection_bic": np.inf,
                "selection_aic": np.inf,
                "selection_icl": np.inf,
                "selection_loglik_train": -np.inf,
                "selection_test_loglik": -np.inf,
                "selection_test_rmse": np.inf,
                "selection_test_mae": np.inf,
                "selection_prediction_finite": False,
                "selection_prediction_stable": False,
                "refit_loglik_train": -np.inf,
                "refit_bic": np.inf,
                "refit_aic": np.inf,
                "refit_icl": np.inf,
                "refit_test_loglik": -np.inf,
                "refit_test_rmse": np.inf,
                "refit_test_mae": np.inf,
                "refit_prediction_finite": False,
                "refit_prediction_stable": False,
                "refit_pi": "[]",
                "refit_extras": "[]",
                "shared_active": np.nan,
                "symmetric_diff": np.nan,
                "n_iter": 0,
                "min_pi": np.nan,
                "error": str(e)[:260],
            }
        )
    row["seconds"] = float(time.time() - t0)
    return row


def run_dataset(
    spec: DatasetSpec,
    *,
    n_train: int,
    n_test: int,
    p_screen: int,
    k_min: int,
    k_max: int,
    lambdas: Sequence[float],
    max_iter: int,
    tol: float,
    n_starts: int,
    seed: int,
    active_threshold: float,
    init_strategy: str,
    refit_active: bool,
    refit_max_iter: int | None,
    refit_n_starts: int | None,
    min_active_per_component: int,
    out_dir: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = min(spec.X.shape[0], int(n_train) + int(n_test))
    idx = rng.choice(spec.X.shape[0], size=n_total, replace=False)
    X_all = np.asarray(spec.X)[idx]
    y_all = np.asarray(spec.y)[idx]
    groups_all = None if spec.groups is None else np.asarray(spec.groups)[idx]
    train_idx, test_idx = split_train_test_indices(
        n_total,
        test_size=int(n_test),
        seed=seed,
        groups=groups_all,
    )
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    X_train, names, keep_idx = screen_features(
        X_all[train_idx],
        y_train,
        spec.feature_names,
        kind=spec.kind,
        p_screen=p_screen,
    )
    X_test = X_all[test_idx][:, keep_idx]
    offset_all = None if spec.offset is None else np.asarray(spec.offset, dtype=float)[idx]
    offset_train = None if offset_all is None else offset_all[train_idx]
    offset_test = None if offset_all is None else offset_all[test_idx]

    init = "random" if spec.kind == "count" else "quantile"
    if init_strategy != "auto":
        init = init_strategy
    combos = family_tuples(spec.family_space, k_max, k_min=k_min)
    tasks = [(fnames, lam) for fnames in combos for lam in lambdas]

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{spec.name}_screen_raw.csv")
    bic_path = os.path.join(out_dir, f"{spec.name}_top10_bic.csv")
    pred_path = os.path.join(out_dir, f"{spec.name}_top10_pred.csv")
    rmse_path = os.path.join(out_dir, f"{spec.name}_top10_rmse.csv")

    print(
        f"\n[{spec.name}] kind={spec.kind} n_train={X_train.shape[0]} "
        f"n_test={X_test.shape[0]} p={X_train.shape[1]} "
        f"families={spec.family_space} tasks={len(tasks)}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    for i, (fnames, lam) in enumerate(tasks, 1):
        row = fit_one(
            fnames=fnames,
            lam=float(lam),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=seed + i,
            init=init,
            max_iter=max_iter,
            tol=tol,
            n_starts=n_starts,
            active_threshold=active_threshold,
            feature_names=names,
            offset_train=offset_train,
            offset_test=offset_test,
            refit_active=refit_active,
            refit_max_iter=refit_max_iter,
            refit_n_starts=refit_n_starts,
            min_active_per_component=min_active_per_component,
        )
        row["dataset"] = spec.name
        rows.append(row)
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        if i % 10 == 0 or i == len(tasks):
            ok = sum(bool(r["converged"]) for r in rows)
            print(f"[{spec.name}] {i}/{len(tasks)} done | converged={ok}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)
    bic_col = "selection_bic" if "selection_bic" in df.columns else "bic"
    loglik_col = "selection_test_loglik" if "selection_test_loglik" in df.columns else "test_loglik"
    rmse_col = "selection_test_rmse" if "selection_test_rmse" in df.columns else "test_rmse"
    ok = df[df["converged"] & np.isfinite(df[bic_col]) & np.isfinite(df[loglik_col])].copy()
    ok.sort_values(bic_col).head(10).to_csv(bic_path, index=False)
    ok.sort_values(loglik_col, ascending=False).head(10).to_csv(pred_path, index=False)
    ok.sort_values(rmse_col).head(10).to_csv(rmse_path, index=False)
    if "publication_candidate" in ok.columns:
        ok[ok["publication_candidate"]].sort_values(bic_col).head(20).to_csv(
            os.path.join(out_dir, f"{spec.name}_top20_publication_bic.csv"),
            index=False,
        )

    if ok.empty:
        print(f"[{spec.name}] no usable fits", flush=True)
        return df

    best_bic = ok.sort_values(bic_col).iloc[0]
    best_pred = ok.sort_values(loglik_col, ascending=False).iloc[0]
    print(
        f"[{spec.name}] best BIC: {best_bic['families']} lam={best_bic['lambda']} "
        f"BIC={best_bic[bic_col]:.3f} testLL={best_bic[loglik_col]:.3f} "
        f"active={best_bic['active_counts']} source={best_bic.get('selection_source', 'penalized_fit')}",
        flush=True,
    )
    print(
        f"[{spec.name}] best pred: {best_pred['families']} lam={best_pred['lambda']} "
        f"BIC={best_pred[bic_col]:.3f} testLL={best_pred[loglik_col]:.3f} "
        f"active={best_pred['active_counts']}",
        flush=True,
    )
    best_rmse = ok.sort_values(rmse_col).iloc[0]
    print(
        f"[{spec.name}] best RMSE: {best_rmse['families']} lam={best_rmse['lambda']} "
        f"BIC={best_rmse[bic_col]:.3f} RMSE={best_rmse[rmse_col]:.3f} "
        f"active={best_rmse['active_counts']}",
        flush=True,
    )
    if (
        best_bic["families"] == best_pred["families"]
        and bool(best_bic["nonidentical"])
        and float(best_bic["lambda"]) == float(best_pred["lambda"])
    ):
        print(f"[{spec.name}] PROMISING: non-identical model wins both BIC and prediction.", flush=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cheap local screen for real-data MixGLM examples.")
    parser.add_argument(
        "--datasets",
        default="rand,blog,cali_log,cali_raw,crime_beta,crime_logit",
        help="Comma-separated dataset keys.",
    )
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--p-screen", type=int, default=20)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=2)
    parser.add_argument("--lambdas", default="0,20,50,100")
    parser.add_argument("--exclude-families", default="", help="Comma-separated families to drop from every dataset.")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--n-starts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--active-threshold", type=float, default=1e-6)
    parser.add_argument("--refit-active", action="store_true")
    parser.add_argument("--refit-max-iter", type=int, default=None)
    parser.add_argument("--refit-n-starts", type=int, default=None)
    parser.add_argument("--min-active-per-component", type=int, default=0)
    parser.add_argument(
        "--init",
        default="auto",
        choices=["auto", "random", "quantile", "kmeans_y", "kmeans_glm", "quantile_glm"],
    )
    parser.add_argument("--out-dir", default="experiments/real_data/screen_outputs")
    return parser.parse_args()


def main() -> None:
    register_families()
    register_links()
    register_penalties()
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    exclude = {x.strip().lower() for x in args.exclude_families.split(",") if x.strip()}

    all_rows: List[pd.DataFrame] = []
    for ds in datasets:
        spec = load_dataset(ds)
        if exclude:
            kept = [f for f in spec.family_space if f.lower() not in exclude]
            if not kept:
                raise ValueError(f"All families were excluded for dataset {ds}.")
            spec = DatasetSpec(
                name=spec.name,
                kind=spec.kind,
                family_space=kept,
                X=spec.X,
                y=spec.y,
                feature_names=spec.feature_names,
                groups=spec.groups,
                offset=spec.offset,
            )
        df = run_dataset(
            spec,
            n_train=args.n_train,
            n_test=args.n_test,
            p_screen=args.p_screen,
            k_min=args.k_min,
            k_max=args.k_max,
            lambdas=lambdas,
            max_iter=args.max_iter,
            tol=args.tol,
            n_starts=args.n_starts,
            seed=args.seed,
            active_threshold=args.active_threshold,
            init_strategy=args.init,
            refit_active=args.refit_active,
            refit_max_iter=args.refit_max_iter,
            refit_n_starts=args.refit_n_starts,
            min_active_per_component=args.min_active_per_component,
            out_dir=args.out_dir,
        )
        all_rows.append(df)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_path = os.path.join(args.out_dir, "all_screen_raw.csv")
        all_df.to_csv(all_path, index=False)
        print(f"\nSaved combined screen to {all_path}", flush=True)


if __name__ == "__main__":
    main()
