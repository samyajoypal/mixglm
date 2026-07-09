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


def active_sets(model: MixtureGLM, *, threshold: float) -> List[List[int]]:
    assert model.result_ is not None
    out: List[List[int]] = []
    for beta in model.result_.betas:
        b = np.asarray(beta, dtype=float)
        out.append([j for j in range(1, b.size) if abs(float(b[j])) > threshold])
    return out


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
) -> Dict[str, Any]:
    t0 = time.time()
    row: Dict[str, Any] = {
        "families": "+".join(fnames),
        "K": len(fnames),
        "lambda": float(lam),
        "nonidentical": len(set(fnames)) > 1,
        "converged": False,
        "error": "",
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
            compute_icl=False,
            verbose=False,
            offset=offset_train,
        )
        res = model.result_
        if res is None:
            raise RuntimeError("fit returned no result_")
        row.update(
            {
                "converged": bool(res.converged),
                "loglik_train": float(res.loglik),
                "bic": float(res.bic),
                "aic": float(res.aic),
                "test_loglik": heldout_loglik(model, y_test, X_test, offset=offset_test),
                "pi": json.dumps([float(x) for x in res.pi]),
                "extras": json.dumps(
                    [{str(k): float(v) for k, v in ex.items()} for ex in res.extras]
                ),
            }
        )
        y_pred = model.predict_mean(X_test, offset=offset_test)
        pred_finite = bool(np.all(np.isfinite(y_pred)))
        row["test_rmse"] = float(np.sqrt(np.mean((np.asarray(y_test, dtype=float) - y_pred) ** 2)))
        row["test_mae"] = float(np.mean(np.abs(np.asarray(y_test, dtype=float) - y_pred)))
        row["prediction_finite"] = pred_finite
        row["prediction_max"] = float(np.max(y_pred)) if pred_finite else np.inf
        row["prediction_q99"] = float(np.quantile(y_pred, 0.99)) if pred_finite else np.inf
        observed_max = max(float(np.max(y_train)), float(np.max(y_test)), 1.0)
        row["prediction_stable"] = bool(pred_finite and row["prediction_max"] <= 10.0 * observed_max)
        train_mean = float(np.mean(y_train))
        train_median = float(np.median(y_train))
        baseline_mean_rmse = float(
            np.sqrt(np.mean((np.asarray(y_test, dtype=float) - train_mean) ** 2))
        )
        baseline_median_mae = float(
            np.mean(np.abs(np.asarray(y_test, dtype=float) - train_median))
        )
        row["baseline_mean_rmse"] = baseline_mean_rmse
        row["baseline_median_mae"] = baseline_median_mae
        row["rmse_skill"] = float(1.0 - row["test_rmse"] / max(baseline_mean_rmse, 1e-12))
        row["mae_skill"] = float(1.0 - row["test_mae"] / max(baseline_median_mae, 1e-12))
        active = active_sets(model, threshold=active_threshold)
        row.update(active_summary(active))
        row.update(active_feature_summary(model, active, feature_names))
        row["active_counts"] = json.dumps(row["active_counts"])
        row["n_iter"] = int(res.n_iter)
        row["min_pi"] = float(np.min(res.pi))
    except Exception as e:
        row.update(
            {
                "loglik_train": -np.inf,
                "bic": np.inf,
                "aic": np.inf,
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
        )
        row["dataset"] = spec.name
        rows.append(row)
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        if i % 10 == 0 or i == len(tasks):
            ok = sum(bool(r["converged"]) for r in rows)
            print(f"[{spec.name}] {i}/{len(tasks)} done | converged={ok}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)
    ok = df[df["converged"] & np.isfinite(df["bic"]) & np.isfinite(df["test_loglik"])].copy()
    ok.sort_values("bic").head(10).to_csv(bic_path, index=False)
    ok.sort_values("test_loglik", ascending=False).head(10).to_csv(pred_path, index=False)
    ok.sort_values("test_rmse").head(10).to_csv(rmse_path, index=False)

    if ok.empty:
        print(f"[{spec.name}] no usable fits", flush=True)
        return df

    best_bic = ok.sort_values("bic").iloc[0]
    best_pred = ok.sort_values("test_loglik", ascending=False).iloc[0]
    print(
        f"[{spec.name}] best BIC: {best_bic['families']} lam={best_bic['lambda']} "
        f"BIC={best_bic['bic']:.3f} testLL={best_bic['test_loglik']:.3f} "
        f"active={best_bic['active_counts']}",
        flush=True,
    )
    print(
        f"[{spec.name}] best pred: {best_pred['families']} lam={best_pred['lambda']} "
        f"BIC={best_pred['bic']:.3f} testLL={best_pred['test_loglik']:.3f} "
        f"active={best_pred['active_counts']}",
        flush=True,
    )
    best_rmse = ok.sort_values("test_rmse").iloc[0]
    print(
        f"[{spec.name}] best RMSE: {best_rmse['families']} lam={best_rmse['lambda']} "
        f"BIC={best_rmse['bic']:.3f} RMSE={best_rmse['test_rmse']:.3f} "
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
