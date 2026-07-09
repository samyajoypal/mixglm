from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from mixglm.families.registry import register_defaults as register_families
from mixglm.links.registry import register_defaults as register_links
from mixglm.penalties.registry import register_defaults as register_penalties

from experiments.real_data.screen_real_data import (
    DatasetSpec,
    family_tuples,
    fit_one,
    load_dataset,
    screen_features,
)


def _parse_csv(value: str, cast=str) -> List[Any]:
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_id(parts: List[Any]) -> str:
    return "_".join(str(p).replace("/", "-").replace("+", "-").replace(" ", "_") for p in parts)


def _split_indices(
    n: int,
    *,
    test_size: int,
    seed: int,
    groups: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = min(max(1, int(test_size)), int(n) - 1)
    if groups is None:
        perm = rng.permutation(int(n))
        return perm[target:], perm[:target]

    groups = np.asarray(groups)
    if groups.shape != (int(n),):
        raise ValueError(f"groups must have shape ({int(n)},); got {groups.shape}.")
    unique, inverse = np.unique(groups, return_inverse=True)
    if unique.size < 2:
        raise ValueError("Grouped splitting requires at least two distinct groups.")
    counts = np.bincount(inverse)
    order = rng.permutation(unique.size)
    cumulative = np.cumsum(counts[order])
    candidates = np.arange(1, unique.size)
    n_test_groups = int(candidates[np.argmin(np.abs(cumulative[candidates - 1] - target))])
    test_group_codes = order[:n_test_groups]
    test_mask = np.isin(inverse, test_group_codes)
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError("Grouped split produced an empty partition.")
    return train_idx, test_idx


def _prepare_dataset(
    spec: DatasetSpec,
    *,
    n_train: int,
    n_test: int,
    p_screen: int,
    seed: int,
    out_dir: str,
    split_id: int,
    use_full_sample: bool,
    test_fraction: float,
) -> str:
    rng = np.random.default_rng(seed)
    n_available = int(spec.X.shape[0])
    if use_full_sample:
        idx = np.arange(n_available)
    else:
        n_total = min(n_available, int(n_train) + int(n_test))
        idx = rng.choice(n_available, size=n_total, replace=False)

    X_all = np.asarray(spec.X)[idx]
    y_all = np.asarray(spec.y)[idx]
    groups_all = None if spec.groups is None else np.asarray(spec.groups)[idx]
    offset_all = None if spec.offset is None else np.asarray(spec.offset, dtype=float)[idx]
    target_test = (
        int(round(float(test_fraction) * idx.size))
        if use_full_sample
        else int(n_test)
    )
    train_idx, test_idx = _split_indices(
        idx.size,
        test_size=target_test,
        seed=seed,
        groups=groups_all,
    )

    X_train_all = X_all[train_idx]
    X_test_all = X_all[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    X_train, names, keep_idx = screen_features(
        X_train_all,
        y_train,
        spec.feature_names,
        kind=spec.kind,
        p_screen=p_screen,
    )
    X_test = X_test_all[:, keep_idx]
    offset_train = None if offset_all is None else offset_all[train_idx]
    offset_test = None if offset_all is None else offset_all[test_idx]
    groups_train = None if groups_all is None else groups_all[train_idx]
    groups_test = None if groups_all is None else groups_all[test_idx]

    prep_dir = os.path.join(out_dir, "prepared")
    os.makedirs(prep_dir, exist_ok=True)
    path = os.path.join(prep_dir, f"{spec.name}_split{split_id}.npz")
    payload: Dict[str, Any] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "keep_idx": np.asarray(keep_idx, dtype=int),
        "feature_names": np.asarray(names, dtype=object),
    }
    if offset_train is not None:
        payload["offset_train"] = offset_train
        payload["offset_test"] = offset_test
    if groups_train is not None:
        payload["groups_train"] = groups_train
        payload["groups_test"] = groups_test
    np.savez_compressed(path, **payload)

    group_overlap = 0
    if groups_train is not None and groups_test is not None:
        group_overlap = len(set(np.asarray(groups_train).tolist()) & set(np.asarray(groups_test).tolist()))
    meta = {
        "dataset": spec.name,
        "split_id": int(split_id),
        "split_seed": int(seed),
        "kind": spec.kind,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "p": int(X_train.shape[1]),
        "family_space": list(spec.family_space),
        "full_sample": bool(use_full_sample),
        "grouped_split": groups_all is not None,
        "n_train_groups": int(np.unique(groups_train).size) if groups_train is not None else None,
        "n_test_groups": int(np.unique(groups_test).size) if groups_test is not None else None,
        "group_overlap": int(group_overlap),
        "offset_used": offset_all is not None,
        "train_zero_fraction": float(np.mean(y_train == 0)),
        "test_zero_fraction": float(np.mean(y_test == 0)),
        "train_response_mean": float(np.mean(y_train)),
        "test_response_mean": float(np.mean(y_test)),
    }
    with open(os.path.join(prep_dir, f"{spec.name}_split{split_id}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def _process_task(task: Dict[str, Any], cache_dir: str) -> Dict[str, Any]:
    register_families()
    register_links()
    register_penalties()

    cache_path = os.path.join(cache_dir, task["task_id"] + ".json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    data = np.load(task["prepared_path"], allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    offset_train = data["offset_train"] if "offset_train" in data.files else None
    offset_test = data["offset_test"] if "offset_test" in data.files else None
    feature_names = [str(x) for x in data["feature_names"].tolist()] if "feature_names" in data.files else None

    row = fit_one(
        fnames=tuple(task["families"]),
        lam=float(task["lambda"]),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=int(task["seed"]),
        init=str(task["init"]),
        max_iter=int(task["max_iter"]),
        tol=float(task["tol"]),
        n_starts=int(task["n_starts"]),
        active_threshold=float(task["active_threshold"]),
        feature_names=feature_names,
        offset_train=offset_train,
        offset_test=offset_test,
    )
    row.update(
        {
            "dataset": task["dataset"],
            "split_id": int(task["split_id"]),
            "split_seed": int(task["split_seed"]),
            "n_train": int(y_train.shape[0]),
            "n_test": int(y_test.shape[0]),
            "offset_used": offset_train is not None,
            "init": task["init"],
            "task_id": task["task_id"],
        }
    )
    with open(cache_path, "w") as f:
        json.dump(row, f)
    return row


def _write_leaderboards(df: pd.DataFrame, out_dir: str) -> None:
    df.to_csv(os.path.join(out_dir, "targeted_screen_raw.csv"), index=False)
    summary_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    usable = df[
        df["converged"]
        & np.isfinite(df["bic"])
        & np.isfinite(df["test_loglik"])
    ].copy()
    usable["test_loglik_per_obs"] = usable["test_loglik"] / usable["n_test"]

    for (dataset, split_id), g in usable.groupby(["dataset", "split_id"], sort=True):
        ok = g[g["converged"] & np.isfinite(g["bic"]) & np.isfinite(g["test_loglik"])].copy()
        if ok.empty:
            continue
        stable = ok[
            ok["prediction_stable"]
            & np.isfinite(ok["test_rmse"])
            & np.isfinite(ok["test_mae"])
        ].copy()
        prefix = os.path.join(out_dir, f"{dataset}_split{int(split_id)}")
        ok.sort_values("bic").head(20).to_csv(prefix + "_top20_bic.csv", index=False)
        ok.sort_values("test_loglik", ascending=False).head(20).to_csv(prefix + "_top20_loglik.csv", index=False)
        stable.sort_values("test_rmse").head(20).to_csv(prefix + "_top20_rmse.csv", index=False)
        stable.sort_values("test_mae").head(20).to_csv(prefix + "_top20_mae.csv", index=False)
        nonid = ok[ok["nonidentical"]].copy()
        if not nonid.empty:
            nonid.sort_values("bic").head(20).to_csv(prefix + "_top20_nonident_bic.csv", index=False)
            nonid.sort_values("test_loglik", ascending=False).head(20).to_csv(
                prefix + "_top20_nonident_loglik.csv", index=False
            )
            nonid[nonid["prediction_stable"]].sort_values("test_rmse").head(20).to_csv(
                prefix + "_top20_nonident_rmse.csv", index=False
            )

        best_bic = ok.sort_values("bic").iloc[0]
        best_ll = ok.sort_values("test_loglik", ascending=False).iloc[0]
        best_rmse = stable.sort_values("test_rmse").iloc[0] if not stable.empty else best_bic
        best_k1 = ok[ok["K"] == 1].sort_values("bic").iloc[0]
        identical = ok[(ok["K"] >= 2) & (~ok["nonidentical"])]
        nonidentical = ok[ok["nonidentical"]]
        best_identical = identical.sort_values("bic").iloc[0] if not identical.empty else None
        best_nonidentical = nonidentical.sort_values("bic").iloc[0] if not nonidentical.empty else None
        summary_rows.append(
            {
                "dataset": dataset,
                "split_id": int(split_id),
                "n_train": int(best_bic["n_train"]),
                "n_test": int(best_bic["n_test"]),
                "best_bic_families": best_bic["families"],
                "best_bic_init": best_bic["init"],
                "best_bic_lambda": best_bic["lambda"],
                "best_bic": best_bic["bic"],
                "best_bic_nonidentical": bool(best_bic["nonidentical"]),
                "best_bic_prediction_finite": bool(best_bic["prediction_finite"]),
                "best_bic_prediction_stable": bool(best_bic["prediction_stable"]),
                "delta_bic_vs_k1": float(best_bic["bic"] - best_k1["bic"]),
                "delta_bic_vs_identical": (
                    float(best_bic["bic"] - best_identical["bic"])
                    if best_identical is not None else np.nan
                ),
                "best_loglik_families": best_ll["families"],
                "best_loglik_init": best_ll["init"],
                "best_loglik_lambda": best_ll["lambda"],
                "best_test_loglik": best_ll["test_loglik"],
                "best_loglik_nonidentical": bool(best_ll["nonidentical"]),
                "best_rmse_families": best_rmse["families"],
                "best_rmse_init": best_rmse["init"],
                "best_rmse_lambda": best_rmse["lambda"],
                "best_test_rmse": best_rmse["test_rmse"],
                "best_rmse_nonidentical": bool(best_rmse["nonidentical"]),
            }
        )
        comparison_models = [("overall", best_bic), ("k1", best_k1)]
        if best_identical is not None:
            comparison_models.append(("identical", best_identical))
        if best_nonidentical is not None:
            comparison_models.append(("nonidentical", best_nonidentical))
        for label, row in comparison_models:
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "split_id": int(split_id),
                    "class": label,
                    "families": row["families"],
                    "K": int(row["K"]),
                    "lambda": float(row["lambda"]),
                    "init": row["init"],
                    "bic": float(row["bic"]),
                    "delta_bic_from_split_winner": float(row["bic"] - best_bic["bic"]),
                    "test_loglik": float(row["test_loglik"]),
                    "test_loglik_per_obs": float(row["test_loglik_per_obs"]),
                    "test_rmse": float(row["test_rmse"]),
                    "test_mae": float(row["test_mae"]),
                    "rmse_skill": float(row["rmse_skill"]),
                    "mae_skill": float(row["mae_skill"]),
                    "prediction_finite": bool(row["prediction_finite"]),
                    "prediction_stable": bool(row["prediction_stable"]),
                    "pi": row["pi"],
                    "active_counts": row["active_counts"],
                }
            )
        print(
            f"[{dataset} split {int(split_id)}] best BIC={best_bic['families']} init={best_bic['init']} "
            f"lam={best_bic['lambda']} bic={best_bic['bic']:.3f} ll={best_bic['test_loglik']:.3f} "
            f"rmse={best_bic['test_rmse']:.3f} active={best_bic['active_counts']}",
            flush=True,
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "targeted_screen_summary.csv"), index=False)
        pd.DataFrame(comparison_rows).to_csv(os.path.join(out_dir, "split_comparisons.csv"), index=False)

    config_cols = ["dataset", "split_id", "families", "K", "lambda", "nonidentical"]
    best_init = usable.sort_values("bic").drop_duplicates(config_cols, keep="first").copy()
    best_init["delta_bic"] = best_init["bic"] - best_init.groupby(["dataset", "split_id"])["bic"].transform("min")
    stability = (
        best_init.groupby(["dataset", "families", "K", "lambda", "nonidentical"], as_index=False)
        .agg(
            splits_observed=("split_id", "nunique"),
            bic_wins=("delta_bic", lambda x: int(np.sum(np.isclose(x, 0.0)))),
            mean_delta_bic=("delta_bic", "mean"),
            median_delta_bic=("delta_bic", "median"),
            max_delta_bic=("delta_bic", "max"),
            mean_test_loglik_per_obs=("test_loglik_per_obs", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_test_mae=("test_mae", "mean"),
            mean_rmse_skill=("rmse_skill", "mean"),
            mean_mae_skill=("mae_skill", "mean"),
            finite_prediction_rate=("prediction_finite", "mean"),
            stable_prediction_rate=("prediction_stable", "mean"),
            median_min_pi=("min_pi", "median"),
        )
        .sort_values(["dataset", "bic_wins", "mean_delta_bic"], ascending=[True, False, True])
    )
    stability.to_csv(os.path.join(out_dir, "model_stability.csv"), index=False)

    family_best = best_init.sort_values("bic").drop_duplicates(
        ["dataset", "split_id", "families", "K"], keep="first"
    )
    family_best["delta_bic"] = family_best["bic"] - family_best.groupby(
        ["dataset", "split_id"]
    )["bic"].transform("min")
    family_stability = (
        family_best.groupby(["dataset", "families", "K", "nonidentical"], as_index=False)
        .agg(
            splits_observed=("split_id", "nunique"),
            bic_wins=("delta_bic", lambda x: int(np.sum(np.isclose(x, 0.0)))),
            mean_delta_bic=("delta_bic", "mean"),
            median_delta_bic=("delta_bic", "median"),
            mean_test_loglik_per_obs=("test_loglik_per_obs", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_test_mae=("test_mae", "mean"),
            finite_prediction_rate=("prediction_finite", "mean"),
            stable_prediction_rate=("prediction_stable", "mean"),
        )
        .sort_values(["dataset", "bic_wins", "mean_delta_bic"], ascending=[True, False, True])
    )
    family_stability.to_csv(os.path.join(out_dir, "family_stability.csv"), index=False)


def main() -> None:
    register_families()
    register_links()
    register_penalties()

    out_dir = os.environ.get("MIXGLM_REAL_OUTPUT_ROOT", "experiments/real_data/targeted_outputs/v1_kmeans_glm")
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)

    datasets = _parse_csv(os.environ.get("MIXGLM_REAL_DATASETS", "rand,blog,super_raw,parkinsons_log,crime_beta"))
    inits = _parse_csv(os.environ.get("MIXGLM_REAL_INITS", "kmeans_glm,quantile_glm"))
    lambdas = _parse_csv(os.environ.get("MIXGLM_REAL_LAMBDAS", "5,10,20,30,50,100"), float)
    k_min = int(os.environ.get("MIXGLM_REAL_K_MIN", "2"))
    k_max = int(os.environ.get("MIXGLM_REAL_K_MAX", "3"))
    n_train = int(os.environ.get("MIXGLM_REAL_N_TRAIN", "2000"))
    n_test = int(os.environ.get("MIXGLM_REAL_N_TEST", "1000"))
    p_screen = int(os.environ.get("MIXGLM_REAL_P_SCREEN", "40"))
    max_iter = int(os.environ.get("MIXGLM_REAL_MAX_ITER", "120"))
    tol = float(os.environ.get("MIXGLM_REAL_TOL", "1e-3"))
    n_starts = int(os.environ.get("MIXGLM_REAL_N_STARTS", "2"))
    active_threshold = float(os.environ.get("MIXGLM_REAL_ACTIVE_THRESHOLD", "1e-6"))
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("MIXGLM_REAL_N_JOBS", "1")))
    seed = int(os.environ.get("MIXGLM_REAL_SEED", "20260624"))
    n_splits = int(os.environ.get("MIXGLM_REAL_N_SPLITS", "1"))
    test_fraction = float(os.environ.get("MIXGLM_REAL_TEST_FRACTION", "0.2"))
    use_full_sample = _env_bool("MIXGLM_REAL_USE_FULL_SAMPLE", False)
    if n_splits < 1:
        raise ValueError("MIXGLM_REAL_N_SPLITS must be at least 1.")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("MIXGLM_REAL_TEST_FRACTION must lie strictly between 0 and 1.")

    metadata = {
        "datasets": datasets,
        "inits": inits,
        "lambdas": lambdas,
        "k_min": k_min,
        "k_max": k_max,
        "n_train": n_train,
        "n_test": n_test,
        "p_screen": p_screen,
        "max_iter": max_iter,
        "tol": tol,
        "n_starts": n_starts,
        "active_threshold": active_threshold,
        "n_jobs": n_jobs,
        "seed": seed,
        "n_splits": n_splits,
        "test_fraction": test_fraction,
        "use_full_sample": use_full_sample,
        "screening_scope": "training_only",
        "grouped_when_available": True,
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    tasks: List[Dict[str, Any]] = []
    for d_i, ds in enumerate(datasets):
        spec = load_dataset(ds)
        for split_id in range(n_splits):
            split_seed = seed + 10000 * d_i + 1000 * split_id
            prep_path = _prepare_dataset(
                spec,
                n_train=n_train,
                n_test=n_test,
                p_screen=p_screen,
                seed=split_seed,
                out_dir=out_dir,
                split_id=split_id,
                use_full_sample=use_full_sample,
                test_fraction=test_fraction,
            )
            combos = family_tuples(spec.family_space, k_max=k_max, k_min=k_min)
            for fnames in combos:
                for init in inits:
                    for lam in lambdas:
                        task_id = _safe_id(
                            [
                                spec.name,
                                "S" + str(split_id),
                                "K" + str(len(fnames)),
                                "+".join(fnames),
                                "L" + str(lam),
                                init,
                            ]
                        )
                        tasks.append(
                            {
                                "task_id": task_id,
                                "dataset": spec.name,
                                "split_id": split_id,
                                "split_seed": split_seed,
                                "prepared_path": prep_path,
                                "families": list(fnames),
                                "lambda": float(lam),
                                "init": init,
                                "seed": seed + len(tasks) + 17,
                                "max_iter": max_iter,
                                "tol": tol,
                                "n_starts": n_starts,
                                "active_threshold": active_threshold,
                            }
                        )

    print(f"Targeted real-data screen: {len(tasks)} tasks, n_jobs={n_jobs}", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_process_task)(task, cache_dir) for task in tasks)
    df = pd.DataFrame(results)
    _write_leaderboards(df, out_dir)
    print(f"Done in {(time.time() - t0) / 60.0:.2f} min. Outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
