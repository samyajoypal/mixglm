#!/usr/bin/env python3
"""Freeze the compact numerical record used by the CSDA manuscript.

This script reads the completed simulation and application output directories,
applies the manuscript's declared validity and admissibility rules, and writes
small publication-facing CSV and JSON files. It does not fit any model.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "parkinsons": {
        "full": "experiments/real_data/targeted_outputs/"
        "v6_parkinsons_reduced_final_20260820",
        "split": "experiments/real_data/targeted_outputs/"
        "v5_parkinsons_grouped_reduced_20260820/targeted_screen_raw.csv",
        "split_filter": None,
    },
    "rand": {
        "full": "experiments/real_data/targeted_outputs/"
        "rand_final_inference_v2_lower_penalty",
        "split": "experiments/real_data/targeted_outputs/"
        "v3_statscomp_refit_lower_penalty/targeted_screen_raw.csv",
        "split_filter": "randhealth_notmdvis_baseline",
    },
}


def bool_values(series: pd.Series, default: bool = False) -> pd.Series:
    values = series.fillna(default)
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    return values.astype(str).str.lower().eq("true")


def write_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, float_format="%.15g")


def valid_full_rows(data: pd.DataFrame) -> pd.DataFrame:
    valid = bool_values(data["refit_converged"])
    active = ~bool_values(data["has_intercept_only_component"], default=True)
    result = data[valid & active].copy()
    result["score"] = pd.to_numeric(result["selection_bic"], errors="coerce")
    return result[np.isfinite(result["score"])].copy()


def valid_split_rows(data: pd.DataFrame) -> pd.DataFrame:
    valid = bool_values(data["refit_converged"])
    active = ~bool_values(data["has_intercept_only_component"], default=True)
    stable = bool_values(data["refit_prediction_stable"])
    result = data[valid & active & stable].copy()
    result["score"] = pd.to_numeric(result["refit_bic"], errors="coerce")
    return result[np.isfinite(result["score"])].copy()


def family_search_tables(root: Path, outdir: Path) -> None:
    source = (
        root
        / "experiments/simulations/paper_outputs/"
        "v4_positive_repaired_20260624_b"
    )
    selection = pd.read_csv(source / "table_scenario_A.csv")

    for criterion in ("bic", "pred"):
        candidates = pd.read_csv(source / f"top10_{criterion}_all_reps.csv")
        candidates["replicate"] = candidates.groupby(["example_id", "n"])[
            "rank"
        ].transform(lambda ranks: ranks.eq(1).cumsum())
        inclusion = (
            candidates.groupby(["example_id", "n", "replicate"])[
                "is_equiv_structure"
            ]
            .any()
            .groupby(["example_id", "n"])
            .mean()
            .rename(f"Top10_{criterion.upper()}_Inclusion_Rate")
        )
        selection = selection.merge(inclusion, on=["example_id", "n"], how="left")

    write_csv(selection, outdir / "simulation_family_search.csv")
    write_csv(
        pd.read_csv(source / "table_scenario_B.csv"),
        outdir / "simulation_sparse.csv",
    )

    old_inference = pd.read_csv(source / "table_scenario_C_main.csv")
    old_inference = old_inference[
        old_inference["method"].eq("louis")
        & old_inference["example_id"].isin([1, 4])
    ].copy()
    gamma_inference = pd.read_csv(
        root
        / "paper_outputs/v7_gamma_lognormal_full_inference_20260817/"
        "table_scenario_C_main.csv"
    )
    columns = list(
        dict.fromkeys([*old_inference.columns.tolist(), *gamma_inference.columns.tolist()])
    )
    inference = pd.concat(
        [old_inference.reindex(columns=columns), gamma_inference.reindex(columns=columns)],
        ignore_index=True,
    ).sort_values(["example_id", "n"])
    write_csv(inference, outdir / "simulation_inference.csv")
    write_csv(
        pd.read_csv(root / "experiments/simulations/louis_validation_summary.csv"),
        outdir / "louis_validation_summary.csv",
    )


def model_class(data: pd.DataFrame) -> pd.Series:
    values = np.full(len(data), "Identical, K>1", dtype=object)
    values[data["K"].eq(1).to_numpy()] = "K=1"
    values[bool_values(data["nonidentical"]).to_numpy()] = "Non-identical"
    return pd.Series(values, index=data.index)


def application_tables(root: Path, outdir: Path) -> None:
    leaderboards: list[pd.DataFrame] = []
    full_winners: list[pd.DataFrame] = []
    split_winners: list[pd.DataFrame] = []
    supports: list[pd.DataFrame] = []
    lambdas: list[pd.DataFrame] = []
    parameters: list[pd.DataFrame] = []
    active_summaries: list[dict[str, object]] = []
    metadata: dict[str, object] = {}
    models: dict[str, object] = {}
    louis: dict[str, object] = {}

    for dataset, paths in DATASETS.items():
        full_dir = root / str(paths["full"])
        full = valid_full_rows(pd.read_csv(full_dir / "full_data_leaderboard_raw.csv"))
        unique = full.loc[full.groupby("families")["score"].idxmin()].sort_values(
            "score"
        )
        shown = unique.head(10).copy()
        shown.insert(0, "rank", np.arange(1, len(shown) + 1))
        shown["dataset"] = dataset
        leaderboards.append(
            shown[
                [
                    "dataset",
                    "rank",
                    "families",
                    "K",
                    "lambda",
                    "nonidentical",
                    "score",
                    "selection_loglik_train",
                    "active_counts",
                    "refit_pi",
                ]
            ]
        )

        full = full.copy()
        full["model_class"] = model_class(full)
        winners = full.loc[full.groupby("model_class")["score"].idxmin()].copy()
        winners["dataset"] = dataset
        full_winners.append(
            winners[
                [
                    "dataset",
                    "model_class",
                    "families",
                    "K",
                    "lambda",
                    "score",
                    "selection_loglik_train",
                    "active_counts",
                    "refit_pi",
                ]
            ]
        )

        split = pd.read_csv(root / str(paths["split"]))
        if paths["split_filter"] is not None:
            split = split[split["dataset"].eq(paths["split_filter"])].copy()
        split = valid_split_rows(split)
        split["model_class"] = model_class(split)
        for split_id, group in split.groupby("split_id"):
            for class_name in ("K=1", "Identical, K>1", "Non-identical"):
                candidates = group[group["model_class"].eq(class_name)]
                if candidates.empty:
                    raise RuntimeError(
                        f"No admissible {class_name} candidate for {dataset} split {split_id}"
                    )
                row = candidates.loc[candidates["score"].idxmin()].copy()
                split_winners.append(
                    pd.DataFrame(
                        {
                            "dataset": [dataset],
                            "split_id": [int(split_id)],
                            "model_class": [class_name],
                            "families": [row["families"]],
                            "K": [int(row["K"])],
                            "lambda": [float(row["lambda"])],
                            "bic": [float(row["score"])],
                            "test_loglik": [float(row["refit_test_loglik"])],
                            "n_test": [int(row["n_test"])],
                            "log_score": [
                                float(row["refit_test_loglik"]) / int(row["n_test"])
                            ],
                            "rmse": [float(row["refit_test_rmse"])],
                            "mae": [float(row["refit_test_mae"])],
                            "active_counts": [row["active_counts"]],
                        }
                    )
                )

        support = pd.read_csv(full_dir / "selection_frequencies.csv")
        support.insert(0, "dataset", dataset)
        supports.append(support)

        lambda_data = pd.read_csv(full_dir / "bootstrap_lambda_frequencies.csv")
        lambda_data.insert(0, "dataset", dataset)
        lambdas.append(lambda_data)

        parameter_data = pd.read_csv(full_dir / "bootstrap_parameter_summary.csv")
        parameter_data.insert(0, "dataset", dataset)
        parameters.append(parameter_data)

        draws = pd.read_csv(full_dir / "bootstrap_draws.csv")
        draws = draws[bool_values(draws["success"])].copy()
        counts = draws["active_counts"].map(ast.literal_eval)
        for component in range(max(map(len, counts))):
            values = np.asarray([row[component] for row in counts], dtype=float)
            active_summaries.append(
                {
                    "dataset": dataset,
                    "component": component,
                    "n_valid": len(values),
                    "q25": np.quantile(values, 0.25),
                    "median": np.quantile(values, 0.5),
                    "q75": np.quantile(values, 0.75),
                }
            )

        metadata[dataset] = json.loads((full_dir / "run_metadata.json").read_text())
        models[dataset] = json.loads((full_dir / "post_lasso_model.json").read_text())
        louis[dataset] = json.loads(
            (full_dir / "post_lasso_louis_diagnostics.json").read_text()
        )

    leaders = pd.concat(leaderboards, ignore_index=True)
    write_csv(leaders, outdir / "application_leaderboards.csv")
    write_csv(
        pd.concat(full_winners, ignore_index=True),
        outdir / "application_full_class_winners.csv",
    )

    split_data = pd.concat(split_winners, ignore_index=True).sort_values(
        ["dataset", "split_id", "model_class"]
    )
    write_csv(split_data, outdir / "application_split_class_winners.csv")
    means = (
        split_data.groupby(["dataset", "model_class"], as_index=False)[
            ["log_score", "rmse", "mae"]
        ]
        .mean()
        .sort_values(["dataset", "model_class"])
    )
    write_csv(means, outdir / "application_split_class_means.csv")

    contrasts: list[dict[str, object]] = []
    for (dataset, split_id), group in split_data.groupby(["dataset", "split_id"]):
        indexed = group.set_index("model_class")
        nonidentical = indexed.loc["Non-identical"]
        identical = indexed.loc["Identical, K>1"]
        contrasts.append(
            {
                "dataset": dataset,
                "split_id": split_id,
                "bic_advantage": identical["bic"] - nonidentical["bic"],
                "logscore_advantage": nonidentical["log_score"]
                - identical["log_score"],
                "rmse_advantage": identical["rmse"] - nonidentical["rmse"],
            }
        )
    write_csv(pd.DataFrame(contrasts), outdir / "application_split_contrasts.csv")
    write_csv(
        pd.concat(supports, ignore_index=True),
        outdir / "bootstrap_selection_frequencies.csv",
    )
    write_csv(
        pd.concat(lambdas, ignore_index=True),
        outdir / "bootstrap_lambda_frequencies.csv",
    )
    write_csv(
        pd.concat(parameters, ignore_index=True),
        outdir / "bootstrap_parameter_summary.csv",
    )
    write_csv(
        pd.DataFrame(active_summaries),
        outdir / "bootstrap_active_count_summary.csv",
    )
    for name, payload in (
        ("application_run_metadata.json", metadata),
        ("application_post_lasso_models.json", models),
        ("application_louis_diagnostics.json", louis),
    ):
        (outdir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/paper_figures/csda_data"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    family_search_tables(root, outdir)
    application_tables(root, outdir)
    print(f"Wrote the compact CSDA result record to {outdir}")


if __name__ == "__main__":
    main()
