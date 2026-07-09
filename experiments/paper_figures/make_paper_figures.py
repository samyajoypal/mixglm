#!/usr/bin/env python3
"""Build publication figures from saved paper result artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXAMPLE_LABELS = {
    1: "Gaussian--Student-t",
    2: "Gamma--lognormal",
    3: "Poisson--NB2",
    4: "Poisson--ZIP",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _short_family_name(name: str) -> str:
    replacements = {
        "student_t": "t",
        "skew_normal": "SN",
        "gaussian": "G",
        "poisson": "Pois",
        "nb2": "NB2",
        "zinb": "ZINB",
        "zip": "ZIP",
    }
    out = str(name)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out.replace("+", " + ")


def simulation_sample_size_figure(root: Path, outdir: Path) -> None:
    sim = root / "experiments/simulations/paper_outputs/v4_positive_repaired_20260624_b"
    a = pd.read_csv(sim / "table_scenario_A.csv")
    c = pd.read_csv(sim / "table_scenario_C_main.csv")
    c = c[c["method"].eq("louis")].copy()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex="col")
    ax = axes[0, 0]
    for ex, g in a.groupby("example_id"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["Oracle_Eq_Selection_Rate"], marker="o", label=EXAMPLE_LABELS[int(ex)])
    ax.set_ylabel("Equivalence recovery")
    ax.set_ylim(-0.03, 1.04)
    ax.set_title("Family-structure recovery")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    for ex, g in a.groupby("example_id"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["Test_LL_Selected"], marker="o", label=EXAMPLE_LABELS[int(ex)])
    ax.set_ylabel("Held-out log score")
    ax.set_title("Prediction score")

    ax = axes[1, 0]
    for ex, g in c.groupby("example_id"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["Mean_Abs_Bias"], marker="o", label=EXAMPLE_LABELS[int(ex)])
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("Mean absolute bias")
    ax.set_title("Coefficient accuracy")

    ax = axes[1, 1]
    for ex, g in c.groupby("example_id"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["Coverage"], marker="o", label=EXAMPLE_LABELS[int(ex)])
    ax.axhline(0.95, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.86, 0.99)
    ax.set_title("Nominal 95% intervals")

    fig.tight_layout()
    _save(fig, outdir, "simulation_sample_size")


def variable_selection_figure(root: Path, outdir: Path) -> None:
    sim = root / "experiments/simulations/paper_outputs/v4_positive_repaired_20260624_b"
    b = pd.read_csv(sim / "table_scenario_B.csv")
    b = b[b["penalty"].isin(["none", "lasso", "enet"])].copy()
    penalty_order = ["none", "lasso", "enet"]
    metric_order = ["TPR", "FPR", "FDR"]
    labels = {"none": "None", "lasso": "Lasso", "enet": "Elastic net"}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    x = np.arange(len(metric_order))
    width = 0.23
    for ax, (ex, g) in zip(axes, b.groupby("example_id")):
        for j, pen in enumerate(penalty_order):
            vals = g[g["penalty"].eq(pen)].iloc[0][metric_order].to_numpy(dtype=float)
            ax.bar(x + (j - 1) * width, vals, width=width, label=labels[pen])
        ax.set_xticks(x, metric_order)
        ax.set_ylim(0, 1.03)
        ax.set_title(EXAMPLE_LABELS[int(ex)])
        if ax is axes[0]:
            ax.set_ylabel("Rate")
        ax.grid(axis="y", color="0.9", linewidth=0.7)
    axes[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, outdir, "variable_selection")


def application_leaderboards_figure(root: Path, outdir: Path) -> None:
    park = pd.read_csv(
        root / "experiments/real_data/targeted_outputs/v2_less_sparse_k123/parkinsons_log_top20_bic.csv"
    ).head(10)
    rand = pd.read_csv(
        root / "experiments/real_data/targeted_outputs/rand_final_inference_v1/full_data_top50_bic.csv"
    ).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))
    for ax, data, title in [
        (axes[0], park, "Parkinson"),
        (axes[1], rand, "RAND"),
    ]:
        d = data.copy()
        d["label"] = d["families"].map(_short_family_name)
        d["delta"] = d["bic"] - d["bic"].min()
        d = d.iloc[::-1]
        colors = np.where(d["nonidentical"].to_numpy(dtype=bool), "#4477AA", "#999999")
        ax.barh(d["label"], d["delta"], color=colors)
        ax.set_title(title)
        ax.set_xlabel("Delta BIC from winner")
        ax.grid(axis="x", color="0.9", linewidth=0.7)
    fig.tight_layout()
    _save(fig, outdir, "application_leaderboards")


def _parse_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(str(value))


def application_support_figure(root: Path, outdir: Path) -> None:
    park = pd.read_csv(
        root / "experiments/real_data/targeted_outputs/v2_less_sparse_k123/parkinsons_log_top20_bic.csv"
    ).iloc[0]
    rand = pd.read_csv(
        root / "experiments/real_data/targeted_outputs/rand_final_inference_v1/full_data_top50_bic.csv"
    ).iloc[0]

    items = []
    for title, row in [("Parkinson BIC winner", park), ("RAND BIC winner", rand)]:
        component_features = _parse_list(row["active_features"])
        features = sorted({f for comp in component_features for f in comp})
        mat = np.zeros((len(features), len(component_features)), dtype=float)
        for j, comp in enumerate(component_features):
            selected = set(comp)
            for i, feat in enumerate(features):
                mat[i, j] = 1.0 if feat in selected else 0.0
        items.append((title, features, mat))

    height = max(3.2, 0.21 * max(len(items[0][1]), len(items[1][1])) + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, height))
    for ax, (title, features, mat) in zip(axes, items):
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title)
        ax.set_yticks(np.arange(len(features)), features)
        ax.set_xticks(np.arange(mat.shape[1]), [f"C{j + 1}" for j in range(mat.shape[1])])
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Component")
    fig.tight_layout()
    _save(fig, outdir, "application_supports")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    _setup_style()
    root = args.root.resolve()
    outdir = (root / args.outdir).resolve() if not args.outdir.is_absolute() else args.outdir
    simulation_sample_size_figure(root, outdir)
    variable_selection_figure(root, outdir)
    application_leaderboards_figure(root, outdir)
    application_support_figure(root, outdir)
    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()
