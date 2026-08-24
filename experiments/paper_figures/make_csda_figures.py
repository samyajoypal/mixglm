#!/usr/bin/env python3
"""Generate the sequential figures used by the CSDA submission.

The script reads the frozen simulation and real-data artifacts used in the
paper.  It does not refit any model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


EXAMPLE_LABELS = {
    1: "Gaussian--Student t",
    2: "Gamma--lognormal",
    3: "Poisson--NB2",
    4: "Poisson--ZIP",
}

COLORS = {
    1: "#2864A5",
    2: "#C44E52",
    3: "#2A8C82",
    4: "#D18F24",
    "nonidentical": "#2864A5",
    "identical": "#8C8C8C",
    "k1": "#C44E52",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def save_pdf(fig: plt.Figure, outdir: Path, number: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"fig{number}.pdf", bbox_inches="tight")
    plt.close(fig)


def short_family_name(value: object) -> str:
    names = {
        "gaussian": "G",
        "student_t": "t",
        "skew_normal": "SN",
        "poisson": "P",
        "nb2": "NB2",
        "zip": "ZIP",
        "zinb": "ZINB",
    }
    return " + ".join(names.get(part, part) for part in str(value).split("+"))


def figure_simulation(root: Path, outdir: Path) -> None:
    data_dir = root / "experiments/paper_figures/csda_data"
    selection = pd.read_csv(data_dir / "simulation_family_search.csv")
    inference = pd.read_csv(data_dir / "simulation_inference.csv")
    inference = inference[inference["example_id"].eq(2)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax = axes[0, 0]
    for example, group in selection.groupby("example_id"):
        group = group.sort_values("n")
        ax.plot(
            group["n"],
            group["Oracle_Eq_Selection_Rate"],
            marker="o",
            color=COLORS[int(example)],
            label=EXAMPLE_LABELS[int(example)],
        )
    ax.set_title("(a) Equivalent family-tuple recovery")
    ax.set_ylabel("Selection probability")
    ax.set_ylim(-0.02, 1.03)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axes[0, 1]
    for example, group in selection.groupby("example_id"):
        group = group.sort_values("n")
        ax.plot(
            group["n"],
            group["BIC_is_Best_Pred_Rate"],
            marker="o",
            color=COLORS[int(example)],
            label=EXAMPLE_LABELS[int(example)],
        )
    ax.set_title("(b) BIC winner is prediction winner")
    ax.set_ylabel("Agreement probability")
    ax.set_ylim(-0.02, 1.03)
    ax.grid(axis="y", color="0.90", linewidth=0.6)

    inference = inference.sort_values("n")
    ax = axes[1, 0]
    ax.plot(
        inference["n"],
        inference["Parameter_RMSE"],
        marker="o",
        color=COLORS[2],
    )
    ax.set_title("(c) Gamma--lognormal parameter accuracy")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Parameter RMSE")
    ax.grid(axis="y", color="0.90", linewidth=0.6)

    ax = axes[1, 1]
    coverage = inference["Coverage"].to_numpy(dtype=float)
    mcse = inference["Coverage_MCSE"].to_numpy(dtype=float)
    ax.errorbar(
        inference["n"],
        coverage,
        yerr=1.96 * mcse,
        marker="o",
        capsize=2.5,
        color=COLORS[2],
    )
    ax.axhline(0.95, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_title("(d) Gamma--lognormal Wald coverage")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Coverage of nominal 95% intervals")
    ax.set_ylim(0.91, 0.97)
    ax.grid(axis="y", color="0.90", linewidth=0.6)

    for ax in axes.ravel():
        ax.set_xticks(sorted(selection["n"].unique()) if ax in axes[0] else inference["n"])
    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    save_pdf(fig, outdir, 1)


def figure_variable_selection(root: Path, outdir: Path) -> None:
    path = root / "experiments/paper_figures/csda_data/simulation_sparse.csv"
    data = pd.read_csv(path)
    penalties = ["none", "lasso", "enet"]
    metrics = ["TPR", "FPR", "FDR"]
    labels = {"none": "None", "lasso": "Lasso", "enet": "Elastic net"}
    bar_colors = ["#8C8C8C", "#2864A5", "#D18F24"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    x = np.arange(len(metrics))
    width = 0.24
    for ax, example in zip(axes, [1, 2, 4]):
        group = data[data["example_id"].eq(example)]
        for j, penalty in enumerate(penalties):
            row = group[group["penalty"].eq(penalty)].iloc[0]
            ax.bar(
                x + (j - 1) * width,
                row[metrics].to_numpy(dtype=float),
                width,
                color=bar_colors[j],
                label=labels[penalty],
            )
        ax.set_xticks(x, ["TPR", "FPR", "FDR"])
        ax.set_title(EXAMPLE_LABELS[example])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="0.90", linewidth=0.6)
    axes[0].set_ylabel("Average rate")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.0)
    save_pdf(fig, outdir, 2)


def figure_application_leaderboards(root: Path, outdir: Path) -> None:
    data = pd.read_csv(
        root / "experiments/paper_figures/csda_data/application_leaderboards.csv"
    )
    park = data[data["dataset"].eq("parkinsons")].sort_values("rank")
    rand = data[data["dataset"].eq("rand")].sort_values("rank")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    for ax, data, title in [
        (axes[0], park, "(a) Parkinson's telemonitoring"),
        (axes[1], rand, "(b) RAND non-physician visits"),
    ]:
        shown = data.copy()
        shown["delta"] = shown["score"] - shown["score"].min()
        shown["label"] = shown["families"].map(short_family_name)
        shown = shown.iloc[::-1]
        colors = np.where(
            shown["nonidentical"].astype(bool),
            COLORS["nonidentical"],
            COLORS["identical"],
        )
        ax.barh(shown["label"], shown["delta"], color=colors, height=0.72)
        ax.set_title(title)
        ax.set_xlabel("BIC difference from winner")
        ax.grid(axis="x", color="0.90", linewidth=0.6)
    fig.tight_layout(w_pad=1.8)
    save_pdf(fig, outdir, 3)


def figure_split_validation(root: Path, outdir: Path) -> None:
    data = pd.read_csv(
        root / "experiments/paper_figures/csda_data/application_split_contrasts.csv"
    )
    contrasts = [
        data[data["dataset"].eq(dataset)].sort_values("split_id")
        for dataset in ("parkinsons", "rand")
    ]
    metrics = [
        ("bic_advantage", "BIC advantage"),
        ("logscore_advantage", "Log-score advantage"),
        ("rmse_advantage", "RMSE advantage"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex=True)
    for row, (data, row_label) in enumerate(
        zip(contrasts, ["Parkinson", "RAND"])
    ):
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            values = data[metric].to_numpy(dtype=float)
            colors = np.where(values >= 0, COLORS["nonidentical"], COLORS["identical"])
            ax.bar(data["split_id"], values, color=colors, width=0.68)
            ax.axhline(0, color="0.25", linewidth=0.7)
            ax.grid(axis="y", color="0.91", linewidth=0.55)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{row_label}\n(non-identical better $>$ 0)")
            if row == 1:
                ax.set_xlabel("Split")
            ax.set_xticks(range(5))
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    save_pdf(fig, outdir, 4)


def support_matrix(
    path: Path, dataset: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    data = data[data["dataset"].eq(dataset)].copy()
    selected_features = set(
        data.loc[data["selected_in_full_fit"].astype(bool), "feature"].astype(str)
    )
    data = data[data["feature"].astype(str).isin(selected_features)].copy()
    order = (
        data.groupby("feature")["feature_index"]
        .min()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    frequencies = (
        data.pivot(index="feature", columns="component", values="selection_frequency")
        .reindex(order)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    selected = (
        data.pivot(index="feature", columns="component", values="selected_in_full_fit")
        .reindex(order)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    return order, frequencies, selected


def figure_bootstrap_support(root: Path, outdir: Path) -> None:
    support_path = (
        root
        / "experiments/paper_figures/csda_data/bootstrap_selection_frequencies.csv"
    )
    items = [
        (
            "(a) Parkinson's telemonitoring",
            support_path,
            "parkinsons",
            ["Gaussian 1", "Gaussian 2", "Student t"],
        ),
        (
            "(b) RAND non-physician visits",
            support_path,
            "rand",
            ["Poisson", "NB2"],
        ),
    ]
    matrices = [
        (title, *support_matrix(path, dataset), labels)
        for title, path, dataset, labels in items
    ]
    max_rows = max(len(item[1]) for item in matrices)
    fig_height = max(5.2, 0.205 * max_rows + 1.3)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, fig_height))
    image = None
    for ax, (title, features, frequencies, selected, labels) in zip(axes, matrices):
        image = ax.imshow(
            frequencies,
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_yticks(np.arange(len(features)), features)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax.tick_params(length=0)
        for i in range(selected.shape[0]):
            for j in range(selected.shape[1]):
                if selected[i, j]:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            linewidth=0.8,
                        )
                    )
        for spine in ax.spines.values():
            spine.set_visible(False)
    assert image is not None
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.10, top=0.92, wspace=0.58)
    colorbar_axis = fig.add_axes([0.90, 0.20, 0.022, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Bootstrap selection frequency")
    save_pdf(fig, outdir, 5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--outdir", type=Path, default=Path("csda_submission"))
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir.resolve() if args.outdir.is_absolute() else (root / args.outdir)
    setup_style()
    figure_simulation(root, outdir)
    figure_variable_selection(root, outdir)
    figure_application_leaderboards(root, outdir)
    figure_split_validation(root, outdir)
    figure_bootstrap_support(root, outdir)
    print(f"Wrote fig1.pdf through fig5.pdf to {outdir}")


if __name__ == "__main__":
    main()
