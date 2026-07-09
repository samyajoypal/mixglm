import os
import glob
import json
import pandas as pd
import numpy as np

def aggregate_scenario_a(files):
    records = []
    for f in files:
        with open(f, 'r') as fp:
            d = json.load(fp)
        if d.get("scenario") != "A": continue

        records.append({
            "example_id": d["example_id"],
            "n": d["n"],
            "true_K": d["true_K"],
            "true_model_rank": d["true_model_rank"],
            "is_correct_top1": 1 if d["true_model_rank"] == 1 else 0
        })

    if not records: return
    df = pd.DataFrame(records)

    print("\n=== SCENARIO A: Model Selection ===")
    summary = df.groupby(["example_id", "n"]).agg(
        num_reps=("example_id", "count"),
        prop_correct_top1=("is_correct_top1", "mean"),
        avg_rank=("true_model_rank", lambda x: np.mean([v for v in x if v > 0]))
    ).reset_index()
    print(summary.to_string(index=False))


def aggregate_scenario_b(files):
    records = []
    for f in files:
        with open(f, 'r') as fp:
            d = json.load(fp)
        if d.get("scenario") != "B": continue

        for p in d["penalties"]:
            records.append({
                "example_id": d["example_id"],
                "n": d["n"],
                "penalty": p["penalty"],
                "mse": p["mse_avg"],
                "tpr": p["tpr_avg"],
                "fpr": p["fpr_avg"],
                "hat_zeros": p["hat_zeros"]
            })

    if not records: return
    df = pd.DataFrame(records)

    print("\n=== SCENARIO B: High-Dimensional Variable Selection ===")
    summary = df.groupby(["example_id", "n", "penalty"]).agg(
        num_reps=("example_id", "count"),
        mse=("mse", "mean"),
        tpr=("tpr", "mean"),
        fpr=("fpr", "mean"),
        hat_zeros=("hat_zeros", "mean")
    ).reset_index()
    print(summary.to_string(index=False))


def aggregate_scenario_c(files):
    records = []
    for f in files:
        with open(f, 'r') as fp:
            d = json.load(fp)
        if d.get("scenario") != "C": continue
        if "error" in d: continue

        for cov in d["coverage"]:
            records.append({
                "example_id": d["example_id"],
                "n": d["n"],
                "param": cov["param"],
                "bias": cov["bias"],
                "se": cov["se"],
                "coverage": 1 if cov["covers"] else 0,
                "ci_length": cov["ci_length"]
            })

    if not records: return
    df = pd.DataFrame(records)

    print("\n=== SCENARIO C: Inference (Coverage & SE) ===")
    summary = df.groupby(["example_id", "n", "param"]).agg(
        num_reps=("example_id", "count"),
        mean_bias=("bias", "mean"),
        mean_se=("se", "mean"),
        coverage=("coverage", "mean"),
        mean_ci_len=("ci_length", "mean")
    ).reset_index()
    print(summary.to_string(index=False))


def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    all_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not all_files:
        print("No json results found.")
        return

    aggregate_scenario_a(all_files)
    aggregate_scenario_b(all_files)
    aggregate_scenario_c(all_files)

if __name__ == "__main__":
    main()
