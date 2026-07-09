import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, './src')
from mixglm.utils.repro import set_global_seed
from mixglm.families.registry import register_defaults as register_families
from mixglm.penalties.registry import register_defaults as register_penalties
from mixglm.selection.full_pipeline import full_beam_pipeline
from mixglm.utils.parallel import ParallelConfig

# Suppress harmless SciPy/NumPy warnings during EM exploration
warnings.filterwarnings("ignore")

def compute_test_nll(model, y_test, X_test):
    # Get responsibilities log-likelihood
    try:
        # P(Y|X) = sum_k pi_k f_k(y | x)
        X_use = model._X_fit_scale(X_test)
        n = y_test.shape[0]
        K = len(model.components)
        log_terms = np.empty((n, K), dtype=float)

        for k, comp in enumerate(model.components):
            mu = comp.link.inverse(X_use @ model.result_.betas[k])
            ll = comp.family.loglik_component(y=y_test, mu=mu, extra=model.result_.extras[k])
            log_terms[:, k] = np.log(np.clip(model.result_.pi[k], 1e-300, 1.0)) + ll

        m = np.max(log_terms, axis=1, keepdims=True)
        log_prob = m + np.log(np.sum(np.exp(log_terms - m), axis=1, keepdims=True))
        return float(-np.sum(log_prob))
    except Exception as e:
        return np.inf

def run_dataset(name, X_file, y_file, feat_file, kind, K_max=2):
    print(f"\n==========================================")
    print(f"Running Analysis on: {name}")

    X = np.load(X_file)
    y = np.load(y_file)
    with open(feat_file, "r") as f:
        feature_names = json.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {n_jobs} cores for parallel search.")

    lasso_grid = [1.0, 5.0, 10.0, 20.0, 50.0]

    t0 = time.time()
    best, all_models = full_beam_pipeline(
        y=y_train, X=X_train, kind=kind, K_max=K_max, beam_width=4,
        criterion="bic", do_none=True,
        lasso_grid=lasso_grid, ridge_grid=[], enet_grid=[],
        em_kwargs={"max_iter": 100, "tol": 1e-4, "n_starts": 3},
        init="quantile" if kind=="continuous" else "random",
        standardize=True, verbose=False, show_progress=False,
        parallel=ParallelConfig(n_jobs=n_jobs, backend="loky"),
        tuning_n_jobs=n_jobs
    )
    t1 = time.time()

    # Process all models
    model_stats = []
    for m in all_models:
        if not m.model.result_.converged:
            continue
        test_nll = compute_test_nll(m.model, y_test, X_test)
        y_pred = m.model.predict_mean(X_test)
        test_mse = float(np.mean((y_test - y_pred)**2))

        model_stats.append({
            "families": " + ".join(m.families),
            "penalty": m.penalty,
            "bic": m.score,
            "test_nll": test_nll,
            "test_mse": test_mse,
            "k": len(m.families)
        })

    df_models = pd.DataFrame(model_stats)
    top10_bic = df_models.sort_values("bic").head(10)

    if kind == "continuous":
        top10_pred = df_models.sort_values("test_mse").head(10)
    else:
        top10_pred = df_models.sort_values("test_nll").head(10)

    out_dir = "experiments/real_data/real_outputs"
    os.makedirs(out_dir, exist_ok=True)

    safe_name = name.replace(" ", "_").replace("&", "and")
    top10_bic.to_csv(f"{out_dir}/{safe_name}_top10_bic.csv", index=False)
    top10_pred.to_csv(f"{out_dir}/{safe_name}_top10_pred.csv", index=False)

    # Extract features for best model
    betas = best.model.betas_original_scale()
    active_vars_info = {}
    for k, b in enumerate(betas):
        active_idx = np.where(np.abs(b) > 1e-4)[0]
        # Safety check for feature bounds
        active_names = [feature_names[i] if i < len(feature_names) else f"Feat_{i}" for i in active_idx]
        active_vars_info[f"Component_{k+1}"] = active_names

    with open(f"{out_dir}/{safe_name}_active_features.json", "w") as f:
        json.dump(active_vars_info, f, indent=2)

    # Perform inference for the best model to get standard errors
    print("Computing numeric standard errors for the best model...")
    try:
        inf_df, _ = best.model.inference_table(y_train, X_train, method="numeric")
        inf_df.to_csv(f"{out_dir}/{safe_name}_inference.csv", index=False)
    except Exception as e:
        print(f"Inference failed: {e}")

    res = {
        "dataset": name,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "p": X.shape[1],
        "best_families": best.families,
        "best_penalty": best.penalty,
        "bic": best.score,
        "time": t1-t0
    }
    return res

def main():
    register_families()
    register_penalties()
    set_global_seed(42)

    os.makedirs("experiments/real_data/real_outputs", exist_ok=True)
    results = []

    import sys
    dataset_to_run = sys.argv[1] if len(sys.argv) > 1 else "all"

    # 1. Communities and Crime (Continuous)
    if (dataset_to_run in ["all", "crime"]) and os.path.exists("data/crime_X.npy"):
        res_crime = run_dataset("Communities & Crime", "data/crime_X.npy", "data/crime_y.npy", "data/crime_features.json", kind="continuous", K_max=2)
        results.append(res_crime)

    # 2. BlogFeedback (Counts)
    if (dataset_to_run in ["all", "blog"]) and os.path.exists("data/blog_X.npy"):
        res_blog = run_dataset("BlogFeedback", "data/blog_X.npy", "data/blog_y.npy", "data/blog_features.json", kind="counts", K_max=2)
        results.append(res_blog)

    # Save results to JSON
    out_json = f"experiments/real_data/real_outputs/real_data_results_{dataset_to_run}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
