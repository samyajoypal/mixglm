import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

sys.path.insert(0, './src')
from mixglm.utils.repro import set_global_seed
from mixglm.families.registry import FAMILIES, register_defaults as register_families
from mixglm.penalties.registry import PENALTIES, register_defaults as register_penalties
from mixglm.links.registry import LINKS, register_defaults as register_links
from mixglm.model.component import ComponentSpec
from mixglm.model.mixture_glm import MixtureGLM
from mixglm.links.registry import LINKS
from mixglm.penalties.lasso import LassoPenalty

warnings.filterwarnings("ignore")

def compute_r2(model, y_test, X_test):
    try:
        y_pred = model.predict_mean(X_test)
        mse = np.mean((y_test - y_pred)**2)
        var = np.var(y_test)
        if var == 0: return float('-inf')
        return float(1 - (mse / var))
    except:
        return float('-inf')

def compute_test_nll(model, y_test, X_test):
    try:
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
    except:
        return float('inf')

def process_task(task_args, cache_dir):
    register_families()
    register_links()
    register_penalties()
    dataset_name, kind, K, fnames, lam, X_file, y_file = task_args
    task_id = f"{dataset_name.replace(' ', '_')}_K{K}_{'_'.join(fnames)}_L{lam}.json"
    cache_path = os.path.join(cache_dir, task_id)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    print(f"[{dataset_name}] STARTING | K={K} | Fam={fnames} | Lambda={lam}", flush=True)
    t0 = time.time()

    X = np.load(X_file)
    y = np.load(y_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    comps = []
    for fname in fnames:
        fam = FAMILIES.create(fname)
        link = LINKS.create(fam.default_link_name)
        comps.append(ComponentSpec(family=fam, link=link, penalty=LassoPenalty(lam=lam)))

    model = MixtureGLM(comps)
    model.fit(y_train, X_train, max_iter=150, tol=1e-4, n_starts=3,
              init="quantile" if kind=="continuous" else "random",
              standardize=True, verbose=False)

    t1 = time.time()

    if model.result_.converged:
        test_nll = compute_test_nll(model, y_test, X_test)
        r2 = compute_r2(model, y_test, X_test)

        # Save active vars
        active_vars = []
        betas = model.betas_original_scale()
        for k in range(K):
            active = int(np.sum(np.abs(betas[k]) > 1e-4))
            active_vars.append(active)

        res = {
            "dataset": dataset_name, "K": K, "families": fnames, "lam": lam,
            "converged": True, "bic": float(model.result_.bic),
            "test_nll": test_nll, "test_r2": r2,
            "active_vars": active_vars, "time": t1-t0
        }
        print(f"[{dataset_name}] SUCCESS | K={K} | Fam={fnames} | Lam={lam} | BIC: {res['bic']:.2f} | R2: {res['test_r2']:.4f}", flush=True)
    else:
        res = {
            "dataset": dataset_name, "K": K, "families": fnames, "lam": lam,
            "converged": False, "bic": float('inf'), "test_nll": float('inf'), "test_r2": float('-inf'),
            "active_vars": [], "time": t1-t0
        }
        print(f"[{dataset_name}] FAILED | K={K} | Fam={fnames} | Lam={lam}", flush=True)

    with open(cache_path, "w") as f:
        json.dump(res, f)

    return res

def main():
    register_families()
    register_links()
    register_penalties()
    set_global_seed(42)

    cache_dir = "experiments/real_data/real_outputs/checkpoints"
    os.makedirs(cache_dir, exist_ok=True)

    tasks = []
    lambdas = [1.0, 5.0, 10.0, 20.0, 50.0]

    # 1. Crime dataset
    if os.path.exists("data/crime_X.npy"):
        for K in [2, 3]:
            # Always Student-T for heavy tails
            fnames = tuple(["student_t"] * K)
            for lam in lambdas:
                tasks.append(("crime", "continuous", K, fnames, lam, "data/crime_X.npy", "data/crime_y.npy"))

    # 2. BlogFeedback dataset
    if os.path.exists("data/blog_X.npy"):
        for K in [2, 3]:
            structs = [
                tuple(["poisson"] * K),
                tuple(["nb2"] * K),
                tuple(["zip"] * K)
            ]
            if K == 2: structs.append(("poisson", "nb2"))
            if K == 2: structs.append(("poisson", "zip"))

            for fnames in structs:
                for lam in lambdas:
                    tasks.append(("blog", "counts", K, fnames, lam, "data/blog_X.npy", "data/blog_y.npy"))

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Total evaluated models in grid: {len(tasks)}. Parallel cores: {n_jobs}", flush=True)

    results = Parallel(n_jobs=n_jobs)(delayed(process_task)(t, cache_dir) for t in tasks)

    for dataset in ["crime", "blog"]:
        d_res = [r for r in results if r["dataset"] == dataset and r["converged"]]
        if not d_res: continue

        df = pd.DataFrame(d_res)
        df = df.sort_values("bic")

        out_file = f"experiments/real_data/real_outputs/{dataset}_K23_leaderboard.csv"
        df.to_csv(out_file, index=False)
        print(f"\nLeaderboard for {dataset} saved to {out_file}", flush=True)
        print(df.head(5)[["K", "families", "lam", "bic", "test_r2", "active_vars"]].to_string(), flush=True)

if __name__ == "__main__":
    main()
