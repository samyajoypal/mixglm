#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd

from mixglm.utils.repro import set_global_seed
from mixglm.sim.design import DesignConfig, make_design
from mixglm.sim.mixture import SimComponent, sample_mixture
from mixglm.sim.components import (
    gaussian_sampler, student_t_sampler, skew_normal_sampler,
    poisson_sampler, nb2_sampler
)

from mixglm.links.identity import IdentityLink
from mixglm.links.log import LogLink

from mixglm.model.component import ComponentSpec
from mixglm.model.mixture_glm import MixtureGLM
from mixglm.families.registry import FAMILIES, register_defaults as register_families
from mixglm.penalties.registry import PENALTIES, register_defaults as register_penalties
from mixglm.selection.full_pipeline import full_beam_pipeline
from mixglm.utils.parallel import ParallelConfig


def setup_example(example_id, n, p, sparsity=False, rng=None):
    """
    Constructs the design matrix X, the SimComponents, and pi_true for a given example.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine support and active params
    if sparsity:
        # p is large (e.g., 20), active is 5
        s = 5
        b0_active = np.array([1.0, 0.5, -0.7, 1.2, -0.4])
        b1_active = np.array([-0.5, -0.8, 0.9, -1.0, 0.6])
        beta0 = np.concatenate([b0_active, np.zeros(p - s)])
        beta1 = np.concatenate([b1_active, np.zeros(p - s)])
    else:
        # dense setting
        if p == 5:
            beta0 = np.array([1.0, 0.5, -0.7, 1.2, -0.4])
            beta1 = np.array([-0.5, -0.8, 0.9, -1.0, 0.6])
        else:
            # fallback
            beta0 = rng.normal(0, 0.5, size=p)
            beta1 = rng.normal(0, 0.5, size=p)
            beta0[0] = 1.0 # intercept
            beta1[0] = -0.5

    # Base configuration for continuous vs discrete
    if example_id in [1, 2]:
        # Continuous (Identity Link)
        X = make_design(DesignConfig(n=n, p=p, intercept=True, rho=0.2), rng=rng)
        link = IdentityLink()
        kind = "continuous"

        if example_id == 1:
            # Ex 1: Gaussian + Student-t
            families_true = ["gaussian", "student_t"]
            comps = [
                SimComponent(name="gaussian", beta=beta0, link=link, extra={"sigma": 1.0}, sampler=gaussian_sampler),
                SimComponent(name="student_t", beta=beta1, link=link, extra={"df": 4.0, "scale": 1.0}, sampler=student_t_sampler),
            ]
            pi_true = np.array([0.6, 0.4])

        elif example_id == 2:
            # Ex 2: Skew-Normal + Student-t
            families_true = ["skew_normal", "student_t"]
            comps = [
                SimComponent(name="skew_normal", beta=beta0, link=link, extra={"shape": 4.0, "scale": 1.5}, sampler=skew_normal_sampler),
                SimComponent(name="student_t", beta=beta1, link=link, extra={"df": 3.0, "scale": 1.0}, sampler=student_t_sampler),
            ]
            pi_true = np.array([0.5, 0.5])

    elif example_id in [3, 4]:
        # Discrete (Log Link) -> Need smaller X to prevent explosion
        X = rng.normal(scale=0.3, size=(n, p))
        X[:, 0] = 1.0 # Intercept
        link = LogLink()
        kind = "discrete"

        # Make betas smaller to prevent overflow in exp(X@beta)
        beta0 = beta0 * 0.5
        beta1 = beta1 * 0.5

        if example_id == 3:
            # Ex 3: Poisson + NB2 (Well Separated)
            beta0[0] = 1.5 # high mean
            beta1[0] = -0.5 # low mean
            families_true = ["poisson", "nb2"]
            comps = [
                SimComponent(name="poisson", beta=beta0, link=link, extra={}, sampler=poisson_sampler),
                SimComponent(name="nb2", beta=beta1, link=link, extra={"alpha": 0.8}, sampler=nb2_sampler),
            ]
            pi_true = np.array([0.6, 0.4])

        elif example_id == 4:
            # Ex 4: Poisson + NB2 (Highly Overlapped)
            beta0[0] = 0.5
            beta1[0] = 0.5
            # Make the rest of betas very similar
            beta1 = beta0 + rng.normal(0, 0.1, size=p)
            beta1[0] = 0.5
            families_true = ["poisson", "nb2"]
            comps = [
                SimComponent(name="poisson", beta=beta0, link=link, extra={}, sampler=poisson_sampler),
                SimComponent(name="nb2", beta=beta1, link=link, extra={"alpha": 1.5}, sampler=nb2_sampler),
            ]
            pi_true = np.array([0.7, 0.3])

    else:
        raise ValueError("Invalid example_id")

    sim = sample_mixture(X=X, components=comps, pi=pi_true, rng=rng)

    return sim, X, kind, comps, pi_true, families_true


def run_scenario_a(example_id, n, task_id, rng):
    """
    Model Selection (Identifying K and Families)
    """
    p = 5
    sim, X, kind, comps, pi_true, families_true = setup_example(example_id, n, p, sparsity=False, rng=rng)

    t0 = time.time()
    best, all_bests = full_beam_pipeline(
        y=sim.y, X=X,
        kind=kind,
        K_max=3,
        beam_width=5,
        criterion="bic",
        do_none=True,
        ridge_grid=[], lasso_grid=[], enet_grid=[], # Focus on unpenalized for structure selection
        em_kwargs={"max_iter": 150, "tol": 1e-5, "n_starts": 3},
        seed=task_id,
        init="quantile" if kind=="continuous" else "random",
        compute_icl=False,
        standardize=True,
        verbose=False,
        parallel=ParallelConfig(n_jobs=1), # If running in array, each task gets 1 core
        show_progress=False,
    )
    t1 = time.time()

    # Analyze all_bests to find rank of true model
    true_K = len(families_true)
    true_fam_set = set(families_true)

    models = sorted(all_bests, key=lambda x: x.score)
    top_models = []
    true_model_rank = -1

    for i, m in enumerate(models[:10]): # Top 10
        fam_set = set(m.families)
        is_true = (m.K == true_K and fam_set == true_fam_set)
        if is_true and true_model_rank == -1:
            true_model_rank = i + 1

        top_models.append({
            "rank": i + 1,
            "K": m.K,
            "families": list(m.families),
            "bic": m.score,
            "is_true_structure": is_true
        })

    res = {
        "scenario": "A",
        "example_id": example_id,
        "n": n,
        "task_id": task_id,
        "time": t1 - t0,
        "true_K": true_K,
        "true_families": families_true,
        "best_K": best.K,
        "best_families": list(best.families),
        "true_model_rank": true_model_rank, # -1 if not in top 10
        "top_models": top_models
    }
    return res


def run_scenario_b(example_id, task_id, rng):
    """
    High-Dimensional Variable Selection
    """
    n = 1000
    p = 20
    sim, X, kind, comps, pi_true, families_true = setup_example(example_id, n, p, sparsity=True, rng=rng)

    # We fix the true families and K, and only evaluate penalties
    K = len(families_true)

    results_penalties = []

    for pen_name in ["none", "lasso", "enet"]:
        # Build model with specific penalty
        mc = []
        for c in comps:
            if pen_name == "none":
                pen = PENALTIES.create("none")
            elif pen_name == "lasso":
                # unnormalized likelihood needs large lambda
                pen = PENALTIES.create("lasso", lam=20.0)
            elif pen_name == "enet":
                pen = PENALTIES.create("elastic_net", lam=20.0, l1_ratio=0.5)

            mc.append(ComponentSpec(
                family=FAMILIES.create(c.name),
                link=c.link,
                penalty=pen
            ))

        model = MixtureGLM(components=mc)

        t0 = time.time()
        model.fit(
            y=sim.y, X=X,
            max_iter=150, tol=1e-5, n_starts=3, seed=task_id,
            init="quantile" if kind=="continuous" else "random",
            standardize=True, verbose=False
        )
        t1 = time.time()

        if not model.result_.converged:
            continue

        betas_hat = model.betas_original_scale()

        # Align labels
        b0t, b1t = comps[0].beta, comps[1].beta
        b0h, b1h = betas_hat[0], betas_hat[1]

        d00 = np.linalg.norm(b0t - b0h) + np.linalg.norm(b1t - b1h)
        d01 = np.linalg.norm(b0t - b1h) + np.linalg.norm(b1t - b0h)
        if d01 < d00:
            b0h, b1h = b1h, b0h

        # Calculate TPR / FPR (Threshold = 1e-4)
        active0 = (b0t != 0)
        active1 = (b1t != 0)
        pred_active0 = (np.abs(b0h) > 1e-4)
        pred_active1 = (np.abs(b1h) > 1e-4)

        tpr0 = np.sum(active0 & pred_active0) / np.sum(active0)
        fpr0 = np.sum(~active0 & pred_active0) / np.sum(~active0)
        tpr1 = np.sum(active1 & pred_active1) / np.sum(active1)
        fpr1 = np.sum(~active1 & pred_active1) / np.sum(~active1)

        mse0 = np.mean((b0t - b0h)**2)
        mse1 = np.mean((b1t - b1h)**2)

        results_penalties.append({
            "penalty": pen_name,
            "bic": model.result_.bic,
            "time": t1 - t0,
            "tpr_avg": (tpr0 + tpr1) / 2.0,
            "fpr_avg": (fpr0 + fpr1) / 2.0,
            "mse_avg": (mse0 + mse1) / 2.0,
            "hat_zeros": int(np.sum(~pred_active0) + np.sum(~pred_active1))
        })

    res = {
        "scenario": "B",
        "example_id": example_id,
        "n": n,
        "p": p,
        "task_id": task_id,
        "penalties": results_penalties
    }
    return res


def run_scenario_c(example_id, n, task_id, rng):
    """
    Inference (SE, Coverage, CI Length)
    """
    p = 5
    sim, X, kind, comps, pi_true, families_true = setup_example(example_id, n, p, sparsity=False, rng=rng)

    # We fit the true unpenalized model
    mc = []
    for c in comps:
        mc.append(ComponentSpec(
            family=FAMILIES.create(c.name),
            link=c.link,
            penalty=PENALTIES.create("none")
        ))

    model = MixtureGLM(components=mc)
    t0 = time.time()
    model.fit(
        y=sim.y, X=X,
        max_iter=150, tol=1e-5, n_starts=3, seed=task_id,
        init="quantile" if kind=="continuous" else "random",
        standardize=True, verbose=False
    )
    t1 = time.time()

    if not model.result_.converged:
        return {"error": "Did not converge", "task_id": task_id, "scenario": "C"}

    try:
        df_inf, _ = model.inference_table(y=sim.y, X=X, method='numeric')
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}", "task_id": task_id, "scenario": "C"}

    # We need to match the parameters. Note: inference_table is on FIT scale (standardized).
    # Since we want to test coverage, we should extract the true parameters on the FIT scale,
    # OR we just test coverage of the intercept/scale which are easier.
    # Actually, if we standardized X internally, `beta[k][j]` represents the scaled effect.
    # We can fetch the scaler and transform the true betas to the fit scale.
    scaler = model.scaler_
    b0t_fit = scaler.beta_to_fit_scale(comps[0].beta)
    b1t_fit = scaler.beta_to_fit_scale(comps[1].beta)

    # Align labels via original scale just to be sure
    betas_hat = model.betas_original_scale()
    d00 = np.linalg.norm(comps[0].beta - betas_hat[0]) + np.linalg.norm(comps[1].beta - betas_hat[1])
    d01 = np.linalg.norm(comps[0].beta - betas_hat[1]) + np.linalg.norm(comps[1].beta - betas_hat[0])

    swapped = False
    if d01 < d00:
        b0t_fit, b1t_fit = b1t_fit, b0t_fit
        swapped = True

    # Build dictionary of true parameters on fit scale
    true_params = {}

    # pi (in eta space: log(pi[0]/pi[-1])) - wait, numeric_se uses isometric log-ratio or similar.
    # We can skip pi coverage for simplicity, and focus on betas.
    for j in range(p):
        true_params[f"beta[{0 if not swapped else 1}][{j}]"] = float(b0t_fit[j])
        true_params[f"beta[{1 if not swapped else 0}][{j}]"] = float(b1t_fit[j])

    coverage_results = []

    for _, row in df_inf.iterrows():
        param = row['param']
        if param in true_params:
            truth = true_params[param]
            est = row['estimate']
            se = row['se']
            ci_lo = row['ci2.5%']
            ci_hi = row['ci97.5%']

            covers = bool(ci_lo <= truth <= ci_hi)
            ci_length = float(ci_hi - ci_lo)
            bias = float(est - truth)

            coverage_results.append({
                "param": param,
                "truth": truth,
                "estimate": float(est),
                "se": float(se),
                "covers": covers,
                "ci_length": ci_length,
                "bias": bias
            })

    res = {
        "scenario": "C",
        "example_id": example_id,
        "n": n,
        "task_id": task_id,
        "time": t1 - t0,
        "coverage": coverage_results
    }
    return res


def main():
    parser = argparse.ArgumentParser(description="Run HPC Simulation for MixGLM")
    parser.add_argument("--scenario", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--example", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    register_families()
    register_penalties()

    # Base seed calculation to avoid overlap across scenarios/examples/samples/tasks
    # e.g., mapping scenario to int
    scen_map = {"A": 100000, "B": 200000, "C": 300000}
    base_seed = scen_map[args.scenario] + args.example * 10000 + args.n_samples + args.task_id

    set_global_seed(base_seed)
    rng = np.random.default_rng(base_seed)

    sys.path.insert(0, './src')

    if args.scenario == "A":
        res = run_scenario_a(args.example, args.n_samples, args.task_id, rng)
    elif args.scenario == "B":
        res = run_scenario_b(args.example, args.task_id, rng)
    elif args.scenario == "C":
        res = run_scenario_c(args.example, args.n_samples, args.task_id, rng)

    out_file = os.path.join(args.out_dir, f"res_{args.scenario}_ex{args.example}_n{args.n_samples}_task{args.task_id}.json")
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
