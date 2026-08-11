import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from pathlib import Path

sys.path.insert(0, './src')
from mixglm.utils.repro import set_global_seed
from mixglm.families.registry import FAMILIES, register_defaults as register_families
from mixglm.penalties.registry import PENALTIES, register_defaults as register_penalties
from mixglm.model.component import ComponentSpec
from mixglm.model.mixture_glm import MixtureGLM
from mixglm.links.identity import IdentityLink
from mixglm.links.log import LogLink

from mixglm.sim.components import gaussian_sampler, student_t_sampler, gamma_sampler, lognormal_sampler, poisson_sampler, nb2_sampler, zip_sampler
from mixglm.sim.design import DesignConfig, make_design
from mixglm.sim.mixture import SimComponent, sample_mixture

import mixglm.selection.full_pipeline as fp
from mixglm.selection.beam_search import beam_search_models

warnings.filterwarnings("ignore")

SIM_VERSION = "v4_positive_repaired_louis"
OUTPUT_ROOT = os.environ.get("MIXGLM_OUTPUT_ROOT", os.path.join("paper_outputs", SIM_VERSION))


def _parse_int_list(env_name, default):
    raw = os.environ.get(env_name)
    if raw is None:
        return list(default)
    if raw.strip() == "":
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(env_name, default):
    raw = os.environ.get(env_name)
    if raw is None:
        return list(default)
    if raw.strip() == "":
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


SAMPLE_SIZES = _parse_int_list("MIXGLM_SAMPLE_SIZES", [500, 1000, 1500])
SCENARIO_A_EXAMPLES = _parse_int_list("MIXGLM_SCENARIO_A_EXAMPLES", [1, 2, 3, 4])
SCENARIO_B_EXAMPLES = _parse_int_list("MIXGLM_SCENARIO_B_EXAMPLES", [1, 2, 4])
SCENARIO_C_EXAMPLES = _parse_int_list("MIXGLM_SCENARIO_C_EXAMPLES", [1, 2, 4])
LAMBDA_GRID = _parse_float_list("MIXGLM_LAMBDA_GRID", [0.25, 0.5, 1, 2, 5, 10, 20, 50])
SCENARIO_B_N = int(os.environ.get("MIXGLM_SCENARIO_B_N", 1000))
SCENARIO_B_P = int(os.environ.get("MIXGLM_SCENARIO_B_P", 20))
SCENARIO_A_BEAM_WIDTH = int(os.environ.get("MIXGLM_SCENARIO_A_BEAM_WIDTH", 10))
SCENARIO_C_N_STARTS = int(os.environ.get("MIXGLM_SCENARIO_C_N_STARTS", 2))
SCENARIO_C_MAX_ITER = int(os.environ.get("MIXGLM_SCENARIO_C_MAX_ITER", 100))

# Boundary-equivalent classes are reported separately from strict recovery.
BOUNDARY_EQUIV_CLASSES = {
    frozenset(["poisson", "nb2"]): [frozenset(["nb2", "nb2"]), frozenset(["poisson", "nb2"])],
    frozenset(["poisson", "zip"]): [frozenset(["zip", "zip"]), frozenset(["poisson", "zip"])],
    frozenset(["gaussian", "student_t"]): [frozenset(["student_t", "student_t"]), frozenset(["gaussian", "student_t"])],
    frozenset(["gamma", "lognormal"]): [frozenset(["gamma", "lognormal"])]
}

def is_strict_structure(selected, truth):
    return family_signature(selected) == family_signature(truth)


def is_boundary_equivalent(selected, truth):
    t_set = frozenset(truth)
    s_set = frozenset(selected)
    if t_set in BOUNDARY_EQUIV_CLASSES:
        return s_set in BOUNDARY_EQUIV_CLASSES[t_set]
    return s_set == t_set

def family_signature(families):
    return tuple(sorted(str(f) for f in families))


def init_for_kind(kind):
    if kind == "counts":
        return "random"
    if kind == "positive":
        return "kmeans_glm"
    return "quantile"

def test_loglik(model, y, X):
    try:
        X_use = model._X_fit_scale(X)
        n = y.shape[0]
        log_terms = np.empty((n, len(model.components)))
        for k, comp in enumerate(model.components):
            mu = comp.link.inverse(X_use @ model.result_.betas[k])
            ll = comp.family.loglik_component(y=y, mu=mu, extra=model.result_.extras[k])
            log_terms[:, k] = np.log(np.clip(model.result_.pi[k], 1e-300, 1.0)) + ll
        m = np.max(log_terms, axis=1)
        return float(np.mean(m + np.log(np.sum(np.exp(log_terms - m[:, None]), axis=1))))
    except:
        return float('-inf')


def _latex_escape(value):
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    return text


def dataframe_to_latex(df, *, caption, label, float_format="%.3f"):
    cols = list(df.columns)
    align = "l" * len(cols)
    lines = [
        r"\begin{table}",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(float_format % float(v))
            else:
                vals.append(_latex_escape(v))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def mcse_rate(x) -> float:
    vals = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return np.nan
    p = float(np.mean(vals))
    return float(np.sqrt(p * (1.0 - p) / vals.size))


def median_positive_rank(x) -> float:
    vals = np.asarray([v for v in x if np.isfinite(v) and float(v) > 0], dtype=float)
    return float(np.median(vals)) if vals.size else np.nan

def setup_example(example_id, n, p, sparsity=False, rng=None):
    if rng is None: rng = np.random.default_rng()

    if sparsity:
        s = 5
        b0_active = np.array([1.0, 0.8, -0.6, 1.2, -0.5])
        b1_active = np.array([-0.8, -1.0, 0.9, -1.2, 0.7])
        beta0 = np.concatenate([b0_active, np.zeros(p - s)])
        beta1 = np.concatenate([b1_active, np.zeros(p - s)])
    else:
        if p == 5:
            beta0 = np.array([1.0, 0.5, -0.7, 1.2, -0.4])
            beta1 = np.array([-0.5, -0.8, 0.9, -1.0, 0.6])
        else:
            beta0 = rng.normal(0, 0.5, size=p); beta0[0] = 1.0
            beta1 = rng.normal(0, 0.5, size=p); beta1[0] = -0.5

    if example_id == 1:
        X = make_design(DesignConfig(n=n, p=p, intercept=True, rho=0.2), rng=rng)
        kind = "real"
        families_true = ["gaussian", "student_t"]
        comps = [
            SimComponent(name="gaussian", beta=beta0, link=IdentityLink(), extra={"sigma": 1.0}, sampler=gaussian_sampler),
            SimComponent(name="student_t", beta=beta1, link=IdentityLink(), extra={"df": 4.0, "scale": 1.0}, sampler=student_t_sampler),
        ]
    elif example_id == 2:
        X = make_design(DesignConfig(n=n, p=p, intercept=True, rho=0.2), rng=rng)
        kind = "positive"
        families_true = ["gamma", "lognormal"]
        if p == 5:
            beta0 = np.array([-0.60, 0.20, -0.15, 0.15, -0.10])
            beta1 = np.array([1.00, -0.35, 0.30, -0.25, 0.20])
        else:
            b0_active = np.array([-0.60, 0.20, -0.15, 0.15, -0.10])
            b1_active = np.array([1.00, -0.35, 0.30, -0.25, 0.20])
            beta0 = np.concatenate([b0_active, np.zeros(p - b0_active.size)])
            beta1 = np.concatenate([b1_active, np.zeros(p - b1_active.size)])
        comps = [
            SimComponent(name="gamma", beta=beta0, link=LogLink(), extra={"shape": 30.0}, sampler=gamma_sampler),
            SimComponent(name="lognormal", beta=beta1, link=IdentityLink(), extra={"sigma": 1.0}, sampler=lognormal_sampler),
        ]
    elif example_id == 3:
        # FIXED: Means heavily separated and dispersion high for clear identifiability
        X = make_design(DesignConfig(n=n, p=p, intercept=True, rho=0.2), rng=rng)
        kind = "counts"
        b0 = np.array([2.5, 0.3, -0.3, 0.3, -0.3]) if p==5 else np.concatenate([[2.5], beta0[1:]*0.5])
        b1 = np.array([0.0, -0.3, 0.3, -0.3, 0.3]) if p==5 else np.concatenate([[0.0], beta1[1:]*0.5])
        families_true = ["poisson", "nb2"]
        comps = [
            SimComponent(name="poisson", beta=b0, link=LogLink(), extra={}, sampler=poisson_sampler),
            SimComponent(name="nb2", beta=b1, link=LogLink(), extra={"alpha": 2.0}, sampler=nb2_sampler),
        ]
    elif example_id == 4:
        # FIXED: Zero inflation probability increased, Poisson mean increased
        X = make_design(DesignConfig(n=n, p=p, intercept=True, rho=0.2), rng=rng)
        kind = "counts"
        b0 = np.array([2.0, 0.3, -0.3, 0.3, -0.3]) if p==5 else np.concatenate([[2.0], beta0[1:]*0.5])
        b1 = np.array([1.0, -0.3, 0.3, -0.3, 0.3]) if p==5 else np.concatenate([[1.0], beta1[1:]*0.5])
        families_true = ["poisson", "zip"]
        comps = [
            SimComponent(name="poisson", beta=b0, link=LogLink(), extra={}, sampler=poisson_sampler),
            SimComponent(name="zip", beta=b1, link=LogLink(), extra={"theta": 0.6}, sampler=zip_sampler),
        ]

    pi_true = np.array([0.55, 0.45]) if example_id == 2 else np.array([0.6, 0.4])
    sim = sample_mixture(X=X, components=comps, pi=pi_true, rng=rng)
    return sim, X, kind, comps, pi_true, families_true

def run_scenario_a(example_id, n, seed):
    rng = np.random.default_rng(seed)
    sim_train, X_train, kind, comps, pi_true, families_true = setup_example(example_id, n, 5, sparsity=False, rng=rng)
    sim_test, X_test, _, _, _, _ = setup_example(example_id, 2000, 5, sparsity=False, rng=rng)

    if kind == "real": cands = ["gaussian", "student_t", "skew_normal"]
    elif kind == "positive": cands = ["gamma", "lognormal", "inverse_gaussian"]
    else: cands = ["poisson", "nb2", "zip"]

    t0 = time.time()
    all_candidates = beam_search_models(
        y=sim_train.y, X=X_train,
        candidate_families=cands,
        K_max=3,
        beam_width=SCENARIO_A_BEAM_WIDTH,
        criterion="bic",
        seed=seed,
        init=init_for_kind(kind),
        compute_icl=False,
        standardize=True,
        verbose=False,
        max_iter=100,
        tol=1e-4,
        n_starts=2,
        show_progress=False,
    )
    t1 = time.time()

    all_candidates = [c for c in all_candidates if c.success and c.result is not None and c.ic is not None]
    if not all_candidates:
        return {
            "scenario": "A", "example_id": example_id, "n": n, "seed": seed,
            "error": "No successful beam-search candidates",
            "time": t1 - t0,
        }

    bic_ranked = sorted(all_candidates, key=lambda c: c.ic.bic)
    best_bic = bic_ranked[0]

    strict_correct = (
        best_bic.K == len(families_true)
        and is_strict_structure(best_bic.family_names, families_true)
    )
    boundary_equiv_correct = (
        is_boundary_equivalent(best_bic.family_names, families_true)
        if best_bic.K == len(families_true) else False
    )

    in_sample_ll_selected = best_bic.result.loglik
    test_ll_selected = test_loglik(best_bic.model, sim_test.y, X_test) if best_bic.model is not None else float("-inf")

    # Calculate test LL for all candidate models to find the best predictive model.
    pred_scores = []
    for cand in all_candidates:
        if cand.model is not None and cand.model.result_ is not None and cand.model.result_.converged:
            pred_scores.append((cand, test_loglik(cand.model, sim_test.y, X_test)))

    pred_scores.sort(key=lambda x: x[1], reverse=True)
    best_pred_model = pred_scores[0][0] if pred_scores else best_bic

    strict_bic_ranks = [
        i + 1 for i, c in enumerate(bic_ranked)
        if c.K == len(families_true) and is_strict_structure(c.family_names, families_true)
    ]
    boundary_bic_ranks = [
        i + 1 for i, c in enumerate(bic_ranked)
        if c.K == len(families_true) and is_boundary_equivalent(c.family_names, families_true)
    ]
    strict_pred_ranks = [
        i + 1 for i, (c, _) in enumerate(pred_scores)
        if c.K == len(families_true) and is_strict_structure(c.family_names, families_true)
    ]
    boundary_pred_ranks = [
        i + 1 for i, (c, _) in enumerate(pred_scores)
        if c.K == len(families_true) and is_boundary_equivalent(c.family_names, families_true)
    ]
    strict_bic_rank = int(strict_bic_ranks[0]) if strict_bic_ranks else 0
    boundary_bic_rank = int(boundary_bic_ranks[0]) if boundary_bic_ranks else 0
    strict_pred_rank = int(strict_pred_ranks[0]) if strict_pred_ranks else 0
    boundary_pred_rank = int(boundary_pred_ranks[0]) if boundary_pred_ranks else 0
    strict_bic_delta = (
        float(bic_ranked[strict_bic_rank - 1].ic.bic - best_bic.ic.bic)
        if strict_bic_rank > 0 else np.nan
    )
    boundary_bic_delta = (
        float(bic_ranked[boundary_bic_rank - 1].ic.bic - best_bic.ic.bic)
        if boundary_bic_rank > 0 else np.nan
    )

    # Is the BIC-selected model also the best predictive model?
    bic_is_best_pred = int(
        best_bic.K == best_pred_model.K
        and family_signature(best_bic.family_names) == family_signature(best_pred_model.family_names)
    )

    mc = [ComponentSpec(family=FAMILIES.create(c.name), link=c.link, penalty=PENALTIES.create("none")) for c in comps]
    oracle = MixtureGLM(components=mc)
    oracle.fit(y=sim_train.y, X=X_train, max_iter=100, tol=1e-4, n_starts=2, seed=seed,
               init=init_for_kind(kind), standardize=True, verbose=False)

    in_sample_ll_oracle = oracle.result_.loglik if oracle.result_.converged else float('-inf')
    test_ll_oracle = test_loglik(oracle, sim_test.y, X_test) if oracle.result_.converged else float('-inf')

    # For the first replication, save the Top 10 lists to disk as case studies.
    if seed < 100:
        out_dir = os.path.join(OUTPUT_ROOT, "case_studies")
        os.makedirs(out_dir, exist_ok=True)
        bic_list = [
            {
                "Rank": i + 1,
                "K": c.K,
                "Families": " + ".join(c.family_names),
                "LogLik": c.ic.loglik,
                "AIC": c.ic.aic,
                "BIC": c.ic.bic,
            }
            for i, c in enumerate(bic_ranked[:10])
        ]
        pred_list = [
            {
                "Rank": i + 1,
                "K": c.K,
                "Families": " + ".join(c.family_names),
                "BIC": c.ic.bic,
                "Test_LL": score,
            }
            for i, (c, score) in enumerate(pred_scores[:10])
        ]
        pd.DataFrame(bic_list).to_csv(f"{out_dir}/Ex{example_id}_N{n}_Top10_BIC.csv", index=False)
        pd.DataFrame(pred_list).to_csv(f"{out_dir}/Ex{example_id}_N{n}_Top10_Pred.csv", index=False)

    top_bic = [
        {
            "rank": i + 1,
            "K": c.K,
            "families": " + ".join(c.family_names),
            "loglik": float(c.ic.loglik),
            "aic": float(c.ic.aic),
            "bic": float(c.ic.bic),
            "is_true_structure": bool(
                c.K == len(families_true)
                and is_strict_structure(c.family_names, families_true)
            ),
            "is_boundary_equiv_structure": bool(
                c.K == len(families_true) and is_boundary_equivalent(c.family_names, families_true)
            ),
        }
        for i, c in enumerate(bic_ranked[:10])
    ]
    top_pred = [
        {
            "rank": i + 1,
            "K": c.K,
            "families": " + ".join(c.family_names),
            "bic": float(c.ic.bic),
            "test_ll": float(score),
            "is_true_structure": bool(
                c.K == len(families_true)
                and is_strict_structure(c.family_names, families_true)
            ),
            "is_boundary_equiv_structure": bool(
                c.K == len(families_true) and is_boundary_equivalent(c.family_names, families_true)
            ),
        }
        for i, (c, score) in enumerate(pred_scores[:10])
    ]

    return {
        "scenario": "A", "example_id": example_id, "n": n, "seed": seed,
        "strict_correct": int(strict_correct),
        "boundary_equiv_correct": int(boundary_equiv_correct),
        "strict_bic_rank": strict_bic_rank,
        "boundary_bic_rank": boundary_bic_rank,
        "strict_pred_rank": strict_pred_rank,
        "boundary_pred_rank": boundary_pred_rank,
        "strict_top3_bic": int(0 < strict_bic_rank <= 3),
        "strict_top5_bic": int(0 < strict_bic_rank <= 5),
        "strict_top10_bic": int(0 < strict_bic_rank <= 10),
        "boundary_top3_bic": int(0 < boundary_bic_rank <= 3),
        "boundary_top5_bic": int(0 < boundary_bic_rank <= 5),
        "boundary_top10_bic": int(0 < boundary_bic_rank <= 10),
        "strict_top3_pred": int(0 < strict_pred_rank <= 3),
        "strict_top5_pred": int(0 < strict_pred_rank <= 5),
        "strict_top10_pred": int(0 < strict_pred_rank <= 10),
        "boundary_top3_pred": int(0 < boundary_pred_rank <= 3),
        "boundary_top5_pred": int(0 < boundary_pred_rank <= 5),
        "boundary_top10_pred": int(0 < boundary_pred_rank <= 10),
        "strict_bic_delta": strict_bic_delta,
        "boundary_bic_delta": boundary_bic_delta,
        "in_sample_ll_selected": float(in_sample_ll_selected),
        "test_ll_selected": float(test_ll_selected),
        "in_sample_ll_oracle": float(in_sample_ll_oracle),
        "test_ll_oracle": float(test_ll_oracle),
        "bic_is_best_pred": bic_is_best_pred,
        "time": t1 - t0,
        "top_bic": top_bic,
        "top_pred": top_pred
    }

def run_scenario_b(example_id, seed):
    n = SCENARIO_B_N; p = SCENARIO_B_P
    rng = np.random.default_rng(seed)
    sim, X, kind, comps, pi_true, families_true = setup_example(example_id, n, p, sparsity=True, rng=rng)
    sim_test, X_test, _, _, _, _ = setup_example(example_id, 2000, p, sparsity=True, rng=rng)

    def fit_for_penalty(pen_name, lam=0.0, l1_ratio=0.5):
        mc = []
        for c in comps:
            if pen_name == "none": pen = PENALTIES.create("none")
            elif pen_name == "lasso": pen = PENALTIES.create("lasso", lam=float(lam))
            elif pen_name == "enet": pen = PENALTIES.create("elastic_net", lam=float(lam), l1_ratio=float(l1_ratio))
            else: raise ValueError(f"Unknown penalty {pen_name}")
            mc.append(ComponentSpec(family=FAMILIES.create(c.name), link=c.link, penalty=pen))

        model = MixtureGLM(components=mc)
        t0 = time.time()
        model.fit(y=sim.y, X=X, max_iter=100, tol=1e-4, n_starts=2, seed=seed,
                  init=init_for_kind(kind), standardize=True, verbose=False)
        elapsed = time.time() - t0
        return model, elapsed

    def score_model(model, pen_name, lam=0.0, l1_ratio=None, elapsed=np.nan):
        if model.result_ is None or not model.result_.converged:
            return None
        betas_hat = model.betas_original_scale()

        b0t, b1t = comps[0].beta, comps[1].beta
        b0h, b1h = betas_hat[0], betas_hat[1]
        if np.linalg.norm(b0t - b1h) + np.linalg.norm(b1t - b0h) < np.linalg.norm(b0t - b0h) + np.linalg.norm(b1t - b1h):
            b0h, b1h = b1h, b0h

        thr = 1e-4
        active0 = (b0t != 0)
        active1 = (b1t != 0)
        pred0 = (np.abs(b0h) > thr)
        pred1 = (np.abs(b1h) > thr)

        tp = int(np.sum(active0 & pred0) + np.sum(active1 & pred1))
        fp = int(np.sum(~active0 & pred0) + np.sum(~active1 & pred1))
        fn = int(np.sum(active0 & ~pred0) + np.sum(active1 & ~pred1))
        tn = int(np.sum(~active0 & ~pred0) + np.sum(~active1 & ~pred1))

        tpr0 = np.sum(active0 & pred0) / max(1, np.sum(active0))
        fpr0 = np.sum(~active0 & pred0) / max(1, np.sum(~active0))
        mse0 = np.mean((b0t - b0h)**2)

        tpr1 = np.sum(active1 & pred1) / max(1, np.sum(active1))
        fpr1 = np.sum(~active1 & pred1) / max(1, np.sum(~active1))
        mse1 = np.mean((b1t - b1h)**2)

        selected_size = int(np.sum(pred0) + np.sum(pred1))
        active_size = int(np.sum(active0) + np.sum(active1))
        fdr = fp / max(tp + fp, 1)
        fnr = fn / max(tp + fn, 1)

        return {
            "scenario": "B", "example_id": example_id, "seed": seed,
            "penalty": pen_name,
            "lambda": float(lam),
            "l1_ratio": None if l1_ratio is None else float(l1_ratio),
            "bic": float(model.result_.bic),
            "test_ll": float(test_loglik(model, sim_test.y, X_test)),
            "time": float(elapsed),
            "tpr": float((tpr0+tpr1)/2),
            "fpr": float((fpr0+fpr1)/2),
            "fdr": float(fdr),
            "fnr": float(fnr),
            "mse": float((mse0+mse1)/2),
            "selected_size": selected_size,
            "active_size": active_size,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    candidate_rows = []
    for pen_name, lam, l1_ratio in [("none", 0.0, None)]:
        model, elapsed = fit_for_penalty(pen_name, lam=lam)
        row = score_model(model, pen_name, lam=lam, l1_ratio=l1_ratio, elapsed=elapsed)
        if row is not None:
            candidate_rows.append(row)

    for lam in LAMBDA_GRID:
        model, elapsed = fit_for_penalty("lasso", lam=lam)
        row = score_model(model, "lasso", lam=lam, l1_ratio=None, elapsed=elapsed)
        if row is not None:
            candidate_rows.append(row)

    for lam in LAMBDA_GRID:
        model, elapsed = fit_for_penalty("enet", lam=lam, l1_ratio=0.5)
        row = score_model(model, "enet", lam=lam, l1_ratio=0.5, elapsed=elapsed)
        if row is not None:
            candidate_rows.append(row)

    selected = []
    for pen_name in ["none", "lasso", "enet"]:
        rows = [r for r in candidate_rows if r["penalty"] == pen_name]
        if not rows:
            continue
        best = min(rows, key=lambda r: r["bic"])
        best = dict(best)
        best["selection_rule"] = "bic_within_penalty"
        selected.append(best)

    return {"selected": selected, "path": candidate_rows}

def run_scenario_c(example_id, n, seed):
    if example_id not in SCENARIO_C_EXAMPLES: return []
    rng = np.random.default_rng(seed)
    sim, X, kind, comps, pi_true, families_true = setup_example(example_id, n, 5, sparsity=False, rng=rng)

    mc = [ComponentSpec(family=FAMILIES.create(c.name), link=c.link, penalty=PENALTIES.create("none")) for c in comps]
    model = MixtureGLM(components=mc)
    model.fit(y=sim.y, X=X, max_iter=SCENARIO_C_MAX_ITER, tol=1e-4,
              n_starts=SCENARIO_C_N_STARTS, seed=seed,
              init=init_for_kind(kind), standardize=True, verbose=False)

    if model.result_ is None or not model.result_.converged: return []

    methods = [
        m.strip().lower()
        for m in os.environ.get("MIXGLM_INFERENCE_METHODS", "louis,numeric").split(",")
        if m.strip()
    ]

    scaler = model.scaler_
    def to_fit_scale(b_raw):
        if scaler is None: return b_raw
        return scaler.beta_to_fit_scale(b_raw)

    b0t_fit = to_fit_scale(comps[0].beta)
    b1t_fit = to_fit_scale(comps[1].beta)

    betas_hat = model.betas_original_scale()
    direct_distance = float(
        np.linalg.norm(comps[0].beta - betas_hat[0])
        + np.linalg.norm(comps[1].beta - betas_hat[1])
    )
    crossed_distance = float(
        np.linalg.norm(comps[0].beta - betas_hat[1])
        + np.linalg.norm(comps[1].beta - betas_hat[0])
    )
    # Components with different families are intrinsically labelled by family.
    swapped = bool(families_true[0] == families_true[1] and crossed_distance < direct_distance)

    true_params = {}
    for j in range(5):
        true_params[f"beta[0][{j}]"] = float(b0t_fit[j])
        true_params[f"beta[1][{j}]"] = float(b1t_fit[j])
    for k in range(2):
        for name, val in comps[k].extra.items():
            true_params[f"{name}[{k}]"] = float(val)

    true_params_adj = {}
    if swapped:
        for k_true, k_est in [(0, 1), (1, 0)]:
            for j in range(5): true_params_adj[f"beta[{k_est}][{j}]"] = true_params[f"beta[{k_true}][{j}]"]
            for name in comps[k_true].extra.keys(): true_params_adj[f"{name}[{k_est}]"] = true_params[f"{name}[{k_true}]"]
    else:
        true_params_adj = true_params

    results = []
    for method in methods:
        try:
            df_inf, se_res = model.inference_table(
                y=sim.y,
                X=X,
                method=method,
                louis_derivative_method="auto",
            )
        except Exception as e:
            results.append({
                "scenario": "C",
                "example_id": example_id,
                "n": n,
                    "seed": seed,
                    "method": method,
                    "fit_loglik": float(model.result_.loglik),
                    "fit_n_iter": int(model.result_.n_iter),
                    "fit_min_pi": float(np.min(model.result_.pi)),
                    "alignment_swapped": swapped,
                    "direct_beta_distance": direct_distance,
                    "crossed_beta_distance": crossed_distance,
                    "n_starts": SCENARIO_C_N_STARTS,
                    "error": str(e)[:200],
            })
            continue

        if method == "numeric":
            se_success = bool(getattr(se_res, "success", False))
            se_message = str(getattr(se_res, "message", ""))
            derivative_sources = None
        else:
            se_success = bool(getattr(se_res, "cov", None) is not None and getattr(se_res, "se", None) is not None)
            se_message = "Louis observed information"
            derivative_sources = getattr(se_res, "derivative_sources", None)

        for _, row in df_inf.iterrows():
            param = row['param']
            if param in true_params_adj:
                truth = true_params_adj[param]
                covers = bool(row['ci2.5%'] <= truth <= row['ci97.5%'])
                results.append({
                    "scenario": "C", "example_id": example_id, "n": n, "seed": seed,
                    "method": method,
                    "se_success": se_success,
                    "se_message": se_message,
                    "derivative_sources": derivative_sources,
                    "fit_loglik": float(model.result_.loglik),
                    "fit_n_iter": int(model.result_.n_iter),
                    "fit_min_pi": float(np.min(model.result_.pi)),
                    "alignment_swapped": swapped,
                    "direct_beta_distance": direct_distance,
                    "crossed_beta_distance": crossed_distance,
                    "n_starts": SCENARIO_C_N_STARTS,
                    "param": param, "truth": float(truth), "estimate": float(row['estimate']), "se": float(row['se']),
                    "coverage": float(covers), "ci_len": float(row['ci97.5%'] - row['ci2.5%'])
                })
    return results

def process_task(task_type, args, cache_dir):
    task_id = f"{SIM_VERSION}_{task_type}_{'_'.join(map(str, args))}.json"
    cache_path = os.path.join(cache_dir, task_id)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f: return json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if task_type == "A": res = run_scenario_a(*args)
    elif task_type == "B": res = run_scenario_b(*args)
    elif task_type == "C": res = run_scenario_c(*args)

    temp_cache_path = f"{cache_path}.tmp.{os.getpid()}"
    with open(temp_cache_path, "w") as f: json.dump(res, f)
    os.replace(temp_cache_path, cache_path)
    return res

def generate_latex_and_plots(resA, resB, resC):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if resA:
        dfA = pd.DataFrame([r for r in resA if r and "error" not in r])
        if not dfA.empty:
            aggA = dfA.groupby(['example_id', 'n']).agg(
                Strict_Top1_BIC=('strict_correct', 'mean'),
                Strict_Top1_BIC_MCSE=('strict_correct', mcse_rate),
                Boundary_Top1_BIC=('boundary_equiv_correct', 'mean'),
                Boundary_Top1_BIC_MCSE=('boundary_equiv_correct', mcse_rate),
                Strict_Top3_BIC=('strict_top3_bic', 'mean'),
                Strict_Top5_BIC=('strict_top5_bic', 'mean'),
                Strict_Top10_BIC=('strict_top10_bic', 'mean'),
                Boundary_Top3_BIC=('boundary_top3_bic', 'mean'),
                Boundary_Top5_BIC=('boundary_top5_bic', 'mean'),
                Boundary_Top10_BIC=('boundary_top10_bic', 'mean'),
                Median_Strict_BIC_Rank=('strict_bic_rank', median_positive_rank),
                Mean_Strict_BIC_Delta=('strict_bic_delta', 'mean'),
                BIC_is_Best_Pred_Rate=('bic_is_best_pred', 'mean'),
                In_Sample_LL_Selected=('in_sample_ll_selected', 'mean'),
                In_Sample_LL_Oracle=('in_sample_ll_oracle', 'mean'),
                Test_LL_Selected=('test_ll_selected', 'mean'),
                Test_LL_Oracle=('test_ll_oracle', 'mean')
            ).reset_index()
            aggA.to_csv(os.path.join(OUTPUT_ROOT, "table_scenario_A.csv"), index=False)
            texA = dataframe_to_latex(aggA, float_format="%.2f", caption="Scenario A: Model Selection", label="tab:scen_a")
            with open(os.path.join(OUTPUT_ROOT, "table_scenario_A.tex"), "w") as f: f.write(texA)

    if resB:
        selected_rows = []
        path_rows = []
        for item in resB:
            if isinstance(item, dict):
                selected_rows.extend(item.get("selected", []))
                path_rows.extend(item.get("path", []))
            elif isinstance(item, list):
                selected_rows.extend(item)

        dfB = pd.DataFrame(selected_rows)
        if not dfB.empty:
            aggB = dfB.groupby(['example_id', 'penalty']).agg(
                Lambda=('lambda', 'mean'),
                MSE=('mse', 'mean'),
                TPR=('tpr', 'mean'),
                FPR=('fpr', 'mean'),
                FDR=('fdr', 'mean'),
                FNR=('fnr', 'mean'),
                Selected=('selected_size', 'mean'),
                Test_LL=('test_ll', 'mean')
            ).reset_index()
            aggB.to_csv(os.path.join(OUTPUT_ROOT, "table_scenario_B.csv"), index=False)
            texB = dataframe_to_latex(aggB, float_format="%.3f", caption="Scenario B: Variable Selection", label="tab:scen_b")
            with open(os.path.join(OUTPUT_ROOT, "table_scenario_B.tex"), "w") as f: f.write(texB)
        if path_rows:
            pd.DataFrame(path_rows).to_csv(os.path.join(OUTPUT_ROOT, "scenario_B_lambda_paths.csv"), index=False)

    if resC:
        dfC = pd.DataFrame([
            item for sublist in resC for item in sublist
            if isinstance(item, dict) and "error" not in item and "param" in item
        ])
        if not dfC.empty:
            if "method" not in dfC.columns:
                dfC["method"] = "numeric"
            if "se_success" not in dfC.columns:
                dfC["se_success"] = np.nan
            dfC["abs_bias"] = np.abs(dfC["estimate"] - dfC["truth"])
            dfC["squared_error"] = (dfC["estimate"] - dfC["truth"]) ** 2
            aggC = dfC.groupby(['example_id', 'n', 'method', 'param']).agg(
                Truth=('truth', 'mean'),
                Mean_Est=('estimate', 'mean'),
                Mean_SE=('se', 'mean'),
                SE_Success=('se_success', 'mean'),
                Coverage=('coverage', 'mean'),
                Coverage_MCSE=('coverage', mcse_rate),
                CI_Length=('ci_len', 'mean')
            ).reset_index()
            aggC.to_csv(os.path.join(OUTPUT_ROOT, "table_scenario_C.csv"), index=False)
            texC = dataframe_to_latex(aggC, float_format="%.3f", caption="Scenario C: Inference", label="tab:scen_c")
            with open(os.path.join(OUTPUT_ROOT, "table_scenario_C.tex"), "w") as f: f.write(texC)
            with open(os.path.join(OUTPUT_ROOT, "table_scenario_C_supp.tex"), "w") as f: f.write(texC)

            aggC_main = dfC.groupby(['example_id', 'n', 'method']).agg(
                Mean_Abs_Bias=('abs_bias', 'mean'),
                Parameter_RMSE=('squared_error', lambda x: float(np.sqrt(np.mean(x)))),
                Mean_SE=('se', 'mean'),
                SE_Success=('se_success', 'mean'),
                Coverage=('coverage', 'mean'),
                Coverage_MCSE=('coverage', mcse_rate),
                CI_Length=('ci_len', 'mean')
            ).reset_index()
            aggC_main.to_csv(os.path.join(OUTPUT_ROOT, "table_scenario_C_main.csv"), index=False)
            texC_main = dataframe_to_latex(
                aggC_main,
                float_format="%.3f",
                caption="Scenario C: Inference summary across regression coefficients",
                label="tab:scen_c_main",
            )
            with open(os.path.join(OUTPUT_ROOT, "table_scenario_C_main.tex"), "w") as f:
                f.write(texC_main)

    if resA:
        top_bic_all = []
        top_pred_all = []
        for r in resA:
            if not r: continue
            if "top_bic" in r:
                for tb in r["top_bic"]:
                    top_bic_all.append({
                        "example_id": r["example_id"],
                        "n": r["n"],
                        "rank": tb.get("rank"),
                        "K": tb.get("K"),
                        "families": tb["families"],
                        "bic": tb["bic"],
                        "is_true_structure": tb.get("is_true_structure", False),
                        "is_boundary_equiv_structure": tb.get("is_boundary_equiv_structure", False),
                    })
            if "top_pred" in r:
                for tp in r["top_pred"]:
                    top_pred_all.append({
                        "example_id": r["example_id"],
                        "n": r["n"],
                        "rank": tp.get("rank"),
                        "K": tp.get("K"),
                        "families": tp["families"],
                        "test_ll": tp.get("test_ll", tp.get("test_nll")),
                        "is_true_structure": tp.get("is_true_structure", False),
                        "is_boundary_equiv_structure": tp.get("is_boundary_equiv_structure", False),
                    })

        if top_bic_all:
            os.makedirs(os.path.join(OUTPUT_ROOT, "top10_freq"), exist_ok=True)
            df_tb = pd.DataFrame(top_bic_all)
            df_tb.to_csv(os.path.join(OUTPUT_ROOT, "top10_bic_all_reps.csv"), index=False)
            bic_summaries = []
            for (ex_id, n_v), sub in df_tb.groupby(['example_id', 'n']):
                freq = sub.groupby(['K', 'families']).agg(
                    count=('families', 'size'),
                    avg_rank=('rank', 'mean'),
                    avg_bic=('bic', 'mean'),
                    true_hits=('is_true_structure', 'sum'),
                    boundary_equiv_hits=('is_boundary_equiv_structure', 'sum'),
                ).reset_index()
                freq = freq.sort_values(['count', 'avg_rank'], ascending=[False, True]).head(10)
                freq.insert(0, "n", n_v)
                freq.insert(0, "example_id", ex_id)
                freq.insert(0, "criterion", "BIC")
                bic_summaries.append(freq)
                freq.to_csv(os.path.join(OUTPUT_ROOT, "top10_freq", f"BIC_freq_Ex{ex_id}_N{n_v}.csv"), index=False)
            if bic_summaries:
                pd.concat(bic_summaries, ignore_index=True).to_csv(
                    os.path.join(OUTPUT_ROOT, "top10_bic_frequency_summary.csv"),
                    index=False,
                )

        if top_pred_all:
            os.makedirs(os.path.join(OUTPUT_ROOT, "top10_freq"), exist_ok=True)
            df_tp = pd.DataFrame(top_pred_all)
            df_tp.to_csv(os.path.join(OUTPUT_ROOT, "top10_pred_all_reps.csv"), index=False)
            pred_summaries = []
            for (ex_id, n_v), sub in df_tp.groupby(['example_id', 'n']):
                freq = sub.groupby(['K', 'families']).agg(
                    count=('families', 'size'),
                    avg_rank=('rank', 'mean'),
                    avg_test_ll=('test_ll', 'mean'),
                    true_hits=('is_true_structure', 'sum'),
                    boundary_equiv_hits=('is_boundary_equiv_structure', 'sum'),
                ).reset_index()
                freq = freq.sort_values(['count', 'avg_rank'], ascending=[False, True]).head(10)
                freq.insert(0, "n", n_v)
                freq.insert(0, "example_id", ex_id)
                freq.insert(0, "criterion", "Pred")
                pred_summaries.append(freq)
                freq.to_csv(os.path.join(OUTPUT_ROOT, "top10_freq", f"Pred_freq_Ex{ex_id}_N{n_v}.csv"), index=False)
            if pred_summaries:
                pd.concat(pred_summaries, ignore_index=True).to_csv(
                    os.path.join(OUTPUT_ROOT, "top10_pred_frequency_summary.csv"),
                    index=False,
                )

def main():
    register_families()
    register_penalties()

    n_reps = int(os.environ.get("MIXGLM_N_REPS", 100))
    n_reps_a = int(os.environ.get("MIXGLM_N_REPS_A", n_reps))
    n_reps_b = int(os.environ.get("MIXGLM_N_REPS_B", n_reps))
    n_reps_c = int(os.environ.get("MIXGLM_N_REPS_C", n_reps))
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("MIXGLM_N_JOBS", 1)))
    cache_dir = os.path.join(OUTPUT_ROOT, "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    metadata = {
        "sim_version": SIM_VERSION,
        "output_root": OUTPUT_ROOT,
        "n_reps": n_reps,
        "n_reps_a": n_reps_a,
        "n_reps_b": n_reps_b,
        "n_reps_c": n_reps_c,
        "sample_sizes": SAMPLE_SIZES,
        "scenario_A_examples": SCENARIO_A_EXAMPLES,
        "scenario_B_examples": SCENARIO_B_EXAMPLES,
        "scenario_C_examples": SCENARIO_C_EXAMPLES,
        "scenario_B_n": SCENARIO_B_N,
        "scenario_B_p": SCENARIO_B_P,
        "scenario_A_beam_width": SCENARIO_A_BEAM_WIDTH,
        "scenario_C_n_starts": SCENARIO_C_N_STARTS,
        "scenario_C_max_iter": SCENARIO_C_MAX_ITER,
        "lambda_grid": LAMBDA_GRID,
        "inference_methods": os.environ.get("MIXGLM_INFERENCE_METHODS", "louis,numeric"),
        "created_unix_time": time.time(),
    }
    with open(os.path.join(OUTPUT_ROOT, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Starting Master Simulation {SIM_VERSION} with {n_jobs} cores (using checkpointing)...")
    print(f"Output root: {OUTPUT_ROOT}")

    tasksA, tasksB, tasksC = [], [], []
    for rep in range(max(n_reps_a, n_reps_b, n_reps_c)):
        if rep < n_reps_a:
            for ex in SCENARIO_A_EXAMPLES:
                seed = rep * 100 + ex
                for n in SAMPLE_SIZES:
                    tasksA.append(("A", (ex, n, seed)))
        if rep < n_reps_c:
            for ex in SCENARIO_C_EXAMPLES:
                seed = rep * 100 + ex
                for n in SAMPLE_SIZES:
                    tasksC.append(("C", (ex, n, seed)))
        if rep < n_reps_b:
            for ex in SCENARIO_B_EXAMPLES:
                seed = rep * 100 + ex
                tasksB.append(("B", (ex, seed)))

    print(f"Evaluating {len(tasksA) + len(tasksB) + len(tasksC)} total simulation tasks...")
    all_tasks = tasksA + tasksB + tasksC

    results = Parallel(n_jobs=n_jobs)(delayed(process_task)(t[0], t[1], cache_dir) for t in all_tasks)

    resA = [r for t, r in zip(all_tasks, results) if t[0] == "A"]
    resB = [r for t, r in zip(all_tasks, results) if t[0] == "B"]
    resC = [r for t, r in zip(all_tasks, results) if t[0] == "C"]

    raw_path = os.path.join(OUTPUT_ROOT, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump({"A": resA, "B": resB, "C": resC}, f)

    print("Generating LaTeX tables and plots...")
    generate_latex_and_plots(resA, resB, resC)
    print(f"Done! Outputs saved in '{OUTPUT_ROOT}' directory.")

if __name__ == "__main__":
    sys.path.insert(0, './src')
    main()
