# examples/smoke_test_04_model_selection.py

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from mixglm.utils.repro import set_global_seed
from mixglm.model import MixtureGLM, ComponentSpec

from mixglm.links.identity import IdentityLink
from mixglm.penalties.base import NoPenalty

# Candidate families (all support R; safe for Gaussian+t data)
from mixglm.families.gaussian import GaussianFamily
from mixglm.families.student_t import StudentTFamily
from mixglm.families.skew_normal import SkewNormalFamily
from mixglm.families.genhyperbolic import GeneralizedHyperbolicFamily
from mixglm.families.skew_t import SkewTFamily
from mixglm.families.azzalini_skew_t import AzzaliniSkewTFamily


Array = np.ndarray


# ------------------------ data generation (fallback) ------------------------

def simulate_gaussian_studentt_nonidentical(
    *,
    n: int,
    p: int,
    pi_true: Array,
    beta0_true: Array,
    beta1_true: Array,
    sigma_true: float,
    df_true: float,
    rng: np.random.Generator,
) -> tuple[Array, Array, Array, Array]:
    """
    True model:
      Z ~ Categorical(pi_true)
      If Z=0: Y = X beta0 + eps, eps ~ N(0, sigma^2)
      If Z=1: Y = X beta1 + eps, eps ~ t_df(0, sigma)   (location 0, scale sigma)

    Returns:
      y, X, z, true_mix_mean = pi0 * mu0 + pi1 * mu1
    """
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0  # intercept

    z = rng.choice(2, size=n, p=pi_true)
    mu0 = X @ beta0_true
    mu1 = X @ beta1_true

    y = np.empty(n, dtype=float)

    m0 = (z == 0)
    y[m0] = mu0[m0] + rng.normal(0.0, sigma_true, size=int(m0.sum()))

    # Student-t errors with scale sigma_true
    # If numpy has standard_t: eps = sigma * standard_t(df)
    eps1 = sigma_true * rng.standard_t(df_true, size=int((~m0).sum()))
    y[~m0] = mu1[~m0] + eps1

    true_mix_mean = pi_true[0] * mu0 + pi_true[1] * mu1
    return y, X, z, true_mix_mean


# ------------------------ helpers ------------------------

def mse(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


def mae(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def r2(y_true: Array, y_pred: Array) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ssr / sst) if sst > 0 else np.nan


def match_labels_by_beta(true_betas: list[Array], est_betas: list[Array]) -> tuple[list[int], float]:
    """
    For K=2, choose permutation minimizing total L2 distance in beta.
    Returns perm mapping true index -> estimated index.
    """
    c1 = np.linalg.norm(true_betas[0] - est_betas[0]) + np.linalg.norm(true_betas[1] - est_betas[1])
    c2 = np.linalg.norm(true_betas[0] - est_betas[1]) + np.linalg.norm(true_betas[1] - est_betas[0])
    if c1 <= c2:
        return [0, 1], float(c1)
    return [1, 0], float(c2)


def make_family(name: str):
    key = name.lower()
    if key == "gaussian":
        return GaussianFamily()
    if key == "student_t":
        return StudentTFamily()
    if key == "skew_normal":
        return SkewNormalFamily()
    if key == "genhyperbolic":
        return GeneralizedHyperbolicFamily()
    if key == "skew_t":
        return SkewTFamily()
    if key == "azzalini_skew_t":
        return AzzaliniSkewTFamily()
    raise KeyError(f"Unknown family name: {name}")


def fit_candidate(
    *,
    y: Array,
    X: Array,
    fam_names: tuple[str, str],
    seed: int,
    verbose: bool = False,
) -> MixtureGLM:
    link = IdentityLink()
    penalty = NoPenalty()

    comps = [
        ComponentSpec(family=make_family(fam_names[0]), link=link, penalty=penalty),
        ComponentSpec(family=make_family(fam_names[1]), link=link, penalty=penalty),
    ]
    m = MixtureGLM(comps)
    m.fit(
        y=y,
        X=X,
        max_iter=250,
        tol=1e-6,
        n_starts=5,
        seed=seed,
        init="kmeans_y",
        verbose=verbose,
        inner_mstep_iter=3,
        min_pi=1e-6,
        compute_icl=True,
        standardize=True,
    )
    return m


# ------------------------ main smoke test ------------------------

def main() -> None:
    rng = set_global_seed(123)

    # ============== 1) Generate data ==============
    # If you already have a sims generator, plug it in here and remove fallback.
    # Example:
    # from mixglm.sims.nonidentical import simulate_gaussian_studentt_nonidentical as sim
    # y, X, z_true, true_mix_mean = sim(...)
    n = 1200
    p = 6
    pi_true = np.array([0.60, 0.40], dtype=float)

    beta0_true = np.zeros(p); beta0_true[0] = 0.5
    beta1_true = np.zeros(p); beta1_true[0] = -0.2
    beta0_true[1] = 1.0; beta0_true[2] = -0.7
    beta1_true[1] = -0.6; beta1_true[3] = 1.2

    sigma_true = 0.9
    df_true = 6.0

    y, X, z_true, true_mix_mean = simulate_gaussian_studentt_nonidentical(
        n=n, p=p, pi_true=pi_true,
        beta0_true=beta0_true, beta1_true=beta1_true,
        sigma_true=sigma_true, df_true=df_true,
        rng=rng,
    )

    print("\n=== Smoke test 04: Non-identical mixture + model selection (no penalty) ===")
    print(f"True model: gaussian + student_t (identity link), pi={pi_true}, sigma={sigma_true}, df={df_true}")
    print(f"n={n}, p={p}")
    print()

    # ============== 2) Candidate model space ==============
    # Keep it small for smoke testing (you can expand later).
    candidates = [
        "gaussian",
        "student_t",
        "skew_normal",
        "genhyperbolic",
        "skew_t",
        "azzalini_skew_t",
    ]

    # All ordered pairs (includes identical + non-identical)
    cand_pairs = list(itertools.product(candidates, repeat=2))

    # ============== 3) Fit all candidates and rank ==============
    rows = []
    fitted_models: dict[tuple[str, str], MixtureGLM] = {}

    for fam_pair in cand_pairs:
        try:
            model = fit_candidate(y=y, X=X, fam_names=fam_pair, seed=123, verbose=False)
            res = model.result_
            fitted_models[fam_pair] = model

            rows.append(
                dict(
                    family_pair=f"{fam_pair[0]} + {fam_pair[1]}",
                    converged=bool(res.converged),
                    loglik=float(res.loglik),
                    aic=float(res.aic),
                    bic=float(res.bic),
                    icl=float(res.icl) if res.icl is not None else np.nan,
                    pi0=float(res.pi[0]),
                    pi1=float(res.pi[1]),
                )
            )
        except Exception as e:
            # if a family fails for some reason, skip it (but record)
            rows.append(
                dict(
                    family_pair=f"{fam_pair[0]} + {fam_pair[1]}",
                    converged=False,
                    loglik=np.nan,
                    aic=np.nan,
                    bic=np.nan,
                    icl=np.nan,
                    pi0=np.nan,
                    pi1=np.nan,
                    error=str(e)[:120],
                )
            )

    df = pd.DataFrame(rows)
    df_ok = df[df["converged"] & df["bic"].notna()].copy()
    df_ok = df_ok.sort_values("bic").reset_index(drop=True)

    top5 = df_ok.head(5).copy()

    print("--- Top 5 models by BIC ---")
    print(top5[["family_pair", "bic", "icl", "loglik", "pi0", "pi1", "converged"]].to_string(index=False))

    # ============== 4) Evaluate best model vs truth ==============
    best_pair = tuple(top5.loc[0, "family_pair"].split(" + "))
    best_pair = (best_pair[0].strip(), best_pair[1].strip())
    best_model = fitted_models[best_pair]
    best_res = best_model.result_

    # Estimated mixture mean predictor
    yhat_mix_mean = best_model.predict_mean(X)

    # Betas on original X scale (because model standardized internally)
    betas_hat = best_model.betas_original_scale()

    # Label match by beta (K=2)
    perm, cost = match_labels_by_beta(
        true_betas=[beta0_true, beta1_true],
        est_betas=[betas_hat[0], betas_hat[1]],
    )
    beta_hat0 = betas_hat[perm[0]]
    beta_hat1 = betas_hat[perm[1]]
    pi_hat_aligned = np.array([best_res.pi[perm[0]], best_res.pi[perm[1]]], dtype=float)

    # Metrics: mixture mean
    pred_table = pd.DataFrame(
        [
            dict(
                selected_model=f"{best_pair[0]} + {best_pair[1]}",
                true_model="gaussian + student_t",
                bic=float(best_res.bic),
                icl=float(best_res.icl) if best_res.icl is not None else np.nan,
                pi_true0=float(pi_true[0]),
                pi_hat0=float(pi_hat_aligned[0]),
                pi_true1=float(pi_true[1]),
                pi_hat1=float(pi_hat_aligned[1]),
                mse_Ey=float(mse(true_mix_mean, yhat_mix_mean)),
                mae_Ey=float(mae(true_mix_mean, yhat_mix_mean)),
                r2_Ey=float(r2(true_mix_mean, yhat_mix_mean)),
                label_match_cost=float(cost),
            )
        ]
    )

    # Coef table
    feat_names = ["intercept"] + [f"x{j}" for j in range(1, p)]
    coef_rows = []
    for j, fn in enumerate(feat_names):
        coef_rows.append(dict(component="true0/hat0", feature=fn, beta_true=float(beta0_true[j]), beta_hat=float(beta_hat0[j]), abs_err=float(abs(beta0_true[j]-beta_hat0[j]))))
        coef_rows.append(dict(component="true1/hat1", feature=fn, beta_true=float(beta1_true[j]), beta_hat=float(beta_hat1[j]), abs_err=float(abs(beta1_true[j]-beta_hat1[j]))))
    coef_df = pd.DataFrame(coef_rows)

    # Extras table (aligned)
    # extras are dicts, so just show them as strings for smoke test
    extra0_hat = best_res.extras[perm[0]]
    extra1_hat = best_res.extras[perm[1]]
    extras_table = pd.DataFrame(
        [
            dict(component="0 (true gaussian)", true_extra=f"sigma={sigma_true}", hat_extra=str(extra0_hat)),
            dict(component="1 (true student_t)", true_extra=f"sigma={sigma_true}, df={df_true}", hat_extra=str(extra1_hat)),
        ]
    )

    print("\n--- Best model: truth vs estimated (mixture mean + pi) ---")
    print(pred_table.to_string(index=False))

    print("\n--- True vs estimated coefficients (aligned by beta) ---")
    print(coef_df.to_string(index=False))

    print("\n--- True vs estimated nuisance/extras (aligned) ---")
    print(extras_table.to_string(index=False))


if __name__ == "__main__":
    main()
