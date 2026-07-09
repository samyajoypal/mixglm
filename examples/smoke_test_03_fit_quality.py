# examples/smoke_test_03_fit_quality.py

from __future__ import annotations

import numpy as np
import pandas as pd

from mixglm.utils.repro import set_global_seed
from mixglm.model import ComponentSpec, MixtureGLM

from mixglm.families.gaussian import GaussianFamily
from mixglm.links.identity import IdentityLink

from mixglm.penalties.base import NoPenalty
from mixglm.penalties.lasso import LassoPenalty


Array = np.ndarray


def simulate_two_component_gaussian_sparse(
    *,
    n: int,
    p: int,
    pi_true: Array,
    beta0_true: Array,
    beta1_true: Array,
    sigma0: float,
    sigma1: float,
    rng: np.random.Generator,
) -> tuple[Array, Array, Array, Array]:
    """
    Two-component Gaussian mixture with component-specific regression.
    Returns: (y, X, z, true_mixture_mean)
    """
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0  # intercept

    z = rng.choice(2, size=n, p=pi_true)
    mu0 = X @ beta0_true
    mu1 = X @ beta1_true

    y = np.empty(n, dtype=float)
    m0 = (z == 0)
    y[m0] = mu0[m0] + rng.normal(0.0, sigma0, size=int(m0.sum()))
    y[~m0] = mu1[~m0] + rng.normal(0.0, sigma1, size=int((~m0).sum()))

    true_mix_mean = pi_true[0] * mu0 + pi_true[1] * mu1
    return y, X, z, true_mix_mean


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
    Match component labels (0/1) by minimizing total L2 distance between beta vectors.
    Returns (perm, cost), where perm maps true index -> estimated index.
    For K=2 we can brute force.
    """
    # perm1: (0->0, 1->1)
    c1 = np.linalg.norm(true_betas[0] - est_betas[0]) + np.linalg.norm(true_betas[1] - est_betas[1])
    # perm2: (0->1, 1->0)
    c2 = np.linalg.norm(true_betas[0] - est_betas[1]) + np.linalg.norm(true_betas[1] - est_betas[0])
    if c1 <= c2:
        return [0, 1], float(c1)
    return [1, 0], float(c2)


def fit_model(
    *,
    y: Array,
    X: Array,
    penalty,
    seed: int,
    standardize: bool = True,
    verbose: bool = False,
) -> MixtureGLM:
    comps = [
        ComponentSpec(family=GaussianFamily(), link=IdentityLink(), penalty=penalty),
        ComponentSpec(family=GaussianFamily(), link=IdentityLink(), penalty=penalty),
    ]
    model = MixtureGLM(comps)
    model.fit(
        y=y,
        X=X,
        max_iter=200,
        tol=1e-6,
        n_starts=5,
        seed=seed,
        init="kmeans_y",
        verbose=verbose,
        inner_mstep_iter=3,
        min_pi=1e-6,
        compute_icl=True,
        standardize=standardize,
    )
    return model


def coef_table(
    *,
    name: str,
    beta_true0: Array,
    beta_true1: Array,
    beta_hat0: Array,
    beta_hat1: Array,
    feature_names: list[str],
) -> pd.DataFrame:
    rows = []
    for j, fn in enumerate(feature_names):
        rows.append(
            dict(
                model=name,
                component=0,
                feature=fn,
                beta_true=float(beta_true0[j]),
                beta_hat=float(beta_hat0[j]),
                abs_err=float(abs(beta_true0[j] - beta_hat0[j])),
            )
        )
        rows.append(
            dict(
                model=name,
                component=1,
                feature=fn,
                beta_true=float(beta_true1[j]),
                beta_hat=float(beta_hat1[j]),
                abs_err=float(abs(beta_true1[j] - beta_hat1[j])),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    rng = set_global_seed(123)

    # ----- simulate -----
    n = 1200
    p = 12  # includes intercept
    pi_true = np.array([0.60, 0.40], dtype=float)

    beta0_true = np.zeros(p)
    beta1_true = np.zeros(p)
    beta0_true[0] = 0.6
    beta1_true[0] = -0.2

    # sparse signals
    beta0_true[1] = 1.2
    beta0_true[3] = -0.9
    beta0_true[7] = 0.6

    beta1_true[1] = 0.5
    beta1_true[3] = -1.1
    beta1_true[9] = 0.8

    sigma0 = 1.0
    sigma1 = 1.1

    y, X, z, true_mix_mean = simulate_two_component_gaussian_sparse(
        n=n, p=p, pi_true=pi_true,
        beta0_true=beta0_true, beta1_true=beta1_true,
        sigma0=sigma0, sigma1=sigma1,
        rng=rng,
    )

    feature_names = ["intercept"] + [f"x{j}" for j in range(1, p)]

    # ----- fit two models -----
    fits = [
        ("none", NoPenalty()),
        ("lasso(lam=1.0)", LassoPenalty(lam=1.0)),
    ]

    pred_rows = []
    coef_dfs = []

    for model_name, pen in fits:
        model = fit_model(y=y, X=X, penalty=pen, seed=123, standardize=True, verbose=False)
        res = model.result_

        # predictions: mixture mean using fitted pi and mu_k
        yhat = model.predict_mean(X)

        # coefficients on original X scale (important because we standardized in fit)
        betas_hat = model.betas_original_scale()

        # handle label switching using beta matching in original scale
        perm, cost = match_labels_by_beta(
            true_betas=[beta0_true, beta1_true],
            est_betas=[betas_hat[0], betas_hat[1]],
        )
        # perm maps true index -> estimated index
        beta_hat0 = betas_hat[perm[0]]
        beta_hat1 = betas_hat[perm[1]]

        # align pi accordingly: estimated index -> true index inverse
        # if perm = [1,0], then true0 uses est1 so pi_true0_hat = pi_est[1]
        pi_hat_aligned = np.array([res.pi[perm[0]], res.pi[perm[1]]], dtype=float)

        pred_rows.append(
            dict(
                model=model_name,
                converged=bool(res.converged),
                loglik=float(res.loglik),
                bic=float(res.bic),
                icl=float(res.icl) if res.icl is not None else np.nan,
                pi_hat0=float(pi_hat_aligned[0]),
                pi_hat1=float(pi_hat_aligned[1]),
                mse_Ey=float(mse(true_mix_mean, yhat)),
                mae_Ey=float(mae(true_mix_mean, yhat)),
                r2_Ey=float(r2(true_mix_mean, yhat)),
                label_match_cost=float(cost),
            )
        )

        coef_dfs.append(
            coef_table(
                name=model_name,
                beta_true0=beta0_true,
                beta_true1=beta1_true,
                beta_hat0=beta_hat0,
                beta_hat1=beta_hat1,
                feature_names=feature_names,
            )
        )

    pred_df = pd.DataFrame(pred_rows).sort_values("bic").reset_index(drop=True)
    coef_df = pd.concat(coef_dfs, axis=0, ignore_index=True)

    # show a compact coefficient view: only the truly nonzero features + largest estimated ones
    nonzero_idx = sorted(set(np.where(beta0_true != 0)[0].tolist() + np.where(beta1_true != 0)[0].tolist()))
    keep_features = {feature_names[j] for j in nonzero_idx}

    # also keep top estimated (by abs beta_hat) per model/component (excluding intercept)
    for model_name in coef_df["model"].unique():
        for k in [0, 1]:
            sub = coef_df[(coef_df["model"] == model_name) & (coef_df["component"] == k) & (coef_df["feature"] != "intercept")]
            top = sub.reindex(sub["beta_hat"].abs().sort_values(ascending=False).index).head(3)
            keep_features |= set(top["feature"].tolist())

    coef_compact = coef_df[coef_df["feature"].isin(sorted(keep_features))].copy()
    coef_compact["feature"] = pd.Categorical(coef_compact["feature"], categories=feature_names, ordered=True)
    coef_compact = coef_compact.sort_values(["model", "component", "feature"]).reset_index(drop=True)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 50)

    print("\n=== Smoke test 03: Fit quality (true vs estimated) ===")
    print("\n--- Prediction quality table (true E[Y|X] vs estimated mixture mean) ---")
    print(pred_df.to_string(index=False))

    print("\n--- Coefficient comparison table (compact) ---")
    print(coef_compact.to_string(index=False))

    print("\nNote: coefficients shown are on the ORIGINAL X scale (model fit used standardization internally).")


if __name__ == "__main__":
    main()
