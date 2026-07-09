# # examples/smoke_test_01.py
# from __future__ import annotations

# import numpy as np
# import pandas as pd

# from mixglm.inference import numeric_se_from_model

# from mixglm.utils.repro import set_global_seed
# from mixglm.model import ComponentSpec, MixtureGLM

# from mixglm.families.gaussian import GaussianFamily
# from mixglm.families.student_t import StudentTFamily

# from mixglm.links.identity import IdentityLink
# from mixglm.penalties.base import NoPenalty


# def simulate_two_component_gauss_t(
    # *,
    # n: int,
    # p: int,
    # pi: np.ndarray,
    # beta0: np.ndarray,
    # beta1: np.ndarray,
    # sigma: float,
    # df: float,
    # rng: np.random.Generator,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # """
    # Simulate y|x from a 2-component mixture:
      # Z ~ Categorical(pi)
      # if Z=0: y = x'beta0 + eps, eps ~ N(0, sigma^2)
      # if Z=1: y = x'beta1 + eps, eps ~ t_df scaled to have stdev approx sigma

    # Returns (y, X, z_true).
    # """
    # X = rng.normal(size=(n, p))
    # X[:, 0] = 1.0  # intercept

    # z = rng.choice(2, size=n, p=pi)

    # mu0 = X @ beta0
    # mu1 = X @ beta1

    # y = np.empty(n, dtype=float)

    # # component 0: Gaussian
    # mask0 = z == 0
    # y[mask0] = mu0[mask0] + rng.normal(loc=0.0, scale=sigma, size=int(mask0.sum()))

    # # component 1: Student-t
    # mask1 = ~mask0
    # # standard t has variance df/(df-2) for df>2, so scale to match sigma
    # t_scale = sigma / np.sqrt(df / (df - 2.0)) if df > 2 else sigma
    # y[mask1] = mu1[mask1] + rng.standard_t(df=df, size=int(mask1.sum())) * t_scale

    # return y, X, z


# def main() -> None:
    # rng = set_global_seed(123)

    # # --- simulate ---
    # n = 800
    # p = 3  # includes intercept at column 0
    # pi_true = np.array([0.6, 0.4])

    # beta0_true = np.array([0.5, 1.0, -0.5])
    # beta1_true = np.array([-0.5, 2.0, 0.25])

    # sigma_true = 0.8
    # df_true = 6.0

    # y, X, z_true = simulate_two_component_gauss_t(
        # n=n,
        # p=p,
        # pi=pi_true,
        # beta0=beta0_true,
        # beta1=beta1_true,
        # sigma=sigma_true,
        # df=df_true,
        # rng=rng,
    # )

    # print("=== Smoke test 01: Gaussian + Student-t mixture GLM ===")
    # print(f"n={n}, p={p}, pi_true={pi_true}")
    # print(f"beta0_true={beta0_true}")
    # print(f"beta1_true={beta1_true}")
    # print(f"sigma_true={sigma_true}, df_true={df_true}")
    # print()

    # # --- build model (2 components) ---
    # # Gaussian: extra is log_sigma in our implementation (verify your student_t/gaussian files)
    # comp0 = ComponentSpec(
        # family=GaussianFamily(),
        # link=IdentityLink(),
        # penalty=NoPenalty(),
    # )
    # comp1 = ComponentSpec(
        # family=StudentTFamily(),
        # link=IdentityLink(),
        # penalty=NoPenalty(),
    # )

    # model = MixtureGLM([comp0, comp1])

    # # --- fit ---
    # model.fit(
        # y=y,
        # X=X,
        # max_iter=150,
        # tol=1e-6,
        # n_starts=5,
        # seed=123,
        # init="kmeans_y",
        # verbose=True,
        # inner_mstep_iter=2,
        # min_pi=1e-6,
        # compute_icl=True,
    # )

    # res = model.result_
    # assert res is not None

    # print("\n=== Fit summary ===")
    # print(model.summary())
    # print()
    # print("\n=== Numeric Hessian SEs (baseline) ===")
    # num = numeric_se_from_model(
        # model=model,
        # y=y,
        # X=X,
        # use_model_scaler=True,     # IMPORTANT
        # eps=1e-5,                  # try 1e-4 if unstable
        # use_pinv=True,
        # rcond=1e-10,
    # )
    # print(num.message)

    # # Build readable names matching your packing order
    # K = len(model.components)
    # p = X.shape[1]

    # names = []
    # # eta_pi (K-1)
    # for j in range(K - 1):
        # names.append(f"eta_pi[{j}]")

    # # betas
    # for k in range(K):
        # for j in range(p):
            # names.append(f"beta[{k}][{j}]")

    # # extras (transformed names)
    # for k, comp in enumerate(model.components):
        # for en in comp.family.extra_param_names:
            # names.append(f"extra[{k}].{en}_t")

    # df_se = pd.DataFrame({
        # "param": names,
        # "se_numeric": num.se,
    # })
    # print(df_se.head(15).to_string(index=False))

    # # quick sanity: show max SE and how many huge
    # huge = np.sum(df_se["se_numeric"] > 1e3)
    # print(f"\nSE diagnostics: max={df_se['se_numeric'].max():.3e}, huge(>1e3)={huge}/{len(df_se)}")

    # # --- quick sanity checks ---
    # tau = res.responsibilities
    # row_sums = np.max(np.abs(tau.sum(axis=1) - 1.0))
    # print(f"max |sum_k tau_ik - 1| = {row_sums:.3e}")

    # # approximate classification accuracy (label switching possible)
    # z_hat = np.argmax(tau, axis=1)
    # acc1 = np.mean(z_hat == z_true)
    # acc2 = np.mean((1 - z_hat) == z_true)
    # acc = max(acc1, acc2)
    # print(f"hard assignment accuracy (up to label switch) = {acc:.3f}")

    # # mixture mean prediction sanity
    # yhat = model.predict_mean(X)
    # mse = float(np.mean((y - yhat) ** 2))
    # print(f"mse of mixture mean predictor = {mse:.4f}")

    # print("\nDone.")


# if __name__ == "__main__":
    # main()


# examples/smoke_test_01.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special

from mixglm.utils.repro import set_global_seed
from mixglm.model import ComponentSpec, MixtureGLM

from mixglm.families.gaussian import GaussianFamily
from mixglm.families.student_t import StudentTFamily

from mixglm.links.identity import IdentityLink
from mixglm.penalties.base import NoPenalty

# Your existing wrapper (kept)
from mixglm.inference import numeric_se_from_model


def simulate_two_component_gauss_t(
    *,
    n: int,
    p: int,
    pi: np.ndarray,
    beta0: np.ndarray,
    beta1: np.ndarray,
    sigma: float,
    df: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate y|x from a 2-component mixture:
      Z ~ Categorical(pi)
      if Z=0: y = x'beta0 + eps, eps ~ N(0, sigma^2)
      if Z=1: y = x'beta1 + eps, eps ~ t_df scaled to have stdev approx sigma

    Returns (y, X, z_true).
    """
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0  # intercept

    z = rng.choice(2, size=n, p=pi)

    mu0 = X @ beta0
    mu1 = X @ beta1

    y = np.empty(n, dtype=float)

    # component 0: Gaussian
    mask0 = z == 0
    y[mask0] = mu0[mask0] + rng.normal(loc=0.0, scale=sigma, size=int(mask0.sum()))

    # component 1: Student-t
    mask1 = ~mask0
    # standard t has variance df/(df-2) for df>2, so scale to match sigma
    t_scale = sigma / np.sqrt(df / (df - 2.0)) if df > 2 else sigma
    y[mask1] = mu1[mask1] + rng.standard_t(df=df, size=int(mask1.sum())) * t_scale

    return y, X, z


def _r2(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ssr = float(np.sum((y_true - y_hat) ** 2))
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return float(1.0 - ssr / max(sst, 1e-12))


def _wald_table(theta: np.ndarray, se: np.ndarray, names: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """
    Wald z/p/CI using normal approximation, no scipy dependency.
    """
    theta = np.asarray(theta, dtype=float)
    se = np.asarray(se, dtype=float)
    z = theta / np.clip(se, 1e-15, np.inf)

    # p = 2*(1 - Phi(|z|)) ; Phi(x)=0.5*(1+erf(x/sqrt(2)))
    # Phi = 0.5 * (1.0 + np.erf(np.abs(z) / np.sqrt(2.0)))
    Phi = 0.5 * (1.0 + special.erf(np.abs(z) / np.sqrt(2.0)))
    pval = 2.0 * (1.0 - Phi)

    # z_{1-alpha/2} via binary search
    target = 1.0 - alpha / 2.0
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        # Phi_mid = 0.5 * (1.0 + np.erf(mid / np.sqrt(2.0)))
        Phi_mid = 0.5 * (1.0 + special.erf(mid / np.sqrt(2.0)))
        if Phi_mid < target:
            lo = mid
        else:
            hi = mid
    zc = 0.5 * (lo + hi)

    ci_lo = theta - zc * se
    ci_hi = theta + zc * se

    return pd.DataFrame(
        {
            "param": names,
            "estimate": theta,
            "se": se,
            "z": z,
            "p": pval,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }
    )


def main() -> None:
    rng = set_global_seed(123)

    # --- simulate ---
    n = 800
    p = 3  # includes intercept at column 0
    pi_true = np.array([0.6, 0.4])

    beta0_true = np.array([0.5, 1.0, -0.5])
    beta1_true = np.array([-0.5, 2.0, 0.25])

    sigma_true = 0.8
    df_true = 6.0

    y, X, z_true = simulate_two_component_gauss_t(
        n=n,
        p=p,
        pi=pi_true,
        beta0=beta0_true,
        beta1=beta1_true,
        sigma=sigma_true,
        df=df_true,
        rng=rng,
    )

    print("=== Smoke test 01: Gaussian + Student-t mixture GLM ===")
    print(f"n={n}, p={p}, pi_true={pi_true}")
    print(f"beta0_true={beta0_true}")
    print(f"beta1_true={beta1_true}")
    print(f"sigma_true={sigma_true}, df_true={df_true}")
    print()

    # --- build model (2 components) ---
    comp0 = ComponentSpec(
        family=GaussianFamily(),
        link=IdentityLink(),
        penalty=NoPenalty(),
    )
    comp1 = ComponentSpec(
        family=StudentTFamily(),
        link=IdentityLink(),
        penalty=NoPenalty(),
    )

    model = MixtureGLM([comp0, comp1])

    # --- fit ---
    model.fit(
        y=y,
        X=X,
        max_iter=150,
        tol=1e-6,
        n_starts=5,
        seed=123,
        init="kmeans_y",
        verbose=True,
        inner_mstep_iter=2,
        min_pi=1e-6,
        compute_icl=True,
        standardize=True,
    )

    res = model.result_
    assert res is not None

    # ---------------- basic summary ----------------
    print("\n=== Fit summary ===")
    print(model.summary())
    print()

    # ---------------- extended summary (new) ----------------
    # This prints mse/r2/adj_r2_like and a Wald table using numeric SEs internally.
    print("\n=== Extended summary (glm-like; numeric SE baseline) ===")
    try:
        print(
            model.summary_extended(
                y=y,
                X=X,
                method="numeric",
                alpha=0.05,
                numeric_eps=1e-5,
                use_pinv=True,
                rcond=1e-10,
                max_rows=50,  # show first 50 rows; set None for all
            )
        )
    except Exception as e:
        print(f"[summary_extended failed] {e}")

    # ---------------- explicit numeric SE block (kept + upgraded) ----------------
    print("\n=== Numeric Hessian SEs (explicit block) ===")
    num = numeric_se_from_model(
        model=model,
        y=y,
        X=X,
        use_model_scaler=True,     # IMPORTANT: match fit scale
        eps=1e-5,                  # try 1e-4 if unstable
        use_pinv=True,
        rcond=1e-10,
    )
    print(num.message)

    # Build readable names matching your packing order
    K = len(model.components)
    pX = X.shape[1]

    names = []
    for j in range(K - 1):
        names.append(f"eta_pi[{j}]")
    for k in range(K):
        for j in range(pX):
            names.append(f"beta[{k}][{j}]")
    for k, comp in enumerate(model.components):
        for en in comp.family.extra_param_names:
            names.append(f"extra[{k}].{en}_t")

    # Wald table using theta_hat + se
    wald = _wald_table(num.theta_hat, num.se, names, alpha=0.05)

    # print a compact view (top 15)
    print("\n--- Wald table (first 15) ---")
    print(wald.head(15).to_string(index=False))

    huge = int(np.sum(wald["se"] > 1e3))
    print(f"\nSE diagnostics: max={wald['se'].max():.3e}, huge(>1e3)={huge}/{len(wald)}")

    # Optional: save for paper appendix
    out_csv = "smoke_test_01_inference_numeric.csv"
    wald.to_csv(out_csv, index=False)
    print(f"Saved inference table to: {out_csv}")

    # --- quick sanity checks ---
    tau = res.responsibilities
    row_sums = np.max(np.abs(tau.sum(axis=1) - 1.0))
    print(f"\nmax |sum_k tau_ik - 1| = {row_sums:.3e}")

    # approximate classification accuracy (label switching possible)
    z_hat = np.argmax(tau, axis=1)
    acc1 = float(np.mean(z_hat == z_true))
    acc2 = float(np.mean((1 - z_hat) == z_true))
    acc = max(acc1, acc2)
    print(f"hard assignment accuracy (up to label switch) = {acc:.3f}")

    # mixture mean prediction sanity
    yhat = model.predict_mean(X)
    mse = float(np.mean((y - yhat) ** 2))
    r2v = _r2(y, yhat)
    print(f"mse of mixture mean predictor = {mse:.4f}")
    print(f"r2 of mixture mean predictor  = {r2v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
