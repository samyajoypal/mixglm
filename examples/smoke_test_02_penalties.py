# examples/smoke_test_02_penalties.py

from __future__ import annotations

import numpy as np

from mixglm.utils.repro import set_global_seed
from mixglm.model import ComponentSpec, MixtureGLM

from mixglm.families.gaussian import GaussianFamily
from mixglm.links.identity import IdentityLink

from mixglm.penalties.base import NoPenalty
from mixglm.penalties.ridge import RidgePenalty
from mixglm.penalties.lasso import LassoPenalty
from mixglm.penalties.elastic_net import ElasticNetPenalty


def simulate_sparse_two_component_gaussian(
    *,
    n: int,
    p: int,
    pi: np.ndarray,
    beta0: np.ndarray,
    beta1: np.ndarray,
    sigma0: float,
    sigma1: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-component Gaussian mixture with different regression coefficients.
    """
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0  # intercept

    z = rng.choice(2, size=n, p=pi)
    mu0 = X @ beta0
    mu1 = X @ beta1

    y = np.empty(n, dtype=float)
    m0 = z == 0
    y[m0] = mu0[m0] + rng.normal(0.0, sigma0, size=int(m0.sum()))
    y[~m0] = mu1[~m0] + rng.normal(0.0, sigma1, size=int((~m0).sum()))
    return y, X, z


def fit_one(
    y: np.ndarray,
    X: np.ndarray,
    penalty0,
    penalty1,
    *,
    seed: int = 123,
    verbose: bool = False,
) -> MixtureGLM:
    comp0 = ComponentSpec(
        family=GaussianFamily(),
        link=IdentityLink(),
        penalty=penalty0,
    )
    comp1 = ComponentSpec(
        family=GaussianFamily(),
        link=IdentityLink(),
        penalty=penalty1,
    )
    model = MixtureGLM([comp0, comp1])
    model.fit(
        y=y,
        X=X,
        max_iter=150,
        tol=1e-6,
        n_starts=5,
        seed=seed,
        init="kmeans_y",
        verbose=verbose,
        inner_mstep_iter=2,
        min_pi=1e-6,
        compute_icl=True,
    )
    return model


def l0_count(beta: np.ndarray, thr: float = 1e-2) -> int:
    # number of "active" coefficients excluding intercept
    b = np.asarray(beta, dtype=float).copy()
    if b.size <= 1:
        return 0
    return int(np.sum(np.abs(b[1:]) > thr))


def main() -> None:
    rng = set_global_seed(123)

    # --- simulate sparse truth ---
    n = 1000
    p = 15  # includes intercept
    pi_true = np.array([0.5, 0.5])

    beta0_true = np.zeros(p)
    beta1_true = np.zeros(p)

    # intercepts
    beta0_true[0] = 0.5
    beta1_true[0] = -0.25

    # only 3 nonzero covariates (sparse)
    beta0_true[1] = 1.25
    beta0_true[3] = -0.9
    beta0_true[7] = 0.6

    beta1_true[1] = 0.4
    beta1_true[3] = -1.2
    beta1_true[9] = 0.8

    sigma0_true = 1.0
    sigma1_true = 1.0

    y, X, z_true = simulate_sparse_two_component_gaussian(
        n=n, p=p, pi=pi_true,
        beta0=beta0_true, beta1=beta1_true,
        sigma0=sigma0_true, sigma1=sigma1_true,
        rng=rng,
    )

    print("=== Smoke test 02: penalties (Gaussian + Gaussian) ===")
    print(f"n={n}, p={p}, true nonzeros comp0={l0_count(beta0_true)}, comp1={l0_count(beta1_true)}")
    print()

    configs = [
    ("none", NoPenalty(), NoPenalty()),
    ("ridge(lam=50)", RidgePenalty(lam=50.0), RidgePenalty(lam=50.0)),
    ("lasso(lam=1.0)", LassoPenalty(lam=1.0), LassoPenalty(lam=1.0)),
    ("enet(lam=1.0,l1_ratio=0.7)", ElasticNetPenalty(lam=1.0, l1_ratio=0.7), ElasticNetPenalty(lam=1.0, l1_ratio=0.7)),
    ]


    for name, p0, p1 in configs:
        print(f"\n--- Fit: {name} ---")
        model = fit_one(y, X, p0, p1, seed=123, verbose=False)
        res = model.result_

        print(f"converged={res.converged}, loglik={res.loglik:.3f}, BIC={res.bic:.3f}, pi={np.round(res.pi, 4)}")

        # label switching: we just report both betas
        b0, b1 = res.betas
        nz0 = l0_count(b0)
        nz1 = l0_count(b1)

        print(f"active(nonzero) betas (excluding intercept): comp0={nz0}, comp1={nz1}")
        print(f"||beta||_2: comp0={np.linalg.norm(b0):.3f}, comp1={np.linalg.norm(b1):.3f}")

        # quick sanity: responsibilities sum to 1
        tau = res.responsibilities
        max_row_err = float(np.max(np.abs(tau.sum(axis=1) - 1.0)))
        print(f"max |sum_k tau_ik - 1| = {max_row_err:.3e}")
        top0 = np.sort(np.abs(b0[1:]))[::-1][:5]
        top1 = np.sort(np.abs(b1[1:]))[::-1][:5]
        print(f"top5 |beta| (excl intercept): comp0={np.round(top0,4)}, comp1={np.round(top1,4)}")


    print("\nDone.")


if __name__ == "__main__":
    main()
