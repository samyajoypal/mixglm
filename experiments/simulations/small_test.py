import numpy as np
import pandas as pd
import time
import sys

from mixglm.utils.repro import set_global_seed
from mixglm.sim.design import DesignConfig, make_design
from mixglm.sim.mixture import SimComponent, sample_mixture
from mixglm.sim.components import gaussian_sampler, student_t_sampler

from mixglm.links.identity import IdentityLink

from mixglm.selection.full_pipeline import full_beam_pipeline
from mixglm.utils.parallel import ParallelConfig

def main():
    set_global_seed(42)
    rng = np.random.default_rng(42)

    n, p = 800, 12
    # 3 active variables, 9 noise variables (should be set to 0 by Lasso)
    beta0 = np.array([0.5, 1.0, -0.7] + [0.0]*9)
    beta1 = np.array([-0.2, -0.6, 1.2] + [0.0]*9)

    X = make_design(DesignConfig(n=n, p=p, intercept=True, rho=0.1), rng=rng)
    link = IdentityLink()

    comps = [
        SimComponent(
            name="gaussian",
            beta=beta0,
            link=link,
            extra={"sigma": 0.8},
            sampler=gaussian_sampler,
        ),
        SimComponent(
            name="student_t",
            beta=beta1,
            link=link,
            extra={"df": 5.0, "scale": 0.8},
            sampler=student_t_sampler,
        ),
    ]
    pi_true = np.array([0.6, 0.4])
    sim = sample_mixture(X=X, components=comps, pi=pi_true, rng=rng)

    print("=== Small Test: Penalization & Model Selection ===")
    print(f"True model: K=2 (gaussian, student_t), n={n}, p={p}")
    print("True Beta0:", np.round(beta0, 2))
    print("True Beta1:", np.round(beta1, 2))
    print("-" * 50)

    t0 = time.time()
    # Run pipeline with a restricted grid to keep it fast
    best, all_bests = full_beam_pipeline(
        y=sim.y, X=X,
        kind="continuous",
        K_max=2,          # Max components = 2
        beam_width=3,     # Keep top 3 candidate models per step
        criterion="bic",
        do_none=True,     # Unpenalized fit
        ridge_grid=[],    # Skip ridge for now
        lasso_grid=[5.0, 10.0, 20.0, 50.0, 100.0], # Lasso to test sparsity
        enet_grid=[],
        em_kwargs={"max_iter": 150, "tol": 1e-5, "n_starts": 3},
        seed=42,
        init="quantile",
        compute_icl=False,
        standardize=True,
        verbose=False,
        parallel=ParallelConfig(n_jobs=-1, prefer="processes"),
        show_progress=False,
    )
    t1 = time.time()

    print(f"\nPipeline completed in {t1-t0:.2f} seconds.")
    print("\n--- Top Models Discovered (by Penalty) ---")
    rows = []
    for b in all_bests:
        rows.append({
            "Penalty": b.penalty,
            "K": b.K,
            "Families": " + ".join(b.families),
            "BIC": b.score,
        })
    print(pd.DataFrame(rows).sort_values("BIC").to_string(index=False))

    print(f"\n--- Overall Best Model ---")
    print(f"Selected Penalty: {best.penalty}")
    print(f"Selected Families: {best.families}")
    print(f"Selected K: {best.K}")
    print(f"BIC: {best.score:.2f}")

    if best.K == 2:
        betas_hat = best.model.betas_original_scale()
        # manual alignment based on L2 distance
        b0t, b1t = beta0, beta1
        b0h, b1h = betas_hat
        d00 = np.linalg.norm(b0t - b0h) + np.linalg.norm(b1t - b1h)
        d01 = np.linalg.norm(b0t - b1h) + np.linalg.norm(b1t - b0h)
        if d01 < d00:
            b0h, b1h = b1h, b0h

        print("\n--- Estimated Coefficients (True vs Hat) ---")
        df = pd.DataFrame({
            "Feature": [f"x{i}" for i in range(p)],
            "True_Beta0": b0t,
            "Hat_Beta0": np.round(b0h, 3),
            "True_Beta1": b1t,
            "Hat_Beta1": np.round(b1h, 3)
        })
        print(df.to_string(index=False))

        # Sparsity check
        print("\n--- Sparsity Check (Threshold = 1e-4) ---")
        zero_true0 = (b0t == 0)
        zero_hat0 = (np.abs(b0h) < 1e-4)
        zero_true1 = (b1t == 0)
        zero_hat1 = (np.abs(b1h) < 1e-4)

        print(f"Comp 0 (True Zeros: {zero_true0.sum()}): Hat Zeros: {zero_hat0.sum()}")
        print(f"Comp 1 (True Zeros: {zero_true1.sum()}): Hat Zeros: {zero_hat1.sum()}")

    print("\n--- Best Lasso Model Sparsity Check ---")
    lasso_bests = [b for b in all_bests if b.penalty == 'lasso' and b.K == 2]
    if lasso_bests:
        best_lasso = lasso_bests[0]
        betas_hat_lasso = best_lasso.model.betas_original_scale()
        b0t, b1t = beta0, beta1
        b0h, b1h = betas_hat_lasso
        d00 = np.linalg.norm(b0t - b0h) + np.linalg.norm(b1t - b1h)
        d01 = np.linalg.norm(b0t - b1h) + np.linalg.norm(b1t - b0h)
        if d01 < d00:
            b0h, b1h = b1h, b0h
        zero_hat0 = (np.abs(b0h) < 1e-4)
        zero_hat1 = (np.abs(b1h) < 1e-4)
        print(f"Lasso Comp 0 Hat Zeros: {zero_hat0.sum()} (out of {zero_true0.sum()} true zeros)")
        print(f"Lasso Comp 1 Hat Zeros: {zero_hat1.sum()} (out of {zero_true1.sum()} true zeros)")

if __name__ == '__main__':
    # Add src to path to ensure mixglm is importable if run directly
    sys.path.insert(0, './src')
    main()
