# examples/smoke_test_05_full_pipeline.py
from __future__ import annotations
import numpy as np
import pandas as pd

from mixglm.utils.repro import set_global_seed
from mixglm.sim.design import DesignConfig, make_design
from mixglm.sim.mixture import SimComponent, sample_mixture
from mixglm.sim.components import gaussian_sampler, student_t_sampler

from mixglm.links.identity import IdentityLink

from mixglm.selection.full_pipeline import full_beam_pipeline
from mixglm.utils.parallel import ParallelConfig


def align_labels_by_beta(betas_true, betas_hat):
    # K=2 alignment by L2 distance
    b0t, b1t = betas_true
    b0h, b1h = betas_hat
    d00 = np.linalg.norm(b0t - b0h) + np.linalg.norm(b1t - b1h)
    d01 = np.linalg.norm(b0t - b1h) + np.linalg.norm(b1t - b0h)
    return (0, 1) if d00 <= d01 else (1, 0)


def r2(y_true, y_hat):
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)
    ssr = np.sum((y_true - y_hat) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ssr / max(sst, 1e-12)


def main():
    set_global_seed(123)
    rng = np.random.default_rng(123)

    # ---------- simulate non-identical mixture ----------
    n, p = 1200, 6
    X = make_design(DesignConfig(n=n, p=p, intercept=True, rho=0.2), rng=rng)

    link = IdentityLink()

    beta0 = np.array([0.5, 1.0, -0.7, 0.0, 0.0, 0.0])
    beta1 = np.array([-0.2, -0.6, 0.0, 1.2, 0.0, 0.0])

    comps = [
        SimComponent(
            name="gaussian",
            beta=beta0,
            link=link,
            extra={"sigma": 0.9},
            sampler=gaussian_sampler,
        ),
        SimComponent(
            name="student_t",
            beta=beta1,
            link=link,
            extra={"df": 6.0, "scale": 0.9},
            sampler=student_t_sampler,
        ),
    ]
    pi_true = np.array([0.6, 0.4])
    sim = sample_mixture(X=X, components=comps, pi=pi_true, rng=rng)

    y, z = sim.y, sim.z
    Ey_true = pi_true[0] * sim.comp_loc[0] + pi_true[1] * sim.comp_loc[1]

    print("=== Smoke test 05: Full pipeline (beam search + penalties) ===")
    print(f"True model: gaussian + student_t, K=2, pi={pi_true.tolist()}")

    # ---------- full selection pipeline ----------
    best, all_bests = full_beam_pipeline(
        y=y, X=X,
        kind="continuous",
        K_max=3,
        beam_width=5,
        criterion="bic",
        do_none=True,
        ridge_grid=[0.1, 1.0, 10.0, 50.0],
        lasso_grid=[0.05, 0.1, 0.2, 0.5, 1.0],
        enet_grid=[(0.1, 0.3), (0.1, 0.7), (0.5, 0.5), (1.0, 0.7)],
        em_kwargs={"max_iter": 200, "tol": 1e-6, "n_starts": 5, "compute_icl": True},
        seed=123,
        init="quantile",
        compute_icl=True,
        standardize=True,
        verbose=False,
        parallel=ParallelConfig(n_jobs=-1, prefer="processes"),  # use all cores
        show_progress=True,
    )

    # ---------- table: best per penalty ----------
    rows = []
    for b in all_bests:
        res = b.model.result_
        Ey_hat = b.model.predict_mean(X)
        rows.append({
            "penalty": b.penalty,
            "K": b.K,
            "families": " + ".join(b.families),
            "score(bic)": b.score,
            "loglik": float(res.loglik),
            "pi_hat": np.round(res.pi, 4).tolist(),
            "mse_Ey": float(np.mean((Ey_true - Ey_hat) ** 2)),
            "mae_Ey": float(np.mean(np.abs(Ey_true - Ey_hat))),
            "r2_Ey": float(r2(Ey_true, Ey_hat)),
        })

    df = pd.DataFrame(rows).sort_values("score(bic)")
    print("\n--- Best model per penalty (sorted by BIC) ---")
    print(df.to_string(index=False))

    # ---------- truth vs estimated coefficients for the overall best ----------
    print("\n--- Overall BEST ---")
    print(f"penalty={best.penalty}, K={best.K}, families={best.families}, BIC={best.score:.3f}")

    if best.K == 2:
        betas_hat = best.model.betas_original_scale()
        perm = align_labels_by_beta([beta0, beta1], betas_hat)
        bh0, bh1 = betas_hat[perm[0]], betas_hat[perm[1]]

        comp_tbl = pd.DataFrame({
            "feature": [f"b{j}" for j in range(p)],
            "beta0_true": beta0,
            "beta0_hat": bh0,
            "abs_err0": np.abs(beta0 - bh0),
            "beta1_true": beta1,
            "beta1_hat": bh1,
            "abs_err1": np.abs(beta1 - bh1),
        })
        print("\n--- Coefficients (aligned by beta distance) ---")
        print(comp_tbl.to_string(index=False))


if __name__ == "__main__":
    main()

    '''
    --- Best model per penalty (sorted by BIC) ---
                       penalty  K             families  score(bic)       loglik           pi_hat   mse_Ey   mae_Ey    r2_Ey
                          none  2 student_t + gaussian 4017.123472 -1951.841121 [0.3603, 0.6397] 0.007900 0.071013 0.981918
                         lasso  2 student_t + gaussian 4017.123684 -1951.841227 [0.3603, 0.6397] 0.007896 0.070992 0.981928
    enet(lam=0.1,l1_ratio=0.7)  2 student_t + gaussian 4017.123901 -1951.841336 [0.3603, 0.6397] 0.007893 0.070979 0.981934
                         ridge  2 student_t + gaussian 4017.124042 -1951.841406 [0.3603, 0.6397] 0.007905 0.071032 0.981908

    --- Overall BEST ---
    penalty=none, K=2, families=('student_t', 'gaussian'), BIC=4017.123

    --- Coefficients (aligned by beta distance) ---
    feature  beta0_true  beta0_hat  abs_err0  beta1_true  beta1_hat  abs_err1
         b0         0.5   0.483505  0.016495        -0.2  -0.180907  0.019093
         b1         1.0   1.008559  0.008559        -0.6  -0.677881  0.077881
         b2        -0.7  -0.678449  0.021551         0.0  -0.001015  0.001015
         b3         0.0  -0.031292  0.031292         1.2   1.225627  0.025627
         b4         0.0  -0.031334  0.031334         0.0   0.046410  0.046410
         b5         0.0   0.082733  0.082733         0.0  -0.025006  0.025006

    '''