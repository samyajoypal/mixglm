import numpy as np
import pandas as pd
import time
import sys

from mixglm.utils.repro import set_global_seed
from mixglm.sim.design import DesignConfig, make_design
from mixglm.sim.mixture import SimComponent, sample_mixture
from mixglm.sim.components import poisson_sampler, nb2_sampler

from mixglm.links.log import LogLink

from mixglm.model.component import ComponentSpec
from mixglm.model.mixture_glm import MixtureGLM
from mixglm.families.registry import FAMILIES, register_defaults as register_families
from mixglm.penalties.registry import PENALTIES, register_defaults as register_penalties

def build_model(families, penalty_name, p):
    comps = []
    for f in families:
        if penalty_name == 'none':
            pen = PENALTIES.create('none')
        elif penalty_name == 'ridge':
            pen = PENALTIES.create('ridge', lam=5.0)
        elif penalty_name == 'lasso':
            pen = PENALTIES.create('lasso', lam=20.0)
        elif penalty_name == 'enet':
            pen = PENALTIES.create('elastic_net', lam=20.0, l1_ratio=0.5)

        comp = ComponentSpec(
            family=FAMILIES.create(f),
            link=LogLink(),
            penalty=pen
        )
        comps.append(comp)
    return MixtureGLM(components=comps)


def main():
    register_families()
    register_penalties()
    set_global_seed(42)
    rng = np.random.default_rng(42)

    n, p = 800, 8
    # LogLink means X@beta is exponentiated. Keep betas small to prevent explosion!
    beta0 = np.array([1.0, 0.4, -0.3] + [0.0]*5)
    beta1 = np.array([0.5, -0.5, 0.6] + [0.0]*5)

    # Use smaller variance for X to keep exp(X@beta) bounded
    X = rng.normal(scale=0.5, size=(n, p))
    # Intercept
    X[:, 0] = 1.0

    link = LogLink()

    comps = [
        SimComponent(
            name="poisson",
            beta=beta0,
            link=link,
            extra={},
            sampler=poisson_sampler,
        ),
        SimComponent(
            name="nb2",
            beta=beta1,
            link=link,
            extra={"alpha": 0.5}, # Overdispersion
            sampler=nb2_sampler,
        ),
    ]
    pi_true = np.array([0.7, 0.3])
    sim = sample_mixture(X=X, components=comps, pi=pi_true, rng=rng)

    print("=== Discrete Test: Poisson + Negative Binomial ===")
    print(f"True pi: {pi_true}")
    print(f"True Beta0: {np.round(beta0, 2)}")
    print(f"True Beta1: {np.round(beta1, 2)}")
    print(f"True NB Alpha: 0.5")
    print("-" * 50)

    families = ["poisson", "nb2"]

    # Test all penalty types
    for p_name in ['none', 'ridge', 'lasso', 'enet']:
        model = build_model(families, p_name, p)
        t0 = time.time()

        # We need init="random" or "quantile" for discrete, let's try random
        model.fit(
            y=sim.y, X=X,
            max_iter=150,
            tol=1e-5,
            n_starts=3,
            seed=42,
            init="random",
            standardize=True, # standardized internally, but we look at original scale
            verbose=False
        )
        t1 = time.time()

        if not model.result_.converged:
            print(f"\n{p_name.upper()} model did NOT converge!")
            continue

        print(f"\n[{p_name.upper()}] Fit in {t1-t0:.2f}s | BIC: {model.result_.bic:.2f} | Loglik: {model.result_.loglik:.2f}")

        betas_hat = model.result_.betas # we don't have betas_original_scale easily exposed without relying on the internal scaler logic, wait, we do: model.betas_original_scale()
        # let's try to align
        try:
            betas_hat = model.betas_original_scale()
        except AttributeError:
            # fallback
            betas_hat = model.result_.betas

        b0t, b1t = beta0, beta1
        b0h, b1h = betas_hat

        # Align
        d00 = np.linalg.norm(b0t - b0h) + np.linalg.norm(b1t - b1h)
        d01 = np.linalg.norm(b0t - b1h) + np.linalg.norm(b1t - b0h)
        if d01 < d00:
            b0h, b1h = b1h, b0h
            pi_hat = [model.result_.pi[1], model.result_.pi[0]]
        else:
            pi_hat = model.result_.pi

        print(f"  pi_hat: {np.round(pi_hat, 3)}")
        print(f"  Beta0 hat: {np.round(b0h, 3)}")
        print(f"  Beta1 hat: {np.round(b1h, 3)}")

        if p_name in ['lasso', 'enet']:
            zero_true0 = (beta0 == 0).sum()
            zero_true1 = (beta1 == 0).sum()
            zero_hat0 = (np.abs(b0h) < 1e-4).sum()
            zero_hat1 = (np.abs(b1h) < 1e-4).sum()
            print(f"  Sparsity -> Comp0: {zero_hat0}/{zero_true0} | Comp1: {zero_hat1}/{zero_true1}")

        # For the unpenalized model, test the inference table!
        if p_name == 'none':
            print("\n--- INFERENCE TABLE (Numeric SE) for UNPENALIZED MODEL ---")
            try:
                df_inf, _ = model.inference_table(y=sim.y, X=X, method='numeric')
                print(df_inf.to_string(index=False))
            except Exception as e:
                print(f"Inference failed: {e}")

if __name__ == '__main__':
    sys.path.insert(0, './src')
    main()
