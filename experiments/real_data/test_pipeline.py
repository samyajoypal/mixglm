import sys
import numpy as np
import time

sys.path.insert(0, './src')
from mixglm.families.registry import FAMILIES, register_defaults
from mixglm.penalties.registry import PENALTIES, register_defaults as register_pen
from mixglm.model.component import ComponentSpec
from mixglm.model.mixture_glm import MixtureGLM
from mixglm.links.identity import IdentityLink
from mixglm.links.log import LogLink

register_defaults()
register_pen()

def run_test(name, X, y, comps_baseline, comps_hetero, kind="continuous"):
    # Subsample for quick local test
    idx = np.random.choice(X.shape[0], min(500, X.shape[0]), replace=False)
    Xs = X[idx]
    ys = y[idx]

    # Baseline (Homogeneous K=1)
    print(f"\n--- {name} ---")
    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features. Subsampled to 500 for fast local test.")

    t0 = time.time()
    m_base = MixtureGLM(components=comps_baseline)
    m_base.fit(y=ys, X=Xs, init="random" if kind=="counts" else "quantile", standardize=True, verbose=False)
    t1 = time.time()
    bic_base = m_base.result_.bic if m_base.result_.converged else np.inf
    print(f"Baseline ({comps_baseline[0].family.name}): BIC = {bic_base:.2f}, Time = {t1-t0:.2f}s")

    # Heterogeneous (K=2) with Lasso
    t0 = time.time()
    m_het = MixtureGLM(components=comps_hetero)
    m_het.fit(y=ys, X=Xs, init="random" if kind=="counts" else "quantile", standardize=True, verbose=False)
    t1 = time.time()
    bic_het = m_het.result_.bic if m_het.result_.converged else np.inf

    names = [c.family.name for c in comps_hetero]
    print(f"Heterogeneous ({' + '.join(names)}) + Lasso: BIC = {bic_het:.2f}, Time = {t1-t0:.2f}s")

    if m_het.result_.converged:
        betas = m_het.betas_original_scale()
        for k, b in enumerate(betas):
            active = np.sum(np.abs(b) > 1e-4)
            print(f"  Component {k+1} ({names[k]}): {active}/{len(b)} active variables.")
    else:
        print("  Heterogeneous model did not converge on this subsample.")

    if bic_het < bic_base:
        print("CONCLUSION: Heterogeneous mixture selected as SUPERIOR (Lower BIC).")

def main():
    np.random.seed(42)

    # 1. RAND Health
    try:
        yr = np.load("data/rand_y.npy")
        Xr = np.load("data/rand_X.npy")

        pen_none = PENALTIES.create("none")
        pen_lasso = PENALTIES.create("lasso", lam=5.0)

        c_base = [ComponentSpec(family=FAMILIES.create("nb2"), link=LogLink(), penalty=pen_none)]
        c_het = [
            ComponentSpec(family=FAMILIES.create("zip"), link=LogLink(), penalty=pen_lasso),
            ComponentSpec(family=FAMILIES.create("zinb"), link=LogLink(), penalty=pen_lasso)
        ]
        run_test("RAND Health (Count)", Xr, yr, c_base, c_het, kind="counts")
    except Exception as e:
        print("Error in RAND:", e)

    # 2. California Housing
    try:
        yc = np.load("data/cali_y.npy")
        Xc = np.load("data/cali_X.npy")

        pen_none = PENALTIES.create("none")
        pen_lasso = PENALTIES.create("lasso", lam=10.0)

        c_base = [ComponentSpec(family=FAMILIES.create("gaussian"), link=IdentityLink(), penalty=pen_none)]
        c_het = [
            ComponentSpec(family=FAMILIES.create("lognormal"), link=IdentityLink(), penalty=pen_lasso),
            ComponentSpec(family=FAMILIES.create("gamma"), link=IdentityLink(), penalty=pen_lasso)
        ]
        run_test("California Housing (Continuous)", Xc, yc, c_base, c_het, kind="continuous")
    except Exception as e:
        print("Error in Cali:", e)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
