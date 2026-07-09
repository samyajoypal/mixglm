# src/mixglm/families/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, List

from mixglm.families.base import UnivariateFamily


@dataclass
class FamilyRegistry:
    """
    Simple registry for univariate families.

    Allows:
    - registering a family factory under a name
    - creating families by name (useful for selection/model-space exploration)
    """
    _factories: Dict[str, Callable[[], UnivariateFamily]]

    def __init__(self) -> None:
        self._factories = {}

    def register(self, name: str, factory: Callable[[], UnivariateFamily], *, overwrite: bool = False) -> None:
        key = str(name).lower()
        if (key in self._factories) and (not overwrite):
            raise KeyError(f"Family '{name}' already registered. Use overwrite=True to replace.")
        self._factories[key] = factory

    def create(self, name: str) -> UnivariateFamily:
        key = str(name).lower()
        if key not in self._factories:
            raise KeyError(f"Family '{name}' not registered. Available: {sorted(self._factories.keys())}")
        return self._factories[key]()

    def available(self) -> List[str]:
        return sorted(self._factories.keys())


# Global registry instance
FAMILIES = FamilyRegistry()


def register_defaults() -> None:
    """
    Register built-in families.
    Safe to call multiple times (overwrite=True).
    """
    # local imports to avoid heavy import chains
    from mixglm.families.gaussian import GaussianFamily
    from mixglm.families.student_t import StudentTFamily

    from mixglm.families.poisson import PoissonFamily
    from mixglm.families.negative_binomial import NegativeBinomial2Family

    from mixglm.families.gamma import GammaFamily
    from mixglm.families.exponential import ExponentialFamily
    from mixglm.families.lognormal import LogNormalFamily
    from mixglm.families.inverse_gaussian import InverseGaussianFamily

    from mixglm.families.bernoulli import BernoulliFamily
    from mixglm.families.geometric import GeometricFamily
    from mixglm.families.zip import ZIPoissonFamily
    from mixglm.families.zinb import ZINB2Family

    from mixglm.families.beta import BetaFamily

    from mixglm.families.skew_normal import SkewNormalFamily
    from mixglm.families.genhyperbolic import GeneralizedHyperbolicFamily

    # skew-t variants
    from mixglm.families.skew_t import SkewTFamily              # SciPy jf_skew_t
    from mixglm.families.azzalini_skew_t import AzzaliniSkewTFamily  # external skewt-scipy

    # --- register ---
    FAMILIES.register("gaussian", lambda: GaussianFamily(), overwrite=True)
    FAMILIES.register("student_t", lambda: StudentTFamily(), overwrite=True)

    FAMILIES.register("poisson", lambda: PoissonFamily(), overwrite=True)
    FAMILIES.register("nb2", lambda: NegativeBinomial2Family(), overwrite=True)

    FAMILIES.register("gamma", lambda: GammaFamily(), overwrite=True)
    FAMILIES.register("exponential", lambda: ExponentialFamily(), overwrite=True)
    FAMILIES.register("lognormal", lambda: LogNormalFamily(), overwrite=True)
    FAMILIES.register("inverse_gaussian", lambda: InverseGaussianFamily(), overwrite=True)

    FAMILIES.register("bernoulli", lambda: BernoulliFamily(), overwrite=True)
    FAMILIES.register("geometric", lambda: GeometricFamily(), overwrite=True)
    FAMILIES.register("zip", lambda: ZIPoissonFamily(), overwrite=True)
    FAMILIES.register("zinb", lambda: ZINB2Family(), overwrite=True)

    FAMILIES.register("beta", lambda: BetaFamily(), overwrite=True)

    FAMILIES.register("skew_normal", lambda: SkewNormalFamily(), overwrite=True)
    FAMILIES.register("genhyperbolic", lambda: GeneralizedHyperbolicFamily(), overwrite=True)

    # skew-t: keep both names explicit to avoid ambiguity in papers
    FAMILIES.register("jf_skew_t", lambda: SkewTFamily(), overwrite=True)
    FAMILIES.register("azzalini_skew_t", lambda: AzzaliniSkewTFamily(), overwrite=True)
