# src/mixglm/families/__init__.py
from .base import UnivariateFamily, FamilySupport, FamilyParams
from .gaussian import GaussianFamily

__all__ = ["UnivariateFamily", "FamilySupport", "FamilyParams", "GaussianFamily"]
