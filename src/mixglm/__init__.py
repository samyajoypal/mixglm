# src/mixglm/__init__.py
"""
mixglm: Finite mixture models in a GLM framework (univariate response),
supporting non-identical component families and penalized EM estimation.
"""

from mixglm.model import MixtureGLM, ComponentSpec, MixtureGLMResult

__all__ = ["MixtureGLM", "ComponentSpec", "MixtureGLMResult"]
__version__ = "0.1.0"
