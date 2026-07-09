# src/mixglm/model/__init__.py
from .component import ComponentSpec
from .mixture_glm import MixtureGLM
from .results import MixtureGLMResult

__all__ = [
    "ComponentSpec",
    "MixtureGLM",
    "MixtureGLMResult",
]
