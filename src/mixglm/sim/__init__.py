# src/mixglm/sim/__init__.py

from .design import DesignConfig, make_design
from .mixture import SimComponent, MixtureSimResult, sample_mixture

__all__ = [
    "DesignConfig",
    "make_design",
    "SimComponent",
    "MixtureSimResult",
    "sample_mixture",
]
