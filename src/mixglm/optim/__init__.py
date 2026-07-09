# src/mixglm/optim/__init__.py
from .finite_diff import finite_diff_grad
from .proximal import prox_grad
from .scipy_opt import scipy_minimize_box

__all__ = ["finite_diff_grad", "prox_grad", "scipy_minimize_box"]
