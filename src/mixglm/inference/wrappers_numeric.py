# src/mixglm/inference/wrappers_numeric.py
from __future__ import annotations
import numpy as np

from mixglm.inference.numeric_se import numeric_hessian_se, NumericSE
from mixglm.model.mixture_glm import MixtureGLM

Array = np.ndarray


def numeric_se_from_model(
    *,
    model: MixtureGLM,
    y: Array,
    X: Array,
    use_model_scaler: bool = True,
    eps: float = 1e-5,
    use_pinv: bool = True,
    rcond: float = 1e-10,
) -> NumericSE:
    """
    Numeric Hessian SE wrapper that matches the model's internal X scaling.
    """
    if model.result_ is None:
        raise ValueError("Model must be fitted.")

    y = np.asarray(y)
    X = np.asarray(X, dtype=float)

    if use_model_scaler and getattr(model, "standardize_", False):
        if model.scaler_ is None:
            raise RuntimeError("Model says standardize_=True but scaler_ is None.")
        X_use = model.scaler_.transform(X)
    else:
        X_use = X

    return numeric_hessian_se(
        model=model,
        y=y,
        X=X_use,
        eps=eps,
        use_pinv=use_pinv,
        rcond=rcond,
    )
