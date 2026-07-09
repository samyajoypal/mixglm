# src/mixglm/inference/wrappers.py
from __future__ import annotations

from typing import Optional

import numpy as np

from mixglm.inference.louis import louis_observed_information, LouisResult
from mixglm.model.mixture_glm import MixtureGLM

Array = np.ndarray


def louis_from_model(
    *,
    model: MixtureGLM,
    y: Array,
    X: Array,
    use_model_scaler: bool = True,
    fd_eps: float = 1e-5,
    diag_hessian: bool = False,
    ridge: float = 1e-8,
    derivative_method: str = "auto",
) -> LouisResult:
    """
    Compute Louis observed information using a fitted MixtureGLM.

    Parameters
    ----------
    model : MixtureGLM
        Fitted mixture GLM model.
    y : array-like, shape (n,)
        Response vector (original scale).
    X : array-like, shape (n, p)
        Design matrix on ORIGINAL scale.
    use_model_scaler : bool, default=True
        If True and the model was fitted with standardization,
        X is transformed using model.scaler_ before Louis.
    fd_eps : float
        Finite-difference step size.
    diag_hessian : bool
        If True, only diagonal Hessian blocks are computed (faster).
    ridge : float
        Small ridge added to observed information for numerical stability.
    derivative_method : {"auto", "analytic", "finite_diff"}
        Component derivative engine. "auto" uses closed forms when available
        and finite differences otherwise.

    Returns
    -------
    LouisResult
        Contains observed information, covariance, standard errors, and parameter names.
    """
    if model.result_ is None:
        raise ValueError("Model must be fitted before calling louis_from_model().")

    y = np.asarray(y)
    X = np.asarray(X, dtype=float)

    # --- apply the SAME transformation as used in fitting ---
    if use_model_scaler and getattr(model, "standardize_", False):
        if model.scaler_ is None:
            raise RuntimeError("Model indicates standardization but scaler_ is None.")
        X_use = model.scaler_.transform(X)
    else:
        X_use = X

    res = model.result_

    return louis_observed_information(
        y=y,
        X=X_use,
        components=model.components,
        pi=res.pi,
        betas=res.betas,
        extras=res.extras,
        tau=res.responsibilities,
        fd_eps=fd_eps,
        diag_hessian=diag_hessian,
        ridge=ridge,
        derivative_method=derivative_method,
    )
