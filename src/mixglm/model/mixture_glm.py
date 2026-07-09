# v1
# # src/mixglm/model/mixture_glm.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import List, Dict, Optional, Sequence, Tuple, Any

# import numpy as np

# from mixglm.families.base import UnivariateFamily
# from mixglm.links.base import BaseLink, Link
# from mixglm.penalties.base import BasePenalty, NoPenalty

# Array = np.ndarray


# # ----------------------------- results containers -----------------------------

# @dataclass
# class ComponentSpec:
    # """
    # Specification of one component: (family, link, penalty).

    # family: UnivariateFamily
    # link:   Link used for mu = link.inverse(eta)
    # penalty: penalty applied to beta for this component
    # """
    # family: UnivariateFamily
    # link: Link
    # penalty: BasePenalty

    # def describe(self) -> str:
        # return f"{self.family.name} | link={self.link.name} | penalty={self.penalty.name}(lam={self.penalty.lam})"


# @dataclass
# class MixtureGLMResult:
    # """
    # Fitted model results container.
    # """
    # converged: bool
    # n_iter: int
    # loglik: float
    # bic: float
    # aic: float
    # icl: Optional[float]

    # pi: Array                     # shape (K,)
    # betas: List[Array]            # list of length K, each shape (p,)
    # extras: List[Dict[str, Any]]  # list of length K, nuisance params per component
    # responsibilities: Array       # shape (n, K)

    # history: Dict[str, List[float]]  # e.g. {"loglik": [...], "obj": [...]}


# # ----------------------------- main user-facing model -----------------------------

# class MixtureGLM:
    # """
    # Mixture of GLMs (possibly non-identical component families), estimated by (penalized) EM.

    # Model:
        # p(y_i | x_i) = sum_{k=1}^K pi_k * f_{d_k}(y_i; mu_{ik}, extra_k),
        # mu_{ik} = g_k^{-1}(x_i^T beta_k)

    # Notes:
    # - Component families may differ (non-identical mixture).
    # - Penalties apply component-wise to beta_k (ridge/lasso/elastic-net).
    # - Nuisance parameters extra_k are component-specific and constant across i.

    # The EM implementation lives in mixglm.em.pem (created next).
    # """

    # def __init__(self, components: Sequence[ComponentSpec]):
        # if len(components) == 0:
            # raise ValueError("MixtureGLM requires at least one component.")
        # self.components: List[ComponentSpec] = list(components)
        # self.K: int = len(self.components)

        # # Fitted attributes
        # self.result_: Optional[MixtureGLMResult] = None

    # # ----------------------------- core helpers -----------------------------

    # @staticmethod
    # def _as_1d(y: Array) -> Array:
        # y = np.asarray(y)
        # if y.ndim != 1:
            # raise ValueError("y must be a 1D array of shape (n,).")
        # if np.any(~np.isfinite(y)):
            # raise ValueError("y contains non-finite values.")
        # return y

    # @staticmethod
    # def _as_2d(X: Array) -> Array:
        # X = np.asarray(X)
        # if X.ndim != 2:
            # raise ValueError("X must be a 2D array of shape (n, p).")
        # if np.any(~np.isfinite(X)):
            # raise ValueError("X contains non-finite values.")
        # return X

    # def _validate_data(self, y: Array, X: Array) -> Tuple[Array, Array]:
        # y = self._as_1d(y)
        # X = self._as_2d(X)
        # if X.shape[0] != y.shape[0]:
            # raise ValueError("X and y must have the same number of rows.")
        # # Validate support against each family (screening safety)
        # for comp in self.components:
            # comp.family.support.validate_y(y)
        # return y, X

    # # ----------------------------- public API -----------------------------

    # def fit(
        # self,
        # y: Array,
        # X: Array,
        # *,
        # max_iter: int = 200,
        # tol: float = 1e-6,
        # n_starts: int = 5,
        # seed: Optional[int] = None,
        # init: str = "quantile",  # "quantile" | "random" | "kmeans_y"
        # verbose: bool = False,
        # em_kwargs: Optional[Dict[str, Any]] = None,
    # ) -> "MixtureGLM":
        # """
        # Fit model by penalized EM.

        # Parameters
        # ----------
        # y : array (n,)
        # X : array (n,p)
        # max_iter : EM iterations
        # tol : convergence tolerance (relative improvement in objective)
        # n_starts : number of random/heuristic initializations; keep best final loglik
        # seed : RNG seed
        # init : initialization strategy
        # verbose : print progress
        # em_kwargs : extra options forwarded to EM implementation

        # Returns
        # -------
        # self
        # """
        # y, X = self._validate_data(y, X)
        # em_kwargs = {} if em_kwargs is None else dict(em_kwargs)

        # # import lazily to avoid circular deps while we build the project
        # from mixglm.em.pem import fit_pem  # to be created next

        # res = fit_pem(
            # y=y,
            # X=X,
            # components=self.components,
            # max_iter=max_iter,
            # tol=tol,
            # n_starts=n_starts,
            # seed=seed,
            # init=init,
            # verbose=verbose,
            # **em_kwargs,
        # )
        # self.result_ = res
        # return self

    # def predict_mean(self, X: Array) -> Array:
        # """
        # Predict E[Y|X] under the fitted mixture model.

        # For each x_i:
            # E[Y|x_i] = sum_k pi_k * m_k(mu_{ik}, extra_k)
        # where m_k is the component mean as a function of its parameters.

        # For a first implementation, we approximate m_k by mu_{ik} when the family
        # is parameterized so that mu is the mean.
        # Concrete families can later override a method to compute mean precisely.
        # """
        # if self.result_ is None:
            # raise RuntimeError("Model is not fitted yet.")
        # X = self._as_2d(X)
        # n, p = X.shape
        # K = self.K

        # pi = self.result_.pi
        # betas = self.result_.betas
        # extras = self.result_.extras

        # mu = np.zeros((n, K), dtype=float)
        # for k, comp in enumerate(self.components):
            # eta = X @ betas[k]
            # mu[:, k] = comp.link.inverse(eta)

        # # first-pass: treat mu as the conditional mean
        # return (mu * pi.reshape(1, -1)).sum(axis=1)

    # def predict_responsibilities(self, y: Array, X: Array) -> Array:
        # """
        # Compute posterior responsibilities tau_{ik} for new data under fitted params.
        # """
        # if self.result_ is None:
            # raise RuntimeError("Model is not fitted yet.")
        # y, X = self._validate_data(y, X)

        # pi = self.result_.pi
        # betas = self.result_.betas
        # extras = self.result_.extras

        # n = y.shape[0]
        # K = self.K
        # log_terms = np.empty((n, K), dtype=float)

        # for k, comp in enumerate(self.components):
            # eta = X @ betas[k]
            # mu = comp.link.inverse(eta)
            # ll = comp.family.loglik_component(y=y, mu=mu, extra=extras[k])
            # log_terms[:, k] = np.log(pi[k]) + ll

        # # stable normalize: tau = exp(log_terms - logsumexp)
        # m = np.max(log_terms, axis=1, keepdims=True)
        # denom = m + np.log(np.sum(np.exp(log_terms - m), axis=1, keepdims=True))
        # tau = np.exp(log_terms - denom)
        # return tau

    # def loglik(self) -> float:
        # if self.result_ is None:
            # raise RuntimeError("Model is not fitted yet.")
        # return float(self.result_.loglik)

    # def information_criteria(self) -> Dict[str, float]:
        # if self.result_ is None:
            # raise RuntimeError("Model is not fitted yet.")
        # out = {"aic": float(self.result_.aic), "bic": float(self.result_.bic)}
        # if self.result_.icl is not None:
            # out["icl"] = float(self.result_.icl)
        # return out

    # def summary(self) -> str:
        # if self.result_ is None:
            # return "MixtureGLM(not fitted)"
        # lines = []
        # lines.append(f"MixtureGLM(K={self.K})")
        # lines.append(f"  converged: {self.result_.converged} in {self.result_.n_iter} iterations")
        # lines.append(f"  loglik: {self.result_.loglik:.6f}")
        # lines.append(f"  AIC: {self.result_.aic:.6f} | BIC: {self.result_.bic:.6f}" + (f" | ICL: {self.result_.icl:.6f}" if self.result_.icl is not None else ""))
        # lines.append("  components:")
        # for k, comp in enumerate(self.components):
            # lines.append(f"    [{k}] pi={self.result_.pi[k]:.4f} | {comp.describe()}")
        # return "\n".join(lines)

# # v2
# # src/mixglm/model/mixture_glm.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Sequence, Tuple

# import numpy as np

# from mixglm.model.component import ComponentSpec
# from mixglm.model.results import MixtureGLMResult
# from mixglm.em.pem import fit_pem
# # from mixglm.selection.criteria import evaluate_criteria

# from mixglm.utils.checks import check_same_n
# from mixglm.utils.standardize import Standardizer


# Array = np.ndarray


# class MixtureGLM:
    # """
    # Mixture-GLM model where the response follows a finite mixture distribution and
    # each component has its own regression for the location parameter.

    # For component k:
        # Y_i | Z_i=k ~ f_k( y_i | mu_{ik}, extra_k )
        # mu_{ik} = g_k^{-1}( x_i^T beta_k )

    # Mixing proportions are constant:
        # P(Z_i=k) = pi_k
    # """

    # def __init__(self, components: Sequence[ComponentSpec]) -> None:
        # comps = list(components)
        # if len(comps) < 1:
            # raise ValueError("MixtureGLM requires at least one component.")
        # self.components: List[ComponentSpec] = comps
        # self.result_: Optional[MixtureGLMResult] = None
        # self.standardize_ = False
        # self.scaler_: Standardizer | None = None


    # def fit(
        # self,
        # y: Array,
        # X: Array,
        # *,
        # max_iter: int = 200,
        # tol: float = 1e-6,
        # n_starts: int = 5,
        # seed: Optional[int] = None,
        # init: str = "quantile",
        # verbose: bool = False,
        # inner_mstep_iter: int = 2,
        # min_pi: float = 1e-6,
        # compute_icl: bool = True,
        # standardize: bool = True,
    # ) -> "MixtureGLM":
        # y, X = check_same_n(y, X)
        # self.standardize_ = bool(standardize)

        # X_in = np.asarray(X, dtype=float)
        # if self.standardize_:
            # self.scaler_ = Standardizer(intercept_col=0).fit(X_in)
            # X_fit = self.scaler_.transform(X_in)
        # else:
            # self.scaler_ = None
            # X_fit = X_in

        # res = fit_pem(
            # y=y,
            # X=X_fit,
            # components=self.components,
            # max_iter=max_iter,
            # tol=tol,
            # n_starts=n_starts,
            # seed=seed,
            # init=init,
            # verbose=verbose,
            # inner_mstep_iter=inner_mstep_iter,
            # min_pi=min_pi,
            # compute_icl=compute_icl,
        # )
        # self.result_ = res
        # return self

    # # ------------------------- predictions -------------------------

    # def predict_component_means(self, X: Array) -> Array:
        # """
        # Returns component-wise means mu_{ik} for each i and k: (n, K).
        # """
        # if self.result_ is None:
            # raise ValueError("Model is not fitted.")
        # X = np.asarray(X)

        # n = X.shape[0]
        # K = len(self.components)
        # mus = np.empty((n, K), dtype=float)
        # X_in = np.asarray(X, dtype=float)
        # if X_in.ndim != 2:
            # raise ValueError("X must be a 2D array of shape (n, p).")
        # X_use = self.scaler_.transform(X_in) if (self.standardize_ and self.scaler_ is not None) else X_in
        # for k, comp in enumerate(self.components):
            # mus[:, k] = comp.link.inverse(X_use @ self.result_.betas[k])
        # return mus

    # def predict_mean(self, X: Array) -> Array:
        # """
        # Mixture mean E[Y|X] approximated as sum_k pi_k * mu_k(X).
        # Note: for some families, mu is the location parameter, not necessarily the mean.
        # """
        # if self.result_ is None:
            # raise ValueError("Model is not fitted.")
        # mus = self.predict_component_means(X)
        # return mus @ self.result_.pi

    # def predict_responsibilities(self, y: Array, X: Array) -> Array:
        # """
        # Posterior responsibilities tau_{ik} = P(Z_i=k | y_i, x_i).
        # """
        # if self.result_ is None:
            # raise ValueError("Model is not fitted.")
        # y, X = check_same_n(y, X)
        # X_in = np.asarray(X, dtype=float)
        # X_use = self.scaler_.transform(X_in) if (self.standardize_ and self.scaler_ is not None) else X_in

        # n = y.shape[0]
        # K = len(self.components)
        # log_terms = np.empty((n, K), dtype=float)

        # for k, comp in enumerate(self.components):
            # mu = comp.link.inverse(X_use @ self.result_.betas[k])
            # ll = comp.family.loglik_component(y=y, mu=mu, extra=self.result_.extras[k])
            # # log_terms[:, k] = np.log(self.result_.pi[k]) + ll
            # log_terms[:, k] = np.log(np.clip(self.result_.pi[k], 1e-300, 1.0)) + ll


        # # stable softmax
        # m = np.max(log_terms, axis=1, keepdims=True)
        # ex = np.exp(log_terms - m)
        # tau = ex / np.sum(ex, axis=1, keepdims=True)
        # return tau

    # # ------------------------- diagnostics -------------------------

    # def info_criteria(self) -> Dict[str, float]:
        # """
        # Return AIC/BIC/(ICL) computed from stored result and model structure.
        # """
        # if self.result_ is None:
            # raise ValueError("Model is not fitted.")
        # # already computed in EM, but recompute in a consistent place if needed
        # return {
            # "loglik": float(self.result_.loglik),
            # "aic": float(self.result_.aic),
            # "bic": float(self.result_.bic),
            # "icl": float(self.result_.icl) if self.result_.icl is not None else np.nan,
        # }

    # def summary(self) -> str:
        # if self.result_ is None:
            # return "MixtureGLM (unfitted)"
        # header = self.result_.summary_str()
        # comp_lines = ["Components:"]
        # for k, comp in enumerate(self.components):
            # comp_lines.append(f"  k={k}: {comp.name}")
        # return header + "\n" + "\n".join(comp_lines)

    # def betas_original_scale(self) -> List[Array]:
        # if self.result_ is None:
            # raise ValueError("Model is not fitted.")
        # if not self.standardize_ or self.scaler_ is None:
            # return [b.copy() for b in self.result_.betas]
        # return [self.scaler_.beta_to_original_scale(b) for b in self.result_.betas]




# # Backward-compatible aliases (if older files imported from mixglm.model.mixture_glm)
# # ComponentSpec = ComponentSpec
# # MixtureGLMResult = MixtureGLMResult


# src/mixglm/model/mixture_glm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import special
from mixglm.model.component import ComponentSpec
from mixglm.model.results import MixtureGLMResult
from mixglm.em.pem import fit_pem

from mixglm.utils.checks import check_same_n
from mixglm.utils.standardize import Standardizer

Array = np.ndarray


def _norm_sf_abs(z: Array) -> Array:
    """
    Two-sided p-value helper without requiring scipy.
    p = 2*(1 - Phi(|z|)).
    Uses erf approximation via numpy.
    """
    z = np.asarray(z, dtype=float)
    # Phi(x) = 0.5*(1+erf(x/sqrt(2)))
    Phi = 0.5 * (1.0 + special.erf(np.abs(z) / np.sqrt(2.0)))
    return 2.0 * (1.0 - Phi)


def _zcrit(alpha: float) -> float:
    """
    Approximate z_{1-alpha/2} without scipy.
    Uses a simple binary search on Phi for robustness.
    """
    alpha = float(alpha)
    target = 1.0 - alpha / 2.0

    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        Phi = 0.5 * (1.0 + special.erf(mid / np.sqrt(2.0)))
        if Phi < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _param_names_for_numeric(components: Sequence[ComponentSpec], p: int) -> List[str]:
    """
    Must match the packing order in inference.numeric_se._pack_theta:
      theta = [eta_pi (K-1), betas (K*p), extras (per component, in extra_param_names order)]
    """
    K = len(components)
    names: List[str] = []

    # eta_pi
    for j in range(K - 1):
        names.append(f"eta_pi[{j}]")

    # betas
    for k in range(K):
        for j in range(p):
            names.append(f"beta[{k}][{j}]")

    # extras (transformed names; numeric_se uses transform_extra)
    for k, comp in enumerate(components):
        enames = list(comp.family.extra_param_names)
        for en in enames:
            names.append(f"extra[{k}].{en}_t")

    return names


def _count_params(components: Sequence[ComponentSpec], p: int) -> int:
    K = len(components)
    extra_cnt = sum(len(c.family.extra_param_names) for c in components)
    beta_cnt = 0
    for comp in components:
        if comp.coef_mask is None:
            beta_cnt += p
        else:
            mask = np.asarray(comp.coef_mask, dtype=bool)
            if mask.shape != (p,):
                raise ValueError(f"coef_mask must have shape ({p},); got {mask.shape}.")
            beta_cnt += int(np.sum(mask))
    return int((K - 1) + beta_cnt + extra_cnt)


def _r2(y: Array, yhat: Array) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    return float(1.0 - sse / max(sst, 1e-12))


class MixtureGLM:
    """
    Mixture-GLM model where the response follows a finite mixture distribution and
    each component has its own regression for the location parameter.

    For component k:
        Y_i | Z_i=k ~ f_k( y_i | mu_{ik}, extra_k )
        mu_{ik} = g_k^{-1}( x_i^T beta_k )

    Mixing proportions are constant:
        P(Z_i=k) = pi_k
    """

    def __init__(self, components: Sequence[ComponentSpec]) -> None:
        comps = list(components)
        if len(comps) < 1:
            raise ValueError("MixtureGLM requires at least one component.")
        self.components: List[ComponentSpec] = comps
        self.result_: Optional[MixtureGLMResult] = None
        self.standardize_ = False
        self.scaler_: Standardizer | None = None
        self.offset_used_ = False

    def fit(
        self,
        y: Array,
        X: Array,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        n_starts: int = 5,
        seed: Optional[int] = None,
        init: str = "quantile",
        verbose: bool = False,
        inner_mstep_iter: int = 2,
        min_pi: float = 1e-6,
        compute_icl: bool = True,
        standardize: bool = True,
        offset: Array | None = None,
    ) -> "MixtureGLM":
        y, X = check_same_n(y, X)
        self.standardize_ = bool(standardize)
        offset_use = self._validate_offset(offset, y.shape[0], require_if_fitted=False)
        self.offset_used_ = offset is not None

        X_in = np.asarray(X, dtype=float)
        if self.standardize_:
            self.scaler_ = Standardizer(intercept_col=0).fit(X_in)
            X_fit = self.scaler_.transform(X_in)
        else:
            self.scaler_ = None
            X_fit = X_in

        res = fit_pem(
            y=y,
            X=X_fit,
            components=self.components,
            max_iter=max_iter,
            tol=tol,
            n_starts=n_starts,
            seed=seed,
            init=init,
            verbose=verbose,
            inner_mstep_iter=inner_mstep_iter,
            min_pi=min_pi,
            compute_icl=compute_icl,
            offset=offset_use,
        )
        self.result_ = res
        return self

    # ------------------------- internal: use X on fit-scale -------------------------

    def _X_fit_scale(self, X: Array) -> Array:
        X_in = np.asarray(X, dtype=float)
        if X_in.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n, p).")
        if self.standardize_ and self.scaler_ is not None:
            return self.scaler_.transform(X_in)
        return X_in

    def _validate_offset(
        self,
        offset: Array | None,
        n: int,
        *,
        require_if_fitted: bool = True,
    ) -> Array:
        if offset is None:
            if require_if_fitted and self.offset_used_:
                raise ValueError("This model was fitted with an offset; provide offset for evaluation or prediction.")
            return np.zeros(int(n), dtype=float)
        out = np.asarray(offset, dtype=float).reshape(-1)
        if out.shape != (int(n),):
            raise ValueError(f"offset must have shape ({int(n)},); got {out.shape}.")
        if not np.all(np.isfinite(out)):
            raise ValueError("offset must contain only finite values.")
        return out

    # ------------------------- predictions -------------------------

    def predict_component_means(self, X: Array, *, offset: Array | None = None) -> Array:
        """
        Returns component-wise predictive means E[Y | Z=k, X=x_i] for each
        i and k: (n, K).
        """
        if self.result_ is None:
            raise ValueError("Model is not fitted.")
        X_use = self._X_fit_scale(X)

        n = X_use.shape[0]
        offset_use = self._validate_offset(offset, n)
        K = len(self.components)
        mus = np.empty((n, K), dtype=float)
        for k, comp in enumerate(self.components):
            loc = comp.link.inverse(X_use @ self.result_.betas[k] + offset_use)
            mus[:, k] = comp.family.mean_from_mu(loc, self.result_.extras[k])
        return mus

    def predict_mean(self, X: Array, *, offset: Array | None = None) -> Array:
        """
        Mixture predictive mean E[Y|X] = sum_k pi_k E[Y | Z=k, X].
        """
        if self.result_ is None:
            raise ValueError("Model is not fitted.")
        mus = self.predict_component_means(X, offset=offset)
        return mus @ self.result_.pi

    def predict_responsibilities(
        self,
        y: Array,
        X: Array,
        *,
        offset: Array | None = None,
    ) -> Array:
        """
        Posterior responsibilities tau_{ik} = P(Z_i=k | y_i, x_i).
        """
        if self.result_ is None:
            raise ValueError("Model is not fitted.")
        y, X = check_same_n(y, X)
        X_use = self._X_fit_scale(X)

        n = y.shape[0]
        offset_use = self._validate_offset(offset, n)
        K = len(self.components)
        log_terms = np.empty((n, K), dtype=float)

        for k, comp in enumerate(self.components):
            mu = comp.link.inverse(X_use @ self.result_.betas[k] + offset_use)
            ll = comp.family.loglik_component(y=y, mu=mu, extra=self.result_.extras[k])
            log_terms[:, k] = np.log(np.clip(self.result_.pi[k], 1e-300, 1.0)) + ll

        m = np.max(log_terms, axis=1, keepdims=True)
        ex = np.exp(log_terms - m)
        tau = ex / np.sum(ex, axis=1, keepdims=True)
        return tau

    # ------------------------- diagnostics -------------------------

    def info_criteria(self) -> Dict[str, float]:
        """
        Return AIC/BIC/(ICL) computed from stored result and model structure.
        """
        if self.result_ is None:
            raise ValueError("Model is not fitted.")
        return {
            "loglik": float(self.result_.loglik),
            "aic": float(self.result_.aic),
            "bic": float(self.result_.bic),
            "icl": float(self.result_.icl) if self.result_.icl is not None else np.nan,
        }

    def fit_diagnostics(
        self,
        y: Array,
        X: Array,
        *,
        offset: Array | None = None,
    ) -> Dict[str, float]:
        """
        Quick "lm-like" fit diagnostics based on mixture mean predictions:
          - mse
          - r2
          - adj_r2_like (heuristic; uses param count for df)
        """
        if self.result_ is None:
            raise ValueError("Model is not fitted.")
        y, X = check_same_n(y, X)
        yhat = self.predict_mean(X, offset=offset)
        mse = float(np.mean((np.asarray(y, dtype=float) - yhat) ** 2))
        r2v = _r2(y, yhat)

        n = int(np.asarray(y).shape[0])
        p = int(np.asarray(X).shape[1])
        df_params = _count_params(self.components, p=p)
        # heuristic adjusted R2 (treating param count as df consumption)
        denom = max(n - df_params - 1, 1)
        adj = float(1.0 - (1.0 - r2v) * (n - 1) / denom)

        return {"mse": mse, "r2": float(r2v), "adj_r2_like": adj}

    # ------------------------- inference (numeric SE baseline) -------------------------

    def inference_table(
        self,
        y: Array,
        X: Array,
        *,
        method: str = "numeric",   # "numeric" (baseline) | "louis" (later)
        alpha: float = 0.05,
        numeric_eps: float = 1e-5,
        use_pinv: bool = True,
        rcond: float = 1e-10,
        louis_derivative_method: str = "auto",
        louis_ridge: float = 1e-8,
        offset: Array | None = None,
    ):
        """
        Build a Wald-style coefficient table (estimate, SE, z, p, CI) on the FIT scale.

        Important notes for mixtures:
        - Wald tests are approximate; mixture likelihoods can be ill-conditioned.
        - For penalized fits, these are NOT formally valid; prefer bootstrap.
        """
        if self.result_ is None:
            raise ValueError("Model is not fitted.")
        y, X = check_same_n(y, X)

        X_use = self._X_fit_scale(X)
        y = np.asarray(y)
        offset_use = self._validate_offset(offset, y.shape[0])

        method = str(method).lower()
        if method not in {"numeric", "louis"}:
            raise ValueError("method must be one of: 'numeric', 'louis'.")

        # import lazily so core model has no hard pandas dependency (but you already use it in examples)
        import pandas as pd

        if method == "numeric":
            from mixglm.inference.numeric_se import numeric_hessian_se

            se_res = numeric_hessian_se(
                model=self,
                y=y,
                X=X_use,
                eps=float(numeric_eps),
                use_pinv=bool(use_pinv),
                rcond=float(rcond),
                offset=offset_use,
            )
            theta_hat = np.asarray(se_res.theta_hat, dtype=float)
            se = np.asarray(se_res.se, dtype=float)

            p_dim = int(X_use.shape[1])
            names = _param_names_for_numeric(self.components, p=p_dim)
        else:
            from mixglm.inference.louis import louis_observed_information

            se_res = louis_observed_information(
                y=y,
                X=X_use,
                components=self.components,
                pi=self.result_.pi,
                betas=self.result_.betas,
                extras=self.result_.extras,
                tau=self.result_.responsibilities,
                fd_eps=float(numeric_eps),
                ridge=float(louis_ridge),
                derivative_method=str(louis_derivative_method),
                offset=offset_use,
            )
            theta_hat = np.asarray(se_res.theta_hat, dtype=float)
            if se_res.se is None:
                se = np.full(theta_hat.shape, np.nan, dtype=float)
            else:
                se = np.asarray(se_res.se, dtype=float)
            names = list(se_res.param_names)

        if theta_hat.size != len(names) or se.size != len(names):
            raise RuntimeError(
                f"Mismatch in inference dimensions: theta={theta_hat.size}, se={se.size}, names={len(names)}. "
                "Check pack order vs name builder."
            )

        z = theta_hat / np.clip(se, 1e-15, np.inf)
        pvals = _norm_sf_abs(z)
        zc = _zcrit(alpha)

        ci_lo = theta_hat - zc * se
        ci_hi = theta_hat + zc * se

        df = pd.DataFrame(
            {
                "param": names,
                "estimate": theta_hat,
                "se": se,
                "z": z,
                "p": pvals,
                f"ci{100*(alpha/2):.1f}%": ci_lo,
                f"ci{100*(1-alpha/2):.1f}%": ci_hi,
            }
        )

        return df, se_res

    # ------------------------- summaries -------------------------

    def summary(self) -> str:
        """
        Backwards-compatible summary (no inference).
        Use summary_extended(y, X, ...) if you want SE/CI/p-values.
        """
        if self.result_ is None:
            return "MixtureGLM (unfitted)"
        header = self.result_.summary_str()
        comp_lines = ["Components:"]
        for k, comp in enumerate(self.components):
            comp_lines.append(f"  k={k}: {comp.name}")
        return header + "\n" + "\n".join(comp_lines)

    def summary_extended(
        self,
        y: Array,
        X: Array,
        *,
        method: str = "numeric",
        alpha: float = 0.05,
        numeric_eps: float = 1e-5,
        use_pinv: bool = True,
        rcond: float = 1e-10,
        louis_derivative_method: str = "auto",
        louis_ridge: float = 1e-8,
        max_rows: Optional[int] = None,
        offset: Array | None = None,
    ) -> str:
        """
        Extended "glm-like" summary with:
          - Fit header + components
          - Fit diagnostics (mse, r2, adj_r2_like)
          - Wald table (estimate, SE, z, p, CI)
        """
        if self.result_ is None:
            return "MixtureGLM (unfitted)"

        lines: List[str] = []
        lines.append(self.summary())

        # diagnostics
        diag = self.fit_diagnostics(y, X, offset=offset)
        lines.append("")
        lines.append("Fit diagnostics (mixture mean):")
        lines.append(f"  mse:        {diag['mse']:.6f}")
        lines.append(f"  r2:         {diag['r2']:.6f}")
        lines.append(f"  adj_r2_like:{diag['adj_r2_like']:.6f}")

        # inference
        try:
            tbl, se_res = self.inference_table(
                y=y, X=X,
                method=method,
                alpha=alpha,
                numeric_eps=numeric_eps,
                use_pinv=use_pinv,
                rcond=rcond,
                louis_derivative_method=louis_derivative_method,
                louis_ridge=louis_ridge,
                offset=offset,
            )
            lines.append("")
            lines.append(f"Inference (Wald, method={method}; scale={'standardized' if self.standardize_ else 'original'}):")
            lines.append(f"  SE engine: {se_res.message}")
            if max_rows is not None:
                tbl2 = tbl.head(int(max_rows))
            else:
                tbl2 = tbl
            lines.append(tbl2.to_string(index=False))
            if max_rows is not None and len(tbl) > int(max_rows):
                lines.append(f"... ({len(tbl) - int(max_rows)} more rows)")
        except Exception as e:
            lines.append("")
            lines.append(f"Inference failed: {e}")

        return "\n".join(lines)

    def betas_original_scale(self) -> List[Array]:
        if self.result_ is None:
            raise ValueError("Model is not fitted.")
        if not self.standardize_ or self.scaler_ is None:
            return [b.copy() for b in self.result_.betas]
        return [self.scaler_.beta_to_original_scale(b) for b in self.result_.betas]
