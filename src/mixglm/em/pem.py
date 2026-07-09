# # src/mixglm/em/pem.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import List, Dict, Optional, Tuple, Any, Sequence
# import numpy as np

# from mixglm.model.mixture_glm import ComponentSpec, MixtureGLMResult
# from mixglm.penalties.base import NoPenalty
# from mixglm.families.base import UnivariateFamily

# Array = np.ndarray


# # ----------------------------- helpers: numerics -----------------------------

# def _logsumexp(A: Array, axis: int = 1) -> Array:
    # A = np.asarray(A)
    # m = np.max(A, axis=axis, keepdims=True)
    # return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)


# def _safe_softmax_logweights(logw: Array) -> Array:
    # """Row-wise softmax from log-weights (n,K) -> responsibilities (n,K)."""
    # m = np.max(logw, axis=1, keepdims=True)
    # ex = np.exp(logw - m)
    # return ex / np.sum(ex, axis=1, keepdims=True)


# def _finite_diff_grad(f, x: Array, eps: float = 1e-6) -> Array:
    # """Central finite-difference gradient for scalar function f(x)."""
    # x = np.asarray(x, dtype=float)
    # g = np.zeros_like(x)
    # for j in range(x.size):
        # x1 = x.copy()
        # x2 = x.copy()
        # h = eps * (1.0 + abs(x[j]))
        # x1[j] += h
        # x2[j] -= h
        # g[j] = (f(x1) - f(x2)) / (2.0 * h)
    # return g


# def _weighted_least_squares_beta(X: Array, y_tgt: Array, w: Array, ridge: float = 0.0) -> Array:
    # """
    # Weighted least squares for initialization:
        # argmin_b sum_i w_i (y_tgt_i - x_i^T b)^2 + ridge * ||b||^2
    # """
    # w = np.asarray(w, dtype=float)
    # sw = np.sqrt(np.clip(w, 0.0, None))
    # Xw = X * sw[:, None]
    # yw = y_tgt * sw
    # p = X.shape[1]
    # if ridge > 0:
        # # augment with sqrt(ridge)*I
        # Xw = np.vstack([Xw, np.sqrt(ridge) * np.eye(p)])
        # yw = np.concatenate([yw, np.zeros(p)])
    # b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    # return b


# # ----------------------------- helpers: init strategies -----------------------------

# def _init_tau_quantile(y: Array, K: int) -> Array:
    # n = y.shape[0]
    # qs = np.quantile(y, np.linspace(0, 1, K + 1))
    # # make strict increasing bounds
    # qs[0] -= 1e-12
    # qs[-1] += 1e-12
    # z = np.zeros(n, dtype=int)
    # for k in range(K):
        # mask = (y > qs[k]) & (y <= qs[k + 1])
        # z[mask] = k
    # tau = np.full((n, K), 0.1 / max(K - 1, 1), dtype=float)
    # tau[np.arange(n), z] = 0.9
    # tau /= tau.sum(axis=1, keepdims=True)
    # return tau


# def _init_tau_random(n: int, K: int, rng: np.random.Generator) -> Array:
    # A = rng.gamma(shape=1.0, scale=1.0, size=(n, K))
    # return A / A.sum(axis=1, keepdims=True)


# def _kmeans_1d(y: Array, K: int, rng: np.random.Generator, n_iter: int = 25) -> Tuple[Array, Array]:
    # """Simple 1D k-means: returns (centers, labels)."""
    # y = y.astype(float)
    # n = y.size
    # # init centers by random samples
    # centers = rng.choice(y, size=K, replace=False) if n >= K else np.linspace(y.min(), y.max(), K)
    # labels = np.zeros(n, dtype=int)
    # for _ in range(n_iter):
        # # assign
        # d2 = (y[:, None] - centers[None, :]) ** 2
        # new_labels = np.argmin(d2, axis=1)
        # if np.all(new_labels == labels):
            # break
        # labels = new_labels
        # # update centers
        # for k in range(K):
            # mask = labels == k
            # if np.any(mask):
                # centers[k] = y[mask].mean()
            # else:
                # centers[k] = rng.choice(y)
    # return centers, labels


# def _init_tau_kmeans_y(y: Array, K: int, rng: np.random.Generator) -> Array:
    # _, labels = _kmeans_1d(y, K, rng=rng)
    # n = y.size
    # tau = np.full((n, K), 0.1 / max(K - 1, 1), dtype=float)
    # tau[np.arange(n), labels] = 0.9
    # tau /= tau.sum(axis=1, keepdims=True)
    # return tau


# # ----------------------------- M-step solvers -----------------------------

# def _prox_soft_threshold(v: Array, thresh: float) -> Array:
    # return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)


# def _penalty_prox(beta: Array, penalty, step: float) -> Array:
    # # all penalties implement prox (per our BasePenalty)
    # return penalty.prox(beta, step)


# def _optimize_beta_proxgrad(
    # X: Array,
    # y: Array,
    # tau_k: Array,
    # comp: ComponentSpec,
    # beta0: Array,
    # extra_k: Dict[str, Any],
    # max_iter: int = 200,
    # tol: float = 1e-6,
# ) -> Array:
    # """
    # Proximal gradient on beta for component k:
        # minimize  f(beta) + P(beta),
    # where f(beta) = -sum_i tau_ik log f(y_i; mu_i(beta), extra_k).

    # Gradient of f(beta) is approximated by finite differences (generic, slow but robust).
    # Later we can replace with analytic gradients or autodiff.
    # """
    # beta = beta0.astype(float).copy()
    # penalty = comp.penalty
    # family = comp.family
    # link = comp.link

    # def smooth_obj(b: Array) -> float:
        # mu = link.inverse(X @ b)
        # return family.component_nll(y=y, mu=mu, extra=extra_k, weights=tau_k)

    # # initial step size (backtracking will adapt)
    # step = 1e-2
    # f_old = smooth_obj(beta) + penalty.value(beta)

    # for _ in range(max_iter):
        # g = _finite_diff_grad(smooth_obj, beta)
        # beta_try = _penalty_prox(beta - step * g, penalty, step)

        # f_try = smooth_obj(beta_try) + penalty.value(beta_try)

        # # backtracking if not improved
        # bt = 0
        # while f_try > f_old and bt < 20:
            # step *= 0.5
            # beta_try = _penalty_prox(beta - step * g, penalty, step)
            # f_try = smooth_obj(beta_try) + penalty.value(beta_try)
            # bt += 1

        # if abs(f_old - f_try) / (1.0 + abs(f_old)) < tol:
            # beta = beta_try
            # break

        # beta = beta_try
        # f_old = f_try

    # return beta


# def _optimize_extra_scipy(
    # X: Array,
    # y: Array,
    # tau_k: Array,
    # comp: ComponentSpec,
    # beta_k: Array,
    # extra0: Dict[str, Any],
    # max_iter: int = 200,
# ) -> Dict[str, Any]:
    # """
    # Optimize nuisance parameters for a component using SciPy if available.
    # Falls back to no optimization if SciPy is not installed.

    # We optimize in the family-transformed space.
    # """
    # family = comp.family
    # link = comp.link

    # names = family.extra_param_names
    # if len(names) == 0:
        # return {}

    # try:
        # from scipy.optimize import minimize
    # except Exception:
        # # keep initial values
        # return extra0

    # extra_t0 = family.transform_extra(extra0)
    # x0 = np.array([float(extra_t0[n]) for n in names], dtype=float)

    # bounds_dict = family.bounds_extra()
    # bounds = [bounds_dict.get(n, (None, None)) for n in names]

    # def obj(xvec: Array) -> float:
        # extra_t = {n: float(xvec[j]) for j, n in enumerate(names)}
        # extra = family.inverse_transform_extra(extra_t)
        # family.validate_extra(extra)
        # mu = link.inverse(X @ beta_k)
        # return family.component_nll(y=y, mu=mu, extra=extra, weights=tau_k)

    # res = minimize(obj, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter})
    # xhat = res.x
    # extra_t_hat = {n: float(xhat[j]) for j, n in enumerate(names)}
    # extra_hat = family.inverse_transform_extra(extra_t_hat)
    # family.validate_extra(extra_hat)
    # return extra_hat


# def _mstep_component(
    # X: Array,
    # y: Array,
    # tau_k: Array,
    # comp: ComponentSpec,
    # beta_k: Array,
    # extra_k: Dict[str, Any],
    # inner_iter: int = 2,
# ) -> Tuple[Array, Dict[str, Any]]:
    # """
    # Alternate between beta update and extra update a few times.
    # """
    # # Update beta given extra (prox-grad works for both smooth + non-smooth penalties)
    # beta_new = beta_k
    # extra_new = extra_k

    # for _ in range(inner_iter):
        # beta_new = _optimize_beta_proxgrad(
            # X=X, y=y, tau_k=tau_k, comp=comp, beta0=beta_new, extra_k=extra_new,
            # max_iter=200, tol=1e-6
        # )
        # extra_new = _optimize_extra_scipy(
            # X=X, y=y, tau_k=tau_k, comp=comp, beta_k=beta_new, extra0=extra_new,
            # max_iter=200
        # )
    # return beta_new, extra_new


# # ----------------------------- objective + IC -----------------------------

# def _penalized_loglik(
    # y: Array,
    # X: Array,
    # components: Sequence[ComponentSpec],
    # pi: Array,
    # betas: List[Array],
    # extras: List[Dict[str, Any]],
# ) -> Tuple[float, float]:
    # """
    # Returns (loglik, penalized_loglik) for current parameters.
    # Penalized loglik = loglik - sum_k P_k(beta_k)
    # """
    # n = y.shape[0]
    # K = len(components)
    # log_terms = np.empty((n, K), dtype=float)

    # for k, comp in enumerate(components):
        # mu = comp.link.inverse(X @ betas[k])
        # ll = comp.family.loglik_component(y=y, mu=mu, extra=extras[k])
        # log_terms[:, k] = np.log(pi[k]) + ll

    # ll_obs = float(np.sum(_logsumexp(log_terms, axis=1)))
    # pen = float(sum(comp.penalty.value(betas[k]) for k, comp in enumerate(components)))
    # return ll_obs, ll_obs - pen


# def _param_count(K: int, p: int, components: Sequence[ComponentSpec]) -> int:
    # # (K-1) mixing proportions + sum_k p betas + sum_k nuisance params
    # extra = sum(comp.family.num_extra_params() for comp in components)
    # return (K - 1) + K * p + extra


# def _aic_bic(loglik: float, pcount: int, n: int) -> Tuple[float, float]:
    # aic = -2.0 * loglik + 2.0 * pcount
    # bic = -2.0 * loglik + np.log(n) * pcount
    # return aic, bic


# def _icl(bic: float, tau: Array) -> float:
    # eps = 1e-15
    # ent = np.sum(tau * np.log(np.clip(tau, eps, 1.0)))
    # return float(bic - 2.0 * ent)


# # ----------------------------- main fitting routine -----------------------------

# def fit_pem(
    # *,
    # y: Array,
    # X: Array,
    # components: Sequence[ComponentSpec],
    # max_iter: int = 200,
    # tol: float = 1e-6,
    # n_starts: int = 5,
    # seed: Optional[int] = None,
    # init: str = "quantile",  # "quantile" | "random" | "kmeans_y"
    # verbose: bool = False,
    # inner_mstep_iter: int = 2,
    # min_pi: float = 1e-6,
    # compute_icl: bool = True,
# ) -> MixtureGLMResult:
    # """
    # Fit a (penalized) mixture-of-GLMs model via multi-start penalized EM.

    # This implementation is deliberately generic:
    # - Works with any UnivariateFamily via loglik_component()
    # - Uses proximal gradient with finite-difference gradient for beta updates
    # - Uses SciPy L-BFGS-B for nuisance parameters (if present)

    # Later, we can replace finite-difference gradients with analytic gradients or autodiff.
    # """
    # y = np.asarray(y)
    # X = np.asarray(X)
    # n, p = X.shape
    # K = len(components)

    # rng = np.random.default_rng(seed)

    # best: Optional[MixtureGLMResult] = None
    # best_ll = -np.inf

    # for s in range(n_starts):
        # # ----- initialization -----
        # if init == "quantile":
            # tau = _init_tau_quantile(y, K)
        # elif init == "random":
            # tau = _init_tau_random(n, K, rng)
        # elif init == "kmeans_y":
            # tau = _init_tau_kmeans_y(y, K, rng)
        # else:
            # raise ValueError("init must be one of: 'quantile', 'random', 'kmeans_y'.")

        # pi = np.clip(tau.mean(axis=0), min_pi, 1.0)
        # pi = pi / pi.sum()

        # betas: List[Array] = []
        # extras: List[Dict[str, Any]] = []

        # # initialize component parameters
        # for k, comp in enumerate(components):
            # w = tau[:, k]
            # # crude target for initialization depending on link
            # if comp.link.name.lower() == "identity":
                # y_tgt = y.astype(float)
            # elif comp.link.name.lower() == "log":
                # y_tgt = np.log(np.clip(y.astype(float), 1e-8, None))
            # else:
                # # fallback: zeros
                # y_tgt = np.zeros_like(y, dtype=float)

            # ridge0 = comp.penalty.lam if comp.penalty.name.lower().startswith("ridge") else 1e-8
            # beta0 = _weighted_least_squares_beta(X, y_tgt, w=w, ridge=ridge0)
            # betas.append(beta0)

            # extra0 = comp.family.initialize_extra(y=y, tau_k=w)
            # comp.family.validate_extra(extra0) if comp.family.num_extra_params() > 0 else None
            # extras.append(extra0)

        # history = {"loglik": [], "obj": []}
        # converged = False

        # # ----- EM loop -----
        # ll_obs, ll_pen = _penalized_loglik(y, X, components, pi, betas, extras)
        # history["loglik"].append(ll_obs)
        # history["obj"].append(ll_pen)

        # for t in range(max_iter):
            # # E-step: compute responsibilities
            # log_terms = np.empty((n, K), dtype=float)
            # for k, comp in enumerate(components):
                # mu = comp.link.inverse(X @ betas[k])
                # ll = comp.family.loglik_component(y=y, mu=mu, extra=extras[k])
                # log_terms[:, k] = np.log(pi[k]) + ll

            # tau = _safe_softmax_logweights(log_terms)

            # # M-step: update pi
            # pi = np.clip(tau.mean(axis=0), min_pi, 1.0)
            # pi = pi / pi.sum()

            # # M-step: update each component (beta_k, extra_k)
            # for k, comp in enumerate(components):
                # beta_k, extra_k = _mstep_component(
                    # X=X, y=y, tau_k=tau[:, k], comp=comp,
                    # beta_k=betas[k], extra_k=extras[k],
                    # inner_iter=inner_mstep_iter,
                # )
                # betas[k] = beta_k
                # extras[k] = extra_k

            # # monitor objective
            # ll_obs_new, ll_pen_new = _penalized_loglik(y, X, components, pi, betas, extras)
            # history["loglik"].append(ll_obs_new)
            # history["obj"].append(ll_pen_new)

            # rel = abs(ll_pen_new - ll_pen) / (1.0 + abs(ll_pen))
            # if verbose:
                # print(f"[start {s+1}/{n_starts}] iter {t+1:03d} | loglik={ll_obs_new:.6f} | obj={ll_pen_new:.6f} | rel={rel:.3e}")

            # if rel < tol:
                # converged = True
                # ll_obs, ll_pen = ll_obs_new, ll_pen_new
                # break

            # ll_obs, ll_pen = ll_obs_new, ll_pen_new

        # # compute IC on unpenalized loglik (standard practice)
        # pcount = _param_count(K=K, p=p, components=components)
        # aic, bic = _aic_bic(loglik=ll_obs, pcount=pcount, n=n)
        # icl_val = _icl(bic=bic, tau=tau) if compute_icl else None

        # res = MixtureGLMResult(
            # converged=converged,
            # n_iter=len(history["obj"]) - 1,
            # loglik=float(ll_obs),
            # bic=float(bic),
            # aic=float(aic),
            # icl=float(icl_val) if icl_val is not None else None,
            # pi=pi.copy(),
            # betas=[b.copy() for b in betas],
            # extras=[dict(ex) for ex in extras],
            # responsibilities=tau.copy(),
            # history=history,
        # )

        # if res.loglik > best_ll:
            # best_ll = res.loglik
            # best = res

    # if best is None:
        # raise RuntimeError("fit_pem failed: no successful start.")

    # return best

# src/mixglm/em/pem.py
from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any, Sequence
import numpy as np

from mixglm.model.mixture_glm import ComponentSpec, MixtureGLMResult
from mixglm.em.init import init_parameters
from mixglm.em.responsibilities import compute_responsibilities
from mixglm.em.stopping import StopState
from mixglm.utils.numerics import normalize_simplex

from mixglm.em.mstep import mstep_component

Array = np.ndarray


# ----------------------------- objective evaluation -----------------------------

def _penalized_objective(
    y: Array,
    X: Array,
    components: Sequence[ComponentSpec],
    pi: Array,
    betas: Sequence[Array],
    extras: Sequence[Dict[str, Any]],
    offset: Array | None = None,
) -> Tuple[float, float, Array]:
    """
    Returns:
      loglik_obs, obj_penalized (= loglik_obs - sum_k P_k(beta_k)), responsibilities tau
    """
    tau, ll_obs = compute_responsibilities(
        y=y, X=X, components=components, pi=pi, betas=betas, extras=extras,
        offset=offset,
    )
    # pen = float(sum(comp.penalty.value(betas[k]) for k, comp in enumerate(components)))
    pen = float(sum(comp.penalty.value(np.asarray(betas[k], dtype=float)[1:]) for k, comp in enumerate(components)))

    return float(ll_obs), float(ll_obs - pen), tau


# ----------------------------- main fitting routine -----------------------------

def fit_pem(
    *,
    y: Array,
    X: Array,
    components: Sequence[ComponentSpec],
    max_iter: int = 200,
    tol: float = 1e-6,
    n_starts: int = 5,
    seed: Optional[int] = None,
    init: str = "quantile",  # "quantile" | "random" | "kmeans_y"
    verbose: bool = False,
    inner_mstep_iter: int = 2,
    min_pi: float = 1e-6,
    compute_icl: bool = True,
    offset: Array | None = None,
) -> MixtureGLMResult:
    """
    Multi-start penalized EM (GEM) for mixture-of-GLMs.

    E-step:
      tau_{ik} = pi_k f_k(y_i | x_i) / sum_j pi_j f_j(y_i | x_i)

    M-step (GEM):
      pi_k <- mean_i tau_{ik}
      beta_k, extra_k <- alternating prox-grad + scipy optimize

    Returns the best start by observed-data log-likelihood.
    """
    y = np.asarray(y)
    X = np.asarray(X)
    n, p = X.shape
    K = len(components)
    offset_use = np.zeros(n, dtype=float) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
    if offset_use.shape != (n,):
        raise ValueError(f"offset must have shape ({n},); got {offset_use.shape}.")
    if not np.all(np.isfinite(offset_use)):
        raise ValueError("offset must contain only finite values.")

    rng = np.random.default_rng(seed)

    best: Optional[MixtureGLMResult] = None
    best_ll = -np.inf

    for s in range(n_starts):
        # ----- initialization -----
        init_state = init_parameters(
            y=y, X=X, components=components, init=init, rng=rng,
            min_pi=min_pi, offset=offset_use,
        )
        pi = init_state.pi.copy()
        betas = [b.copy() for b in init_state.betas]
        extras = [dict(ex) for ex in init_state.extras]

        history = {"loglik": [], "obj": []}

        # initial objective
        ll_obs, obj_pen, tau = _penalized_objective(
            y, X, components, pi, betas, extras, offset=offset_use
        )
        history["loglik"].append(ll_obs)
        history["obj"].append(obj_pen)

        stopper = StopState(tol=tol, max_iter=max_iter, min_iter=1)
        stopper.last_obj = obj_pen

        converged = False

        for t in range(max_iter):
            # M-step: update pi
            pi = normalize_simplex(np.clip(tau.mean(axis=0), min_pi, 1.0), min_val=min_pi)

            # M-step: component-wise updates
            for k, comp in enumerate(components):
                beta_k, extra_k = mstep_component(
                    X=X, y=y, tau_k=tau[:, k], comp=comp,
                    beta_k=betas[k], extra_k=extras[k],
                    offset=offset_use,
                    inner_iter=inner_mstep_iter,
                )
                betas[k] = beta_k
                extras[k] = extra_k

            # E-step: update responsibilities and objective
            ll_obs, obj_pen, tau = _penalized_objective(
                y, X, components, pi, betas, extras, offset=offset_use
            )
            history["loglik"].append(ll_obs)
            history["obj"].append(obj_pen)

            rel = abs(history["obj"][-1] - history["obj"][-2]) / (1.0 + abs(history["obj"][-2]))
            if verbose:
                print(
                    f"[start {s+1}/{n_starts}] iter {t+1:03d} "
                    f"| loglik={ll_obs:.6f} | obj={obj_pen:.6f} | rel={rel:.3e}"
                )

            if stopper.update(obj_pen):
                converged = stopper.converged
                break

        # ----- IC -----
        extra_cnt = sum(comp.family.num_extra_params() for comp in components)

        beta_pcount = 0
        for k, comp in enumerate(components):
            if comp.penalty.name != "none":
                beta_pcount += int(np.sum(np.abs(betas[k]) > 1e-5))
            elif comp.coef_mask is not None:
                mask = np.asarray(comp.coef_mask, dtype=bool)
                if mask.shape != (p,):
                    raise ValueError(f"coef_mask must have shape ({p},); got {mask.shape}.")
                beta_pcount += int(np.sum(mask))
            else:
                beta_pcount += p

        pcount = (K - 1) + beta_pcount + extra_cnt

        aic = -2.0 * ll_obs + 2.0 * pcount
        bic = -2.0 * ll_obs + float(np.log(n)) * pcount

        icl_val = None
        if compute_icl:
            eps = 1e-15
            ent = float(np.sum(tau * np.log(np.clip(tau, eps, 1.0))))
            icl_val = float(bic - 2.0 * ent)

        res = MixtureGLMResult(
            converged=converged,
            n_iter=len(history["obj"]) - 1,
            loglik=float(ll_obs),
            bic=float(bic),
            aic=float(aic),
            icl=float(icl_val) if icl_val is not None else None,
            pi=pi.copy(),
            betas=[b.copy() for b in betas],
            extras=[dict(ex) for ex in extras],
            responsibilities=tau.copy(),
            history=history,
        )

        if res.loglik > best_ll:
            best_ll = res.loglik
            best = res

    if best is None:
        raise RuntimeError("fit_pem failed: no successful start.")

    return best
