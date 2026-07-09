# # src/mixglm/inference/louis.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Sequence, Tuple

# import numpy as np

# Array = np.ndarray


# # ---------------------------------------------------------------------
# # Helpers: softmax (with K-1 free params, last fixed to 0 for identifiability)
# # ---------------------------------------------------------------------
# def _softmax_with_last_zero(eta_free: Array) -> Array:
    # """
    # eta_free: (K-1,)
    # returns pi: (K,), where eta = [eta_free, 0] and pi = softmax(eta).
    # """
    # eta_free = np.asarray(eta_free, dtype=float)
    # eta = np.concatenate([eta_free, np.array([0.0])])
    # m = float(np.max(eta))
    # ex = np.exp(eta - m)
    # return ex / np.sum(ex)


# def _dlogpi_deta(pi: Array, k: int) -> Array:
    # """
    # Gradient of log(pi_k) w.r.t eta_free (K-1).
    # eta is [eta_free, 0], pi = softmax(eta).

    # For j in 0..K-2:
      # d log(pi_k) / d eta_j = 1_{k=j} - pi_j     if k < K-1
                             # - pi_j             if k == K-1
    # """
    # K = pi.size
    # g = -pi[: K - 1].copy()
    # if k < K - 1:
        # g[k] += 1.0
    # return g


# def _d2logpi_deta2(pi: Array, k: int) -> Array:
    # """
    # Hessian of log(pi_k) w.r.t eta_free (K-1, K-1), analytic.

    # For softmax with last fixed:
      # d pi_j / d eta_l = pi_j (1_{j=l} - pi_l) for j,l in 0..K-2
      # d log(pi_k)/d eta = e_k - pi_free    (or -pi_free for k=last)
      # Hessian = - d(pi_free)/d eta  (because derivative of -pi_free)
      # => H = -J where J_{j,l} = d pi_j / d eta_l
      # with j,l in 0..K-2

    # This is independent of k (true for softmax log-prob; the second derivative
    # depends only on pi, not on which log(pi_k) you take) under this parameterization.
    # """
    # p = pi[: pi.size - 1]
    # # Jacobian J of p w.r.t eta_free
    # # J = diag(p) - p p^T
    # J = np.diag(p) - np.outer(p, p)
    # return -J


# # ---------------------------------------------------------------------
# # Numeric differentiation (central differences)
# # ---------------------------------------------------------------------
# def _central_grad(fun, x: Array, eps: float) -> Array:
    # x = np.asarray(x, dtype=float)
    # g = np.zeros_like(x, dtype=float)
    # for j in range(x.size):
        # h = eps * max(1.0, abs(x[j]))
        # xp = x.copy()
        # xm = x.copy()
        # xp[j] += h
        # xm[j] -= h
        # g[j] = (fun(xp) - fun(xm)) / (2.0 * h)
    # return g


# def _central_hess(fun, x: Array, eps: float, diag_only: bool = False) -> Array:
    # """
    # Central-difference Hessian. If diag_only=True, fill only diagonal.
    # """
    # x = np.asarray(x, dtype=float)
    # d = x.size
    # H = np.zeros((d, d), dtype=float)

    # f0 = fun(x)

    # # diagonal
    # for j in range(d):
        # h = eps * max(1.0, abs(x[j]))
        # xp = x.copy()
        # xm = x.copy()
        # xp[j] += h
        # xm[j] -= h
        # fp = fun(xp)
        # fm = fun(xm)
        # H[j, j] = (fp - 2.0 * f0 + fm) / (h * h)

    # if diag_only:
        # return H

    # # off-diagonal
    # for j in range(d):
        # hj = eps * max(1.0, abs(x[j]))
        # for l in range(j + 1, d):
            # hl = eps * max(1.0, abs(x[l]))
            # xpp = x.copy()
            # xpm = x.copy()
            # xmp = x.copy()
            # xmm = x.copy()
            # xpp[j] += hj
            # xpp[l] += hl
            # xpm[j] += hj
            # xpm[l] -= hl
            # xmp[j] -= hj
            # xmp[l] += hl
            # xmm[j] -= hj
            # xmm[l] -= hl
            # fpp = fun(xpp)
            # fpm = fun(xpm)
            # fmp = fun(xmp)
            # fmm = fun(xmm)
            # val = (fpp - fpm - fmp + fmm) / (4.0 * hj * hl)
            # H[j, l] = val
            # H[l, j] = val

    # return H


# # ---------------------------------------------------------------------
# # Parameter packing (eta_free, betas, extra_transformed)
# # ---------------------------------------------------------------------
# @dataclass(frozen=True)
# class LouisParamIndex:
    # K: int
    # p: int
    # eta_dim: int
    # beta_slices: List[slice]
    # extra_slices: List[slice]
    # names: List[str]


# def _build_index(components, p: int, include_eta: bool = True) -> LouisParamIndex:
    # K = len(components)
    # eta_dim = (K - 1) if include_eta and K > 1 else 0

    # names: List[str] = []
    # if eta_dim > 0:
        # for j in range(eta_dim):
            # names.append(f"eta_pi[{j}]")

    # beta_slices: List[slice] = []
    # extra_slices: List[slice] = []

    # off = eta_dim

    # for k, comp in enumerate(components):
        # s_beta = slice(off, off + p)
        # beta_slices.append(s_beta)
        # for j in range(p):
            # names.append(f"beta[{k}][{j}]")
        # off += p

        # extra_names = list(comp.family.extra_param_names)
        # s_extra = slice(off, off + len(extra_names))
        # extra_slices.append(s_extra)
        # for en in extra_names:
            # names.append(f"extra[{k}].{en}_t")  # transformed space
        # off += len(extra_names)

    # return LouisParamIndex(
        # K=K, p=p, eta_dim=eta_dim,
        # beta_slices=beta_slices, extra_slices=extra_slices,
        # names=names
    # )


# def _pack_theta(
    # *,
    # pi: Array,
    # betas: Sequence[Array],
    # extras: Sequence[Dict[str, Any]],
    # components,
    # idx: LouisParamIndex,
# ) -> Array:
    # K = idx.K
    # p = idx.p
    # theta = np.zeros(len(idx.names), dtype=float)

    # # eta_free: solve from pi up to additive constant, last fixed 0
    # if idx.eta_dim > 0:
        # pi = np.asarray(pi, dtype=float)
        # # eta_j - eta_last = log(pi_j/pi_last)
        # eta_free = np.log(np.clip(pi[: K - 1], 1e-15, 1.0)) - np.log(np.clip(pi[K - 1], 1e-15, 1.0))
        # theta[: idx.eta_dim] = eta_free

    # # betas + transformed extras
    # for k in range(K):
        # theta[idx.beta_slices[k]] = np.asarray(betas[k], dtype=float).reshape(-1)[:p]
        # extra_t = components[k].family.transform_extra(extras[k])
        # names = list(components[k].family.extra_param_names)
        # if len(names) > 0:
            # theta[idx.extra_slices[k]] = np.array([float(extra_t[n]) for n in names], dtype=float)

    # return theta


# def _unpack_theta(
    # theta: Array,
    # *,
    # components,
    # idx: LouisParamIndex,
# ) -> Tuple[Array, List[Array], List[Dict[str, Any]]]:
    # theta = np.asarray(theta, dtype=float)
    # K = idx.K
    # p = idx.p

    # # pi from eta_free
    # if idx.eta_dim > 0:
        # eta_free = theta[: idx.eta_dim]
        # pi = _softmax_with_last_zero(eta_free)
    # else:
        # pi = np.ones(K, dtype=float) / float(K)

    # betas: List[Array] = []
    # extras: List[Dict[str, Any]] = []

    # for k in range(K):
        # b = theta[idx.beta_slices[k]].copy()
        # betas.append(b)

        # names = list(components[k].family.extra_param_names)
        # if len(names) == 0:
            # extras.append({})
        # else:
            # extra_t = {n: float(theta[idx.extra_slices[k]][j]) for j, n in enumerate(names)}
            # extra = components[k].family.inverse_transform_extra(extra_t)
            # components[k].family.validate_extra(extra)
            # extras.append(extra)

    # return pi, betas, extras


# # ---------------------------------------------------------------------
# # Louis observed information
# # ---------------------------------------------------------------------
# @dataclass
# class LouisResult:
    # info: Array
    # cov: Optional[Array]
    # se: Optional[Array]
    # param_names: List[str]


# def louis_observed_information(
    # *,
    # y: Array,
    # X: Array,
    # components,
    # pi: Array,
    # betas: Sequence[Array],
    # extras: Sequence[Dict[str, Any]],
    # tau: Array,
    # fd_eps: float = 1e-5,
    # diag_hessian: bool = False,
    # ridge: float = 1e-8,
# ) -> LouisResult:
    # """
    # Compute observed information via Louis' identity.

    # Parameters are in an unconstrained space:
      # - mixing proportions: eta_free (K-1), last eta fixed to 0
      # - betas: each component beta_k (p)
      # - extras: each component extra params in family.transform_extra space

    # Returns:
      # info (observed information),
      # cov (inverse if successful),
      # se  (sqrt(diag(cov)) if successful),
      # param_names
    # """
    # y = np.asarray(y)
    # X = np.asarray(X, dtype=float)
    # tau = np.asarray(tau, dtype=float)
    # n, p = X.shape
    # K = len(components)
    # if tau.shape != (n, K):
        # raise ValueError(f"tau must have shape (n,K)=({n},{K}); got {tau.shape}.")

    # idx = _build_index(components, p=p, include_eta=True)
    # theta0 = _pack_theta(pi=pi, betas=betas, extras=extras, components=components, idx=idx)

    # d = theta0.size
    # I = np.zeros((d, d), dtype=float)

    # # Convenience: compute per-(i,k) loglik contribution for component k only (no log pi)
    # def logf_i_k(i: int, k: int, beta_k: Array, extra_k_t: Array) -> float:
        # comp = components[k]
        # # reconstruct extra from transformed vector
        # names = list(comp.family.extra_param_names)
        # if len(names) == 0:
            # extra = {}
        # else:
            # extra_t = {n: float(extra_k_t[j]) for j, n in enumerate(names)}
            # extra = comp.family.inverse_transform_extra(extra_t)
            # comp.family.validate_extra(extra)

        # mu = comp.link.inverse(X[i] @ beta_k)
        # ll = comp.family.loglik_component(y=y[i:i+1], mu=np.array([mu]), extra=extra)
        # return float(ll[0])

    # # Loop over i, build s_i^k and H_i^k
    # for i in range(n):
        # # Precompute eta/pi parts from current theta0 (use pi from unpack)
        # pi_curr, betas_curr, extras_curr = _unpack_theta(theta0, components=components, idx=idx)

        # # eta score & hessian terms depend only on pi
        # H_eta = None
        # if idx.eta_dim > 0:
            # # Hessian of log(pi_k) w.r.t eta_free is the same for all k: -J(pi_free)
            # # So we compute it once.
            # # (We keep it in case you later choose to weight by tau_ik.)
            # H_eta = _d2logpi_deta2(pi_curr, k=0)  # (K-1,K-1)

        # s_all_k = np.zeros((K, d), dtype=float)
        # Hexp = np.zeros((d, d), dtype=float)

        # for k in range(K):
            # # score vector for complete-data contribution if z_i=k
            # s = np.zeros(d, dtype=float)

            # # ---- eta part: grad log(pi_k) ----
            # if idx.eta_dim > 0:
                # s[: idx.eta_dim] = _dlogpi_deta(pi_curr, k=k)

            # # ---- beta + extra part: numeric on component log density ----
            # b0 = betas_curr[k].copy()
            # comp = components[k]
            # names = list(comp.family.extra_param_names)
            # if len(names) == 0:
                # e0 = np.zeros(0, dtype=float)
            # else:
                # et = comp.family.transform_extra(extras_curr[k])
                # e0 = np.array([float(et[n]) for n in names], dtype=float)

            # # gradient wrt beta_k
            # def f_beta(bvec):
                # return logf_i_k(i, k, beta_k=bvec, extra_k_t=e0)

            # g_beta = _central_grad(f_beta, b0, fd_eps)
            # s[idx.beta_slices[k]] = g_beta

            # # gradient wrt extra_t
            # if e0.size > 0:
                # def f_extra(evec):
                    # return logf_i_k(i, k, beta_k=b0, extra_k_t=evec)

                # g_extra = _central_grad(f_extra, e0, fd_eps)
                # s[idx.extra_slices[k]] = g_extra

            # s_all_k[k] = s

            # # ---- expected negative Hessian term (block diagonal) ----
            # w = float(tau[i, k])

            # # eta-eta block
            # if idx.eta_dim > 0 and H_eta is not None:
                # Hexp[: idx.eta_dim, : idx.eta_dim] += w * H_eta

            # # beta-beta block (numeric)
            # Hb = _central_hess(f_beta, b0, fd_eps, diag_only=diag_hessian)
            # Hexp[idx.beta_slices[k], idx.beta_slices[k]] += w * Hb

            # # extra-extra block (numeric)
            # if e0.size > 0:
                # He = _central_hess(f_extra, e0, fd_eps, diag_only=diag_hessian)
                # Hexp[idx.extra_slices[k], idx.extra_slices[k]] += w * He

            # # Cross blocks (eta-beta, eta-extra, beta-extra) are zero for the complete loglik
            # # because log(pi_k) and log f_k are additive and separate in parameters.

        # # Louis identity:
        # # I_obs += - E[H_c | y]  - Var[s_c | y]
        # # Here Hexp is E[H_c | y] (sum_k tau_ik H_i^k)
        # # Var term: sum_k tau_ik (s_k - sbar)(s_k - sbar)^T
        # sbar = np.sum(tau[i, :, None] * s_all_k, axis=0)
        # V = np.zeros((d, d), dtype=float)
        # for k in range(K):
            # dk = (s_all_k[k] - sbar).reshape(-1, 1)
            # V += float(tau[i, k]) * (dk @ dk.T)

        # I += (-Hexp) - V

    # # Symmetrize + small ridge
    # I = 0.5 * (I + I.T)
    # I = I + ridge * np.eye(I.shape[0])

    # # Attempt inversion
    # cov = None
    # se = None
    # try:
        # cov = np.linalg.inv(I)
        # cov = 0.5 * (cov + cov.T)
        # se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    # except Exception:
        # cov = None
        # se = None

    # return LouisResult(info=I, cov=cov, se=se, param_names=idx.names)



# src/mixglm/inference/louis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mixglm.inference.analytic_blocks import component_derivatives

Array = np.ndarray


# ---------------------------------------------------------------------
# Helpers: softmax (with K-1 free params, last fixed to 0 for identifiability)
# ---------------------------------------------------------------------
def _softmax_with_last_zero(eta_free: Array) -> Array:
    eta_free = np.asarray(eta_free, dtype=float)
    eta = np.concatenate([eta_free, np.array([0.0])])
    m = float(np.max(eta))
    ex = np.exp(eta - m)
    return ex / np.sum(ex)


def _dlogpi_deta(pi: Array, k: int) -> Array:
    K = pi.size
    g = -pi[: K - 1].copy()
    if k < K - 1:
        g[k] += 1.0
    return g


def _d2logpi_deta2(pi: Array) -> Array:
    p = pi[: pi.size - 1]
    J = np.diag(p) - np.outer(p, p)   # dp/deta
    return -J                         # d2 log(pi_k)/deta2 for any k under this parametrization


# ---------------------------------------------------------------------
# Numeric differentiation (central differences)
# ---------------------------------------------------------------------
def _central_grad(fun, x: Array, eps: float) -> Array:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x, dtype=float)
    for j in range(x.size):
        h = eps * max(1.0, abs(x[j]))
        xp = x.copy(); xm = x.copy()
        xp[j] += h; xm[j] -= h
        g[j] = (fun(xp) - fun(xm)) / (2.0 * h)
    return g


def _central_hess(fun, x: Array, eps: float, diag_only: bool = False) -> Array:
    x = np.asarray(x, dtype=float)
    d = x.size
    H = np.zeros((d, d), dtype=float)
    f0 = fun(x)

    for j in range(d):
        hj = eps * max(1.0, abs(x[j]))
        xp = x.copy(); xm = x.copy()
        xp[j] += hj; xm[j] -= hj
        fp = fun(xp); fm = fun(xm)
        H[j, j] = (fp - 2.0 * f0 + fm) / (hj * hj)

    if diag_only:
        return H

    for j in range(d):
        hj = eps * max(1.0, abs(x[j]))
        for l in range(j + 1, d):
            hl = eps * max(1.0, abs(x[l]))
            xpp = x.copy(); xpm = x.copy(); xmp = x.copy(); xmm = x.copy()
            xpp[j] += hj; xpp[l] += hl
            xpm[j] += hj; xpm[l] -= hl
            xmp[j] -= hj; xmp[l] += hl
            xmm[j] -= hj; xmm[l] -= hl
            fpp = fun(xpp); fpm = fun(xpm); fmp = fun(xmp); fmm = fun(xmm)
            val = (fpp - fpm - fmp + fmm) / (4.0 * hj * hl)
            H[j, l] = val
            H[l, j] = val

    return H


# ---------------------------------------------------------------------
# Parameter packing (eta_free, betas, extra_transformed)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class LouisParamIndex:
    K: int
    p: int
    eta_dim: int
    beta_slices: List[slice]
    extra_slices: List[slice]
    names: List[str]


def _build_index(components, p: int) -> LouisParamIndex:
    K = len(components)
    eta_dim = (K - 1) if K > 1 else 0

    names: List[str] = []
    for j in range(eta_dim):
        names.append(f"eta_pi[{j}]")

    beta_slices: List[slice] = []
    extra_slices: List[slice] = []

    off = eta_dim
    for k, comp in enumerate(components):
        s_beta = slice(off, off + p)
        beta_slices.append(s_beta)
        for j in range(p):
            names.append(f"beta[{k}][{j}]")
        off += p

        extra_names = list(comp.family.extra_param_names)
        s_extra = slice(off, off + len(extra_names))
        extra_slices.append(s_extra)
        for en in extra_names:
            names.append(f"extra[{k}].{en}_t")
        off += len(extra_names)

    return LouisParamIndex(K=K, p=p, eta_dim=eta_dim,
                          beta_slices=beta_slices, extra_slices=extra_slices,
                          names=names)


def _pack_theta(*, pi: Array, betas: Sequence[Array], extras: Sequence[Dict[str, Any]], components, idx: LouisParamIndex) -> Array:
    K, p = idx.K, idx.p
    theta = np.zeros(len(idx.names), dtype=float)

    if idx.eta_dim > 0:
        pi = np.asarray(pi, dtype=float)
        eta_free = np.log(np.clip(pi[: K - 1], 1e-15, 1.0)) - np.log(np.clip(pi[K - 1], 1e-15, 1.0))
        theta[: idx.eta_dim] = eta_free

    for k in range(K):
        theta[idx.beta_slices[k]] = np.asarray(betas[k], dtype=float).reshape(-1)[:p]
        extra_t = components[k].family.transform_extra(extras[k])
        names = list(components[k].family.extra_param_names)
        if len(names) > 0:
            theta[idx.extra_slices[k]] = np.array([float(extra_t[n]) for n in names], dtype=float)

    return theta


def _unpack_theta(theta: Array, *, components, idx: LouisParamIndex) -> Tuple[Array, List[Array], List[Dict[str, Any]]]:
    theta = np.asarray(theta, dtype=float)
    K, p = idx.K, idx.p

    if K == 1:
        pi = np.array([1.0], dtype=float)
    else:
        eta_free = theta[: idx.eta_dim]
        pi = _softmax_with_last_zero(eta_free)

    betas: List[Array] = []
    extras: List[Dict[str, Any]] = []

    for k in range(K):
        betas.append(theta[idx.beta_slices[k]].copy())

        names = list(components[k].family.extra_param_names)
        if len(names) == 0:
            extras.append({})
        else:
            extra_t = {n: float(theta[idx.extra_slices[k]][j]) for j, n in enumerate(names)}
            extra = components[k].family.inverse_transform_extra(extra_t)
            components[k].family.validate_extra(extra)
            extras.append(extra)

    return pi, betas, extras


# ---------------------------------------------------------------------
# Louis observed information
# ---------------------------------------------------------------------
@dataclass
class LouisResult:
    theta_hat: Array
    info: Array
    cov: Optional[Array]
    se: Optional[Array]
    param_names: List[str]
    derivative_sources: Optional[Dict[str, int]] = None


def louis_observed_information(
    *,
    y: Array,
    X: Array,
    components,
    pi: Array,
    betas: Sequence[Array],
    extras: Sequence[Dict[str, Any]],
    tau: Array,
    fd_eps: float = 1e-5,
    diag_hessian: bool = False,
    ridge: float = 1e-8,
    derivative_method: str = "auto",
    offset: Array | None = None,
) -> LouisResult:
    y = np.asarray(y)
    X = np.asarray(X, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n, p = X.shape
    K = len(components)
    offset_use = np.zeros(n, dtype=float) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
    if offset_use.shape != (n,):
        raise ValueError(f"offset must have shape ({n},); got {offset_use.shape}.")
    if not np.all(np.isfinite(offset_use)):
        raise ValueError("offset must contain only finite values.")
    if tau.shape != (n, K):
        raise ValueError(f"tau must have shape (n,K)=({n},{K}); got {tau.shape}.")

    idx = _build_index(components, p=p)
    theta0 = _pack_theta(pi=pi, betas=betas, extras=extras, components=components, idx=idx)

    # unpack ONCE at theta0
    pi0, betas0, extras0 = _unpack_theta(theta0, components=components, idx=idx)
    H_eta = _d2logpi_deta2(pi0) if idx.eta_dim > 0 else None

    d = theta0.size
    I = np.zeros((d, d), dtype=float)
    derivative_sources: Dict[str, int] = {}

    # main loop
    for i in range(n):
        s_all_k = np.zeros((K, d), dtype=float)
        Hexp = np.zeros((d, d), dtype=float)

        for k in range(K):
            s = np.zeros(d, dtype=float)

            # eta part
            if idx.eta_dim > 0:
                s[: idx.eta_dim] = _dlogpi_deta(pi0, k=k)

            b0 = betas0[k].copy()
            comp = components[k]
            names = list(comp.family.extra_param_names)
            x_i = X[i]
            eta_i = float(x_i @ b0 + offset_use[i])
            deriv = component_derivatives(
                y=float(y[i]),
                eta=eta_i,
                family=comp.family,
                link=comp.link,
                extra=extras0[k],
                method=derivative_method,
                fd_eps=fd_eps,
            )
            derivative_sources[deriv.source] = derivative_sources.get(deriv.source, 0) + 1

            s[idx.beta_slices[k]] = x_i * deriv.score_eta

            if len(names) > 0:
                if deriv.score_extra.size != len(names):
                    raise RuntimeError(
                        f"Derivative block for {comp.family.name} returned {deriv.score_extra.size} "
                        f"extra scores, expected {len(names)}."
                    )
                s[idx.extra_slices[k]] = deriv.score_extra

            s_all_k[k] = s

            w = float(tau[i, k])

            if idx.eta_dim > 0 and H_eta is not None:
                Hexp[: idx.eta_dim, : idx.eta_dim] += w * H_eta

            Hb = deriv.hess_eta_eta * np.outer(x_i, x_i)
            if diag_hessian:
                Hb = np.diag(np.diag(Hb))
            Hexp[idx.beta_slices[k], idx.beta_slices[k]] += w * Hb

            if len(names) > 0:
                He = np.asarray(deriv.hess_extra_extra, dtype=float)
                Hbe = np.outer(x_i, np.asarray(deriv.hess_eta_extra, dtype=float))
                if He.shape != (len(names), len(names)):
                    raise RuntimeError(
                        f"Derivative block for {comp.family.name} returned hess_extra shape {He.shape}, "
                        f"expected {(len(names), len(names))}."
                    )
                if Hbe.shape != (p, len(names)):
                    raise RuntimeError(
                        f"Derivative block for {comp.family.name} returned hess_eta_extra shape "
                        f"{deriv.hess_eta_extra.shape}, expected {(len(names),)}."
                    )
                if diag_hessian:
                    He = np.diag(np.diag(He))
                    Hbe = np.zeros_like(Hbe)
                Hexp[idx.extra_slices[k], idx.extra_slices[k]] += w * He
                Hexp[idx.beta_slices[k], idx.extra_slices[k]] += w * Hbe
                Hexp[idx.extra_slices[k], idx.beta_slices[k]] += w * Hbe.T

        sbar = np.sum(tau[i, :, None] * s_all_k, axis=0)
        V = np.zeros((d, d), dtype=float)
        for k in range(K):
            dk = (s_all_k[k] - sbar).reshape(-1, 1)
            V += float(tau[i, k]) * (dk @ dk.T)

        I += (-Hexp) - V

    I = 0.5 * (I + I.T) + ridge * np.eye(d)

    cov = None
    se = None
    try:
        cov = np.linalg.inv(I)
        cov = 0.5 * (cov + cov.T)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except Exception:
        cov = None
        se = None

    return LouisResult(
        theta_hat=theta0,
        info=I,
        cov=cov,
        se=se,
        param_names=idx.names,
        derivative_sources=derivative_sources,
    )
