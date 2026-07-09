from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy import special

Array = np.ndarray


@dataclass
class ComponentDerivatives:
    """
    Derivatives for one complete-data component log-density contribution.

    Coordinates are:
      eta = x^T beta
      extra_t = family transformed nuisance parameters, in extra_param_names order
    """

    score_eta: float
    hess_eta_eta: float
    score_extra: Array
    hess_extra_extra: Array
    hess_eta_extra: Array
    source: str
    analytic: bool


@dataclass
class MuExtraDerivatives:
    """
    Derivatives in (mu, extra_t) coordinates before link-function chain rule.
    """

    score_mu: float
    hess_mu_mu: float
    score_extra: Array
    hess_extra_extra: Array
    hess_mu_extra: Array


ANALYTIC_FAMILIES = {
    "gaussian",
    "student_t",
    "poisson",
    "nb2",
    "gamma",
    "exponential",
    "lognormal",
    "inverse_gaussian",
    "bernoulli",
    "geometric",
    "beta",
    "zip",
    "zinb",
    "skew_normal",
}

FINITE_DIFF_ONLY_FAMILIES = {
    "jf_skew_t",
    "azzalini_skew_t",
    "genhyperbolic",
}


def _empty() -> Tuple[Array, Array, Array]:
    return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)


def _as_float(x: Any) -> float:
    return float(np.asarray(x, dtype=float).reshape(-1)[0])


def _clip_pos(x: float, eps: float = 1e-12) -> float:
    return float(max(float(x), eps))


def _link_derivatives(link: Any, eta: float) -> Tuple[float, float, float]:
    eta_arr = np.array([float(eta)], dtype=float)
    mu = _as_float(link.inverse(eta_arr))
    d1 = _as_float(link.inverse_deriv(eta_arr))

    name = str(getattr(link, "name", "")).lower()
    if name == "identity":
        d2 = 0.0
    elif name == "log":
        d2 = d1
    elif name == "logit":
        d2 = d1 * (1.0 - 2.0 * mu)
    else:
        h = 1e-5 * max(1.0, abs(float(eta)))
        fp = _as_float(link.inverse(np.array([eta + h], dtype=float)))
        f0 = mu
        fm = _as_float(link.inverse(np.array([eta - h], dtype=float)))
        d2 = (fp - 2.0 * f0 + fm) / (h * h)
    return float(mu), float(d1), float(d2)


def _pack_extra(extra: Dict[str, Any], names: Sequence[str]) -> Array:
    if len(names) == 0:
        return np.zeros(0, dtype=float)
    return np.array([float(extra[n]) for n in names], dtype=float)


def _unpack_extra(v: Array, names: Sequence[str]) -> Dict[str, Any]:
    return {n: float(v[j]) for j, n in enumerate(names)}


def _finite_difference_derivatives(
    *,
    y: float,
    eta: float,
    family: Any,
    link: Any,
    extra: Dict[str, Any],
    fd_eps: float,
) -> ComponentDerivatives:
    names = list(family.extra_param_names)
    extra_t = family.transform_extra(extra)
    e0 = _pack_extra(extra_t, names)
    x0 = np.concatenate([np.array([float(eta)], dtype=float), e0])
    d = x0.size

    def fun(v: Array) -> float:
        eta_v = float(v[0])
        mu_v = _as_float(link.inverse(np.array([eta_v], dtype=float)))
        try:
            if names:
                extra_v = family.inverse_transform_extra(_unpack_extra(v[1:], names))
                family.validate_extra(extra_v)
            else:
                extra_v = {}
            ll = family.loglik_component(
                y=np.array([float(y)]),
                mu=np.array([mu_v], dtype=float),
                extra=extra_v,
            )
            val = float(ll[0])
            return val if np.isfinite(val) else -1e100
        except Exception:
            return -1e100

    score = np.zeros(d, dtype=float)
    hess = np.zeros((d, d), dtype=float)
    f0 = fun(x0)

    for j in range(d):
        hj = fd_eps * max(1.0, abs(float(x0[j])))
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += hj
        xm[j] -= hj
        fp = fun(xp)
        fm = fun(xm)
        score[j] = (fp - fm) / (2.0 * hj)
        hess[j, j] = (fp - 2.0 * f0 + fm) / (hj * hj)

    for j in range(d):
        hj = fd_eps * max(1.0, abs(float(x0[j])))
        for k in range(j + 1, d):
            hk = fd_eps * max(1.0, abs(float(x0[k])))
            xpp = x0.copy()
            xpm = x0.copy()
            xmp = x0.copy()
            xmm = x0.copy()
            xpp[j] += hj
            xpp[k] += hk
            xpm[j] += hj
            xpm[k] -= hk
            xmp[j] -= hj
            xmp[k] += hk
            xmm[j] -= hj
            xmm[k] -= hk
            val = (fun(xpp) - fun(xpm) - fun(xmp) + fun(xmm)) / (4.0 * hj * hk)
            hess[j, k] = val
            hess[k, j] = val

    return ComponentDerivatives(
        score_eta=float(score[0]),
        hess_eta_eta=float(hess[0, 0]),
        score_extra=score[1:].copy(),
        hess_extra_extra=hess[1:, 1:].copy(),
        hess_eta_extra=hess[0, 1:].copy(),
        source=f"{family.name}:finite_diff",
        analytic=False,
    )


def _gaussian(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    s = float(extra["log_sigma2"])
    var = _clip_pos(np.exp(s))
    r = float(y - mu)
    score_extra, hess_extra, hess_mu_extra = _empty()
    score_extra = np.array([-0.5 + 0.5 * r * r / var], dtype=float)
    hess_extra = np.array([[-0.5 * r * r / var]], dtype=float)
    hess_mu_extra = np.array([-r / var], dtype=float)
    return MuExtraDerivatives(r / var, -1.0 / var, score_extra, hess_extra, hess_mu_extra)


def _student_t(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    a = float(extra["log_sigma"])
    b = float(extra["log_nu_m2"])
    sigma2 = _clip_pos(np.exp(2.0 * a))
    w = _clip_pos(np.exp(b))
    nu = 2.0 + w
    r = float(y - mu)
    r2 = r * r
    D = nu * sigma2 + r2

    score_mu = (nu + 1.0) * r / D
    hess_mu_mu = (nu + 1.0) * (r2 - nu * sigma2) / (D * D)

    score_a = -1.0 + (nu + 1.0) * r2 / D
    hess_aa = -2.0 * (nu + 1.0) * r2 * nu * sigma2 / (D * D)
    hess_mu_a = -2.0 * (nu + 1.0) * r * nu * sigma2 / (D * D)

    u = r2 / sigma2
    q = 1.0 + u / nu
    c1 = 0.5 * special.digamma((nu + 1.0) / 2.0)
    c1 -= 0.5 * special.digamma(nu / 2.0)
    c1 -= 0.5 / nu
    l_nu = c1 - 0.5 * np.log(q) + 0.5 * (nu + 1.0) * u / (nu * (nu + u))

    c2 = 0.25 * special.polygamma(1, (nu + 1.0) / 2.0)
    c2 -= 0.25 * special.polygamma(1, nu / 2.0)
    c2 += 0.5 / (nu * nu)
    M = nu * (nu + u)
    N = 0.5 * u * (nu + 1.0)
    Tprime = (0.5 * u * M - N * (2.0 * nu + u)) / (M * M)
    l_nu_nu = c2 + 0.5 * u / M + Tprime

    score_b = w * l_nu
    hess_bb = w * l_nu + w * w * l_nu_nu
    hess_mu_b = w * r * (r2 - sigma2) / (D * D)
    hess_a_b = w * r2 * (r2 - sigma2) / (D * D)

    score_extra = np.array([score_a, score_b], dtype=float)
    hess_extra = np.array([[hess_aa, hess_a_b], [hess_a_b, hess_bb]], dtype=float)
    hess_mu_extra = np.array([hess_mu_a, hess_mu_b], dtype=float)
    return MuExtraDerivatives(score_mu, hess_mu_mu, score_extra, hess_extra, hess_mu_extra)


def _poisson(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    mu = _clip_pos(mu)
    score_extra, hess_extra, hess_mu_extra = _empty()
    return MuExtraDerivatives(y / mu - 1.0, -y / (mu * mu), score_extra, hess_extra, hess_mu_extra)


def _nb2_core(y: float, mu: float, log_alpha: float) -> Tuple[float, float, float, float, float]:
    mu = _clip_pos(mu)
    alpha = _clip_pos(np.exp(float(log_alpha)))
    r = 1.0 / alpha
    denom = r + mu

    score_mu = y / mu - (r + y) / denom
    hess_mu_mu = -y / (mu * mu) + (r + y) / (denom * denom)

    l_r = special.digamma(y + r) - special.digamma(r)
    l_r += np.log(r) + 1.0 - np.log(denom) - (r + y) / denom

    l_rr = special.polygamma(1, y + r) - special.polygamma(1, r)
    l_rr += 1.0 / r - 1.0 / denom - (mu - y) / (denom * denom)

    score_t = -r * l_r
    hess_t_t = r * l_r + r * r * l_rr
    hess_mu_t = r * (mu - y) / (denom * denom)
    return score_mu, hess_mu_mu, score_t, hess_t_t, hess_mu_t


def _nb2(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    sm, hmm, st, htt, hmt = _nb2_core(y, mu, float(extra["log_alpha"]))
    return MuExtraDerivatives(
        sm,
        hmm,
        np.array([st], dtype=float),
        np.array([[htt]], dtype=float),
        np.array([hmt], dtype=float),
    )


def _gamma(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    y = _clip_pos(y, 1e-300)
    mu = _clip_pos(mu)
    k = _clip_pos(np.exp(float(extra["log_shape"])))
    l_k = np.log(y) - y / mu - special.digamma(k) - np.log(mu) + np.log(k) + 1.0
    l_kk = -special.polygamma(1, k) + 1.0 / k

    score_mu = k * (y - mu) / (mu * mu)
    hess_mu_mu = k * (mu - 2.0 * y) / (mu ** 3)
    score_t = k * l_k
    hess_tt = k * l_k + k * k * l_kk
    hess_mu_t = score_mu
    return MuExtraDerivatives(
        score_mu,
        hess_mu_mu,
        np.array([score_t], dtype=float),
        np.array([[hess_tt]], dtype=float),
        np.array([hess_mu_t], dtype=float),
    )


def _exponential(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    y = _clip_pos(y, 1e-300)
    mu = _clip_pos(mu)
    score_extra, hess_extra, hess_mu_extra = _empty()
    return MuExtraDerivatives(
        (y - mu) / (mu * mu),
        (mu - 2.0 * y) / (mu ** 3),
        score_extra,
        hess_extra,
        hess_mu_extra,
    )


def _lognormal(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    y = _clip_pos(y, 1e-300)
    s = _clip_pos(np.exp(float(extra["log_sigma"])))
    s2 = s * s
    r = np.log(y) - mu
    score_mu = r / s2
    hess_mu_mu = -1.0 / s2
    score_t = -1.0 + r * r / s2
    hess_tt = -2.0 * r * r / s2
    hess_mu_t = -2.0 * r / s2
    return MuExtraDerivatives(
        score_mu,
        hess_mu_mu,
        np.array([score_t], dtype=float),
        np.array([[hess_tt]], dtype=float),
        np.array([hess_mu_t], dtype=float),
    )


def _inverse_gaussian(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    y = _clip_pos(y, 1e-300)
    mu = _clip_pos(mu)
    lam = _clip_pos(np.exp(float(extra["log_lambda"])))
    A = (y - mu) ** 2 / (2.0 * mu * mu * y)
    score_mu = lam * (y - mu) / (mu ** 3)
    hess_mu_mu = lam * (2.0 * mu - 3.0 * y) / (mu ** 4)
    score_t = 0.5 - lam * A
    hess_tt = -lam * A
    hess_mu_t = score_mu
    return MuExtraDerivatives(
        score_mu,
        hess_mu_mu,
        np.array([score_t], dtype=float),
        np.array([[hess_tt]], dtype=float),
        np.array([hess_mu_t], dtype=float),
    )


def _bernoulli(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    mu = float(np.clip(mu, 1e-12, 1.0 - 1e-12))
    score_extra, hess_extra, hess_mu_extra = _empty()
    score_mu = y / mu - (1.0 - y) / (1.0 - mu)
    hess_mu_mu = -y / (mu * mu) - (1.0 - y) / ((1.0 - mu) ** 2)
    return MuExtraDerivatives(score_mu, hess_mu_mu, score_extra, hess_extra, hess_mu_extra)


def _geometric(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    mu = _clip_pos(mu)
    score_extra, hess_extra, hess_mu_extra = _empty()
    score_mu = y / mu - (y + 1.0) / (1.0 + mu)
    hess_mu_mu = -y / (mu * mu) + (y + 1.0) / ((1.0 + mu) ** 2)
    return MuExtraDerivatives(score_mu, hess_mu_mu, score_extra, hess_extra, hess_mu_extra)


def _beta(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    y = float(np.clip(y, 1e-8, 1.0 - 1e-8))
    mu = float(np.clip(mu, 1e-8, 1.0 - 1e-8))
    phi = _clip_pos(np.exp(float(extra["log_phi"])))
    a = mu * phi
    b = (1.0 - mu) * phi
    ly = np.log(y)
    l1y = np.log(1.0 - y)

    G = -special.digamma(a) + special.digamma(b) + ly - l1y
    score_mu = phi * G
    hess_mu_mu = -phi * phi * (special.polygamma(1, a) + special.polygamma(1, b))

    l_phi = special.digamma(phi) - mu * special.digamma(a)
    l_phi -= (1.0 - mu) * special.digamma(b)
    l_phi += mu * ly + (1.0 - mu) * l1y
    l_phiphi = special.polygamma(1, phi)
    l_phiphi -= mu * mu * special.polygamma(1, a)
    l_phiphi -= (1.0 - mu) ** 2 * special.polygamma(1, b)

    score_t = phi * l_phi
    hess_tt = phi * l_phi + phi * phi * l_phiphi
    hess_mu_t = phi * G + phi * phi * (
        -mu * special.polygamma(1, a) + (1.0 - mu) * special.polygamma(1, b)
    )
    return MuExtraDerivatives(
        score_mu,
        hess_mu_mu,
        np.array([score_t], dtype=float),
        np.array([[hess_tt]], dtype=float),
        np.array([hess_mu_t], dtype=float),
    )


def _zip(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    mu = _clip_pos(mu)
    lt = float(extra.get("logit_theta", -20.0))
    theta = float(1.0 / (1.0 + np.exp(-np.clip(lt, -700.0, 700.0))))
    theta = float(np.clip(theta, 1e-12, 1.0 - 1e-12))
    dtheta = theta * (1.0 - theta)

    if y > 0:
        score_mu = y / mu - 1.0
        hess_mu_mu = -y / (mu * mu)
        score_t = -theta
        hess_tt = -dtheta
        hess_mu_t = 0.0
    else:
        e = float(np.exp(-mu))
        A = theta + (1.0 - theta) * e
        B = (1.0 - theta) * e
        C = dtheta * (1.0 - e)
        Ct = dtheta * (1.0 - 2.0 * theta) * (1.0 - e)
        score_mu = -B / A
        hess_mu_mu = B * theta / (A * A)
        score_t = C / A
        hess_tt = (Ct * A - C * C) / (A * A)
        Cmu = dtheta * e
        Amu = -B
        hess_mu_t = (Cmu * A - C * Amu) / (A * A)

    return MuExtraDerivatives(
        score_mu,
        hess_mu_mu,
        np.array([score_t], dtype=float),
        np.array([[hess_tt]], dtype=float),
        np.array([hess_mu_t], dtype=float),
    )


def _zinb(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    mu = _clip_pos(mu)
    la = float(extra.get("log_alpha", 0.0))
    lt = float(extra.get("logit_theta", -20.0))
    theta = float(1.0 / (1.0 + np.exp(-np.clip(lt, -700.0, 700.0))))
    theta = float(np.clip(theta, 1e-12, 1.0 - 1e-12))
    dtheta = theta * (1.0 - theta)

    sm, hmm, sa, haa, hma = _nb2_core(y, mu, la)
    if y > 0:
        return MuExtraDerivatives(
            sm,
            hmm,
            np.array([sa, -theta], dtype=float),
            np.array([[haa, 0.0], [0.0, -dtheta]], dtype=float),
            np.array([hma, 0.0], dtype=float),
        )

    r = 1.0 / _clip_pos(np.exp(la))
    p0 = float(np.exp(r * (np.log(r) - np.log(r + mu))))
    A = theta + (1.0 - theta) * p0

    g_mu, H_mu_mu, g_a, H_aa, H_mu_a = sm, hmm, sa, haa, hma
    A_mu = (1.0 - theta) * p0 * g_mu
    A_a = (1.0 - theta) * p0 * g_a
    A_mumu = (1.0 - theta) * p0 * (H_mu_mu + g_mu * g_mu)
    A_aa = (1.0 - theta) * p0 * (H_aa + g_a * g_a)
    A_mua = (1.0 - theta) * p0 * (H_mu_a + g_mu * g_a)

    C = dtheta * (1.0 - p0)
    Ct = dtheta * (1.0 - 2.0 * theta) * (1.0 - p0)
    A_mut = -dtheta * p0 * g_mu
    A_at = -dtheta * p0 * g_a

    score_mu = A_mu / A
    score_a = A_a / A
    score_t = C / A
    hess_mu_mu = (A_mumu * A - A_mu * A_mu) / (A * A)
    hess_aa = (A_aa * A - A_a * A_a) / (A * A)
    hess_mu_a = (A_mua * A - A_mu * A_a) / (A * A)
    hess_tt = (Ct * A - C * C) / (A * A)
    hess_mu_t = (A_mut * A - A_mu * C) / (A * A)
    hess_a_t = (A_at * A - A_a * C) / (A * A)

    return MuExtraDerivatives(
        score_mu,
        hess_mu_mu,
        np.array([score_a, score_t], dtype=float),
        np.array([[hess_aa, hess_a_t], [hess_a_t, hess_tt]], dtype=float),
        np.array([hess_mu_a, hess_mu_t], dtype=float),
    )


def _skew_normal(y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    log_scale = float(extra["log_scale"])
    shape = float(extra["shape"])
    scale = _clip_pos(np.exp(log_scale))
    z = (float(y) - mu) / scale
    t = shape * z

    log_phi = -0.5 * t * t - 0.5 * np.log(2.0 * np.pi)
    log_Phi = special.log_ndtr(t)
    lam = float(np.exp(log_phi - log_Phi))
    lam_prime = -lam * (t + lam)

    lz = -z + shape * lam
    lzz = -1.0 + shape * shape * lam_prime
    lz_shape = lam + shape * z * lam_prime

    score_mu = -lz / scale
    hess_mu_mu = lzz / (scale * scale)
    score_log_scale = -1.0 - z * lz
    score_shape = z * lam
    hess_log_scale = z * lz + z * z * lzz
    hess_shape_shape = z * z * lam_prime
    hess_log_scale_shape = -z * lz_shape
    hess_mu_log_scale = (lz + z * lzz) / scale
    hess_mu_shape = -lz_shape / scale

    return MuExtraDerivatives(
        score_mu,
        hess_mu_mu,
        np.array([score_log_scale, score_shape], dtype=float),
        np.array(
            [
                [hess_log_scale, hess_log_scale_shape],
                [hess_log_scale_shape, hess_shape_shape],
            ],
            dtype=float,
        ),
        np.array([hess_mu_log_scale, hess_mu_shape], dtype=float),
    )


def _mu_extra_derivatives(family: Any, y: float, mu: float, extra: Dict[str, Any]) -> MuExtraDerivatives:
    name = str(family.name).lower()
    if name == "gaussian":
        return _gaussian(y, mu, extra)
    if name == "student_t":
        return _student_t(y, mu, extra)
    if name == "poisson":
        return _poisson(y, mu, extra)
    if name == "nb2":
        return _nb2(y, mu, extra)
    if name == "gamma":
        return _gamma(y, mu, extra)
    if name == "exponential":
        return _exponential(y, mu, extra)
    if name == "lognormal":
        return _lognormal(y, mu, extra)
    if name == "inverse_gaussian":
        return _inverse_gaussian(y, mu, extra)
    if name == "bernoulli":
        return _bernoulli(y, mu, extra)
    if name == "geometric":
        return _geometric(y, mu, extra)
    if name == "beta":
        return _beta(y, mu, extra)
    if name == "zip":
        return _zip(y, mu, extra)
    if name == "zinb":
        return _zinb(y, mu, extra)
    if name == "skew_normal":
        return _skew_normal(y, mu, extra)
    raise NotImplementedError(f"No analytic derivative block implemented for family '{family.name}'.")


def component_derivatives(
    *,
    y: float,
    eta: float,
    family: Any,
    link: Any,
    extra: Dict[str, Any],
    method: str = "analytic",
    fd_eps: float = 1e-5,
) -> ComponentDerivatives:
    """
    Derivatives of log f(y; g^{-1}(eta), extra) in (eta, extra_t) coordinates.

    method:
      - "analytic": require analytic block
      - "auto": use analytic block when available, otherwise finite differences
      - "finite_diff": central finite-difference block
    """
    method = str(method).lower()
    if method not in {"analytic", "auto", "finite_diff"}:
        raise ValueError("method must be one of: 'analytic', 'auto', 'finite_diff'.")

    if method == "finite_diff":
        return _finite_difference_derivatives(
            y=y, eta=eta, family=family, link=link, extra=extra, fd_eps=fd_eps
        )

    try:
        mu, dmu, d2mu = _link_derivatives(link, eta)
        extra_t = family.transform_extra(extra)
        block = _mu_extra_derivatives(family, y=float(y), mu=mu, extra=extra_t)

        score_eta = block.score_mu * dmu
        hess_eta_eta = block.hess_mu_mu * dmu * dmu + block.score_mu * d2mu
        hess_eta_extra = block.hess_mu_extra * dmu

        return ComponentDerivatives(
            score_eta=float(score_eta),
            hess_eta_eta=float(hess_eta_eta),
            score_extra=np.asarray(block.score_extra, dtype=float),
            hess_extra_extra=np.asarray(block.hess_extra_extra, dtype=float),
            hess_eta_extra=np.asarray(hess_eta_extra, dtype=float),
            source=f"{family.name}:analytic",
            analytic=True,
        )
    except Exception:
        if method == "analytic":
            raise
        return _finite_difference_derivatives(
            y=y, eta=eta, family=family, link=link, extra=extra, fd_eps=fd_eps
        )


def available_derivative_families() -> Dict[str, Tuple[str, ...]]:
    return {
        "analytic": tuple(sorted(ANALYTIC_FAMILIES)),
        "finite_diff_only": tuple(sorted(FINITE_DIFF_ONLY_FAMILIES)),
    }
