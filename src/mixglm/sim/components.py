# src/mixglm/sim/components.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np

Array = np.ndarray


# ------------------------- helpers -------------------------

def _get(extra: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in extra:
        if default is None:
            raise KeyError(f"Missing required extra parameter '{key}'.")
        return default
    return extra[key]


def _pos(x: Array, eps: float = 1e-12) -> Array:
    x = np.asarray(x, dtype=float)
    return np.clip(x, eps, None)


def _unit(x: Array, eps: float = 1e-8) -> Array:
    x = np.asarray(x, dtype=float)
    return np.clip(x, eps, 1.0 - eps)


# ------------------------- continuous samplers -------------------------

def gaussian_sampler(loc: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    y = loc + Normal(0, sigma^2)
    extra:
      - log_sigma (float)  OR sigma (float)
    """
    if "sigma" in extra:
        sigma = float(extra["sigma"])
    else:
        sigma = float(np.exp(float(_get(extra, "log_sigma"))))
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return np.asarray(loc, dtype=float) + rng.normal(loc=0.0, scale=sigma, size=np.asarray(loc).shape[0])


def student_t_sampler(loc: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    y = loc + scale * t_df
    extra:
      - df or log_df
      - scale or log_scale
    """
    if "df" in extra:
        df = float(extra["df"])
    else:
        df = float(np.exp(float(_get(extra, "log_df"))))
    if df <= 0:
        raise ValueError("df must be positive.")

    if "scale" in extra:
        scale = float(extra["scale"])
    else:
        scale = float(np.exp(float(_get(extra, "log_scale"))))
    if scale <= 0:
        raise ValueError("scale must be positive.")

    t = rng.standard_t(df=df, size=np.asarray(loc).shape[0])
    return np.asarray(loc, dtype=float) + scale * t


def lognormal_sampler(mu_log: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Y ~ LogNormal(meanlog=mu_log, sdlog=sigma_log)
    extra:
      - log_sigma (sdlog) OR sigma
    """
    if "sigma" in extra:
        sigma = float(extra["sigma"])
    else:
        sigma = float(np.exp(float(_get(extra, "log_sigma"))))
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return rng.lognormal(mean=np.asarray(mu_log, dtype=float), sigma=sigma)


def exponential_sampler(mean: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Exponential with mean parameterization:
      rate = 1/mean
    extra: none
    """
    m = _pos(mean)
    rate = 1.0 / m
    return rng.exponential(scale=1.0 / rate, size=m.shape[0])


def gamma_sampler(mean: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Gamma with mean/shape parameterization.
    Common choice: shape=k, scale=mean/k.
    extra:
      - log_shape OR shape
    """
    if "shape" in extra:
        shape = float(extra["shape"])
    else:
        shape = float(np.exp(float(_get(extra, "log_shape"))))
    if shape <= 0:
        raise ValueError("shape must be positive.")
    m = _pos(mean)
    scale = m / shape
    return rng.gamma(shape=shape, scale=scale, size=m.shape[0])


def inverse_gaussian_sampler(mean: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Inverse Gaussian using NumPy's wald generator:
      rng.wald(mean, scale, size)

    NOTE: NumPy parameter 'scale' equals lambda in some parameterizations.
    We treat extra['log_lambda'] (or lambda) as the IG shape parameter lambda>0.
    extra:
      - log_lambda OR lambda
    """
    if "lambda" in extra:
        lam = float(extra["lambda"])
    else:
        lam = float(np.exp(float(_get(extra, "log_lambda"))))
    if lam <= 0:
        raise ValueError("lambda must be positive.")
    m = _pos(mean)
    return rng.wald(mean=m, scale=lam, size=m.shape[0])


def skew_normal_sampler(loc: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Skew-normal using SciPy:
      scipy.stats.skewnorm.rvs(a=shape, loc=loc, scale=scale)

    extra:
      - shape
      - log_scale OR scale
    """
    shape = float(_get(extra, "shape", 0.0))
    if "scale" in extra:
        scale = float(extra["scale"])
    else:
        scale = float(np.exp(float(_get(extra, "log_scale"))))
    if scale <= 0:
        raise ValueError("scale must be positive.")

    try:
        from scipy.stats import skewnorm
    except Exception as e:
        raise ImportError("skew_normal_sampler requires scipy.stats.skewnorm") from e

    return skewnorm.rvs(a=shape, loc=np.asarray(loc, dtype=float), scale=scale, random_state=rng)


def jf_skew_t_sampler(loc: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Jones-Faddy skew-t (SciPy jf_skew_t) sampler:
      scipy.stats.jf_skew_t.rvs(a, b, loc=loc, scale=scale)

    extra:
      - log_a or a
      - log_b or b
      - log_scale or scale
    """
    if "a" in extra:
        a = float(extra["a"])
    else:
        a = float(np.exp(float(_get(extra, "log_a"))))
    if "b" in extra:
        b = float(extra["b"])
    else:
        b = float(np.exp(float(_get(extra, "log_b"))))
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive.")

    if "scale" in extra:
        scale = float(extra["scale"])
    else:
        scale = float(np.exp(float(_get(extra, "log_scale"))))
    if scale <= 0:
        raise ValueError("scale must be positive.")

    try:
        from scipy.stats import jf_skew_t
    except Exception as e:
        raise ImportError("jf_skew_t_sampler requires scipy.stats.jf_skew_t") from e

    return jf_skew_t.rvs(a=a, b=b, loc=np.asarray(loc, dtype=float), scale=scale, random_state=rng)


def genhyperbolic_sampler(loc: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Generalized Hyperbolic sampler (SciPy genhyperbolic):
      scipy.stats.genhyperbolic.rvs(p, a, b, loc=loc, scale=scale)

    extra:
      - p
      - log_a or a
      - b
      - log_scale or scale

    Must satisfy a > |b|.
    """
    p = float(_get(extra, "p", 1.0))
    if "a" in extra:
        a = float(extra["a"])
    else:
        a = float(np.exp(float(_get(extra, "log_a"))))
    b = float(_get(extra, "b", 0.0))
    if "scale" in extra:
        scale = float(extra["scale"])
    else:
        scale = float(np.exp(float(_get(extra, "log_scale"))))

    if a <= abs(b) + 1e-8:
        raise ValueError(f"genhyperbolic requires a > |b|. Got a={a}, b={b}.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    try:
        from scipy.stats import genhyperbolic
    except Exception as e:
        raise ImportError("genhyperbolic_sampler requires scipy.stats.genhyperbolic") from e

    return genhyperbolic.rvs(p=p, a=a, b=b, loc=np.asarray(loc, dtype=float), scale=scale, random_state=rng)


def azzalini_skew_t_sampler(loc: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Azzalini skew-t sampler using external skewt-scipy package.

    extra:
      - shape (a)
      - df or log_df
      - scale or log_scale
    """
    shape = float(_get(extra, "shape", 0.0))

    if "df" in extra:
        df = float(extra["df"])
    else:
        df = float(np.exp(float(_get(extra, "log_df"))))
    if df <= 0:
        raise ValueError("df must be positive.")

    if "scale" in extra:
        scale = float(extra["scale"])
    else:
        scale = float(np.exp(float(_get(extra, "log_scale"))))
    if scale <= 0:
        raise ValueError("scale must be positive.")

    try:
        from skewt_scipy.skewt import skewt
    except Exception as e:
        raise ImportError("azzalini_skew_t_sampler requires 'skewt-scipy' (skewt_scipy.skewt)") from e

    return skewt.rvs(a=shape, df=df, loc=np.asarray(loc, dtype=float), scale=scale, size=np.asarray(loc).shape[0], random_state=rng)


# ------------------------- discrete samplers -------------------------

def poisson_sampler(mean: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Poisson with mean parameterization.
    """
    lam = _pos(mean)
    return rng.poisson(lam=lam, size=lam.shape[0]).astype(float)


def nb2_sampler(mean: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    NB2 sampler using mean and dispersion alpha (>0):
      Var(Y) = mu + alpha * mu^2

    Convert to NumPy negative_binomial(n, p) with:
      n = 1/alpha
      p = n / (n + mu)

    extra:
      - alpha or log_alpha
    """
    if "alpha" in extra:
        alpha = float(extra["alpha"])
    else:
        alpha = float(np.exp(float(_get(extra, "log_alpha"))))
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    mu = _pos(mean)
    n = 1.0 / alpha
    p = n / (n + mu)
    # NumPy wants integer n? It accepts float n for gamma-poisson mixture style; still works in practice.
    y = rng.negative_binomial(n=n, p=p, size=mu.shape[0])
    return y.astype(float)

def bernoulli_sampler(mu: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """Bernoulli sampler (mu is probability)."""
    p = np.clip(mu, 0.0, 1.0)
    return rng.binomial(n=1, p=p).astype(float)

def geometric_sampler(mu: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """Geometric sampler (failures before first success)."""
    m = _pos(mu)
    p = 1.0 / (1.0 + m)
    return (rng.geometric(p=p) - 1.0).astype(float)

def zip_sampler(mu: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """Zero-Inflated Poisson."""
    theta = float(extra.get("theta", 0.0))
    if "logit_theta" in extra:
        lt = float(extra["logit_theta"])
        theta = 1.0 / (1.0 + np.exp(-lt))

    lam = _pos(mu)
    y_pois = rng.poisson(lam=lam).astype(float)
    is_zero = rng.binomial(n=1, p=theta, size=lam.shape[0])
    y_pois[is_zero == 1] = 0.0
    return y_pois

def zinb_sampler(mu: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """Zero-Inflated Negative Binomial."""
    theta = float(extra.get("theta", 0.0))
    if "logit_theta" in extra:
        lt = float(extra["logit_theta"])
        theta = 1.0 / (1.0 + np.exp(-lt))

    y_nb = nb2_sampler(mu, rng=rng, extra=extra)
    is_zero = rng.binomial(n=1, p=theta, size=mu.shape[0])
    y_nb[is_zero == 1] = 0.0
    return y_nb


# ------------------------- (0,1) samplers -------------------------

def beta_sampler(mu: Array, *, rng: np.random.Generator, extra: Dict[str, Any]) -> Array:
    """
    Beta sampler with mean/precision:
      a = mu * phi
      b = (1-mu) * phi
    extra:
      - phi or log_phi
    """
    if "phi" in extra:
        phi = float(extra["phi"])
    else:
        phi = float(np.exp(float(_get(extra, "log_phi"))))
    if phi <= 0:
        raise ValueError("phi must be positive.")

    m = _unit(mu)
    a = m * phi
    b = (1.0 - m) * phi
    return rng.beta(a=a, b=b, size=m.shape[0])
