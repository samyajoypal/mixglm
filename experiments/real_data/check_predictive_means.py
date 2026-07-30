#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mixglm.families.registry import FAMILIES, register_defaults as register_families


def _assert_close(name: str, got: np.ndarray, expected: np.ndarray, tol: float = 1e-10) -> None:
    err = float(np.max(np.abs(np.asarray(got, dtype=float) - np.asarray(expected, dtype=float))))
    if not np.isfinite(err) or err > tol:
        raise AssertionError(f"{name}: max_abs_error={err:.3e} exceeds tolerance {tol:.1e}")
    print(f"OK   {name:14s} max_abs_error={err:.3e}")


def main() -> int:
    register_families()
    mu = np.array([0.4, 1.2, 3.0], dtype=float)

    identity_mean_families = [
        ("gaussian", {}),
        ("student_t", {"log_sigma": np.log(1.1), "log_nu_m2": np.log(5.0)}),
        ("poisson", {}),
        ("nb2", {"log_alpha": np.log(0.7)}),
        ("gamma", {"log_shape": np.log(3.0)}),
        ("exponential", {}),
        ("inverse_gaussian", {"log_lambda": np.log(2.0)}),
        ("geometric", {}),
    ]

    for fam_name, extra in identity_mean_families:
        fam = FAMILIES.create(fam_name)
        _assert_close(fam_name, fam.mean_from_mu(mu, extra), mu)

    prob_mu = np.array([0.2, 0.4, 0.7], dtype=float)
    for fam_name, extra in [
        ("bernoulli", {}),
        ("beta", {"log_phi": np.log(5.0)}),
    ]:
        fam = FAMILIES.create(fam_name)
        _assert_close(fam_name, fam.mean_from_mu(prob_mu, extra), prob_mu)

    sigma = 0.7
    fam = FAMILIES.create("lognormal")
    _assert_close(
        "lognormal",
        fam.mean_from_mu(mu, {"log_sigma": np.log(sigma)}),
        np.exp(mu + 0.5 * sigma * sigma),
    )

    theta = 1.0 / (1.0 + np.exp(0.7))
    fam = FAMILIES.create("zip")
    _assert_close("zip", fam.mean_from_mu(mu, {"logit_theta": -0.7}), (1.0 - theta) * mu)

    fam = FAMILIES.create("zinb")
    _assert_close(
        "zinb",
        fam.mean_from_mu(mu, {"log_alpha": np.log(0.7), "logit_theta": -0.7}),
        (1.0 - theta) * mu,
    )

    scale = 1.2
    shape = 1.5
    delta = shape / np.sqrt(1.0 + shape * shape)
    fam = FAMILIES.create("skew_normal")
    _assert_close(
        "skew_normal",
        fam.mean_from_mu(mu, {"log_scale": np.log(scale), "shape": shape}),
        mu + scale * delta * np.sqrt(2.0 / np.pi),
    )

    print("Family-specific predictive means validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
