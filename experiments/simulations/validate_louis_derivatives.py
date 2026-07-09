#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mixglm.families.registry import FAMILIES, register_defaults as register_families
from mixglm.links.registry import LINKS, register_defaults as register_links
from mixglm.inference.analytic_blocks import component_derivatives, available_derivative_families


CASES = [
    ("gaussian", "identity", 0.7, 0.2, {"log_sigma2": np.log(1.3)}),
    ("student_t", "identity", 0.7, 0.1, {"log_sigma": np.log(1.2), "log_nu_m2": np.log(4.0)}),
    ("poisson", "log", 3.0, 0.4, {}),
    ("nb2", "log", 3.0, 0.3, {"log_alpha": np.log(0.7)}),
    ("gamma", "log", 1.4, 0.2, {"log_shape": np.log(3.0)}),
    ("exponential", "log", 1.4, 0.2, {}),
    ("lognormal", "identity", 1.3, 0.1, {"log_sigma": np.log(0.7)}),
    ("inverse_gaussian", "log", 1.2, 0.1, {"log_lambda": np.log(2.0)}),
    ("bernoulli", "logit", 1.0, 0.3, {}),
    ("geometric", "log", 2.0, 0.3, {}),
    ("beta", "logit", 0.4, 0.2, {"log_phi": np.log(5.0)}),
    ("zip", "log", 0.0, 0.3, {"logit_theta": -0.7}),
    ("zip", "log", 3.0, 0.3, {"logit_theta": -0.7}),
    ("zinb", "log", 0.0, 0.3, {"log_alpha": np.log(0.7), "logit_theta": -0.7}),
    ("zinb", "log", 3.0, 0.3, {"log_alpha": np.log(0.7), "logit_theta": -0.7}),
    ("skew_normal", "identity", 0.7, 0.1, {"log_scale": np.log(1.2), "shape": 1.5}),
]


def _vectorize(block):
    return np.concatenate(
        [
            np.array([block.score_eta, block.hess_eta_eta]),
            np.ravel(block.score_extra),
            np.ravel(block.hess_extra_extra),
            np.ravel(block.hess_eta_extra),
        ]
    )


def main() -> int:
    register_families()
    register_links()

    print("Registered families:", ", ".join(FAMILIES.available()))
    fams = available_derivative_families()
    print("Analytic Louis blocks:", ", ".join(fams["analytic"]))
    print("Finite-difference fallback:", ", ".join(fams["finite_diff_only"]))
    print()

    failed = False
    for fam_name, link_name, y, eta, extra in CASES:
        fam = FAMILIES.create(fam_name)
        link = LINKS.create(link_name)
        analytic = component_derivatives(
            y=y, eta=eta, family=fam, link=link, extra=extra, method="analytic"
        )
        finite_diff = component_derivatives(
            y=y, eta=eta, family=fam, link=link, extra=extra, method="finite_diff", fd_eps=1e-5
        )
        va = _vectorize(analytic)
        vf = _vectorize(finite_diff)
        max_abs = float(np.nanmax(np.abs(va - vf))) if va.size else 0.0
        rel = max_abs / max(1.0, float(np.nanmax(np.abs(vf))) if vf.size else 1.0)
        ok = np.isfinite(max_abs) and rel <= 2e-3
        failed = failed or not ok
        mark = "OK" if ok else "FAIL"
        print(f"{mark:4s} {fam_name:18s} y={y:<4g} max_abs={max_abs:.3e} rel={rel:.3e}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
