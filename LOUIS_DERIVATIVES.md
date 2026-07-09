# Closed-Form Louis Blocks

This note documents the derivative coordinates used by the Louis observed
information implementation.

For component k and observation i, write

```text
eta_i = x_i^T beta_k,    mu_i = h_k(eta_i),
ell_ik = log f_k(y_i; mu_i, gamma_k),
```

where `gamma_k` denotes the transformed nuisance parameters already used by the
optimizer, in `family.extra_param_names` order.

The family block supplies derivatives in `(mu, gamma)` coordinates:

```text
s_mu      = d ell / d mu
H_mumu    = d^2 ell / d mu^2
s_gamma   = d ell / d gamma
H_gammagamma = d^2 ell / d gamma d gamma^T
H_mugamma = d^2 ell / d mu d gamma^T
```

The link chain rule gives the derivatives in `(eta, gamma)` coordinates:

```text
s_eta      = s_mu h'(eta)
H_etaeta   = H_mumu {h'(eta)}^2 + s_mu h''(eta)
H_etagamma = H_mugamma h'(eta).
```

The complete-data score/Hessian contribution for beta is then

```text
s_beta      = x_i s_eta
H_betabeta  = x_i x_i^T H_etaeta
H_betagamma = x_i H_etagamma.
```

Louis' identity is assembled as

```text
I_obs(theta) =
  - E[H_c(theta) | y]
  - Var[S_c(theta) | y],
```

where the conditional expectation and variance are computed with the posterior
responsibilities.

## Family Inventory

Registered families:

- `gaussian`
- `student_t`
- `poisson`
- `nb2`
- `gamma`
- `exponential`
- `lognormal`
- `inverse_gaussian`
- `bernoulli`
- `geometric`
- `beta`
- `zip`
- `zinb`
- `skew_normal`
- `jf_skew_t`
- `azzalini_skew_t`
- `genhyperbolic`

Implemented analytic Louis blocks:

- `gaussian`
- `student_t`
- `poisson`
- `nb2`
- `gamma`
- `exponential`
- `lognormal`
- `inverse_gaussian`
- `bernoulli`
- `geometric`
- `beta`
- `zip`
- `zinb`
- `skew_normal`

Finite-difference fallback only:

- `jf_skew_t`
- `azzalini_skew_t`
- `genhyperbolic`

These three families delegate density evaluation to SciPy or an external
package. They should not be described as closed-form analytic Louis
implementations until their special-function derivatives are implemented and
validated.

## Validation

Run:

```bash
.venv/bin/python experiments/simulations/validate_louis_derivatives.py
```

The script compares each analytic derivative block against central finite
differences in `(eta, transformed nuisance)` coordinates. This is not a proof,
but it is a useful guard against algebraic and implementation mistakes.
