# Simulation Study Plan

This note records the simulation evidence we have, what is currently usable, and
the clean run plan for a journal-grade paper.

## Current Status

The current checkpoint results are useful for orientation but should not be
treated as final paper evidence.

- Scenario A, model selection: mostly usable for the broad story. Examples 1, 2,
  and 4 show high structure recovery as n grows. Example 3 is not a strict
  family-recovery success because BIC usually prefers `nb2 + nb2` over
  `poisson + nb2`; this is better described as an equivalence/generalization
  phenomenon, not as strict recovery of the data-generating family labels.
- Scenario B, variable selection: not yet publication-ready. Lasso improves
  false positives relative to no penalty, but FPR remains too high in several
  examples. The fixed lambda is too crude; we need a tuning grid and more
  metrics.
- Scenario C, inference: usable only as a baseline for the regular continuous
  case. Example 1 has coverage near nominal. The count example has severe
  undercoverage and should not be presented as a success until the likelihood,
  label alignment, and information calculations are improved.
- Existing top-10 simulation leaderboards were not genuine leaderboards because
  the stored pipeline output contained only the winning unpenalized model. The
  code has been updated so new Scenario A checkpoints store actual BIC and
  held-out likelihood leaderboards from the beam-search candidate pool.
- The old `paper_outputs` directory should be treated as archival. Fresh runs
  now default to `paper_outputs/v3_clean_louis`.

## Clean Run Defaults

The active full-run version is `v3_clean_louis`.

Default output root:

```text
paper_outputs/v3_clean_louis
```

The runner writes `run_metadata.json` at the output root. This file records the
simulation version, examples, sample sizes, inference methods, and lambda grid.

Default scenarios:

- Scenario A: examples `1,2,3,4`, sample sizes `500,1000,1500`.
- Scenario B: examples `1,2,4`, with `n=1000`, `p=20`.
- Scenario C: examples `1,2,4`, sample sizes `500,1000,1500`.
- Scenario A beam width: `10`.

Rationale:

- Example 1: real-valued heavy-tail mixture, `gaussian + student_t`.
- Example 2: positive mixture, `gamma + lognormal`. The lognormal component now
  uses the identity link because `LogNormalFamily` models the log-location.
- Example 3: nested/equivalence count case, `poisson + nb2`; included in
  Scenario A but not as a strict inference/variable-selection showcase.
- Example 4: zero-inflated count case, `poisson + zip`.

Scenario B now tunes lasso and elastic-net over the default lambda grid
`[1,2,5,10,20,50,100,200]` and selects the best lambda within each penalty class
by active-set BIC. It stores both the selected rows and the full lambda path.

Scenario C now runs both `louis` and `numeric` standard errors by default:

```text
MIXGLM_INFERENCE_METHODS=louis,numeric
```

## Required Paper-Level Simulation Evidence

### Scenario A: Family and Component Selection

Goal: show that the method can recover, or nearly recover, the correct
component-family structure when the candidate model class is regular enough and
the components are identifiable.

Required outputs:

- Selection rate for exact family multiset and K.
- Selection rate for scientifically equivalent structures, e.g. `nb2 + nb2`
  when the true count mixture is `poisson + nb2`.
- Held-out log-likelihood of the BIC-selected model versus the oracle true
  structure.
- Top-10 BIC leaderboard and top-10 held-out likelihood leaderboard for a
  representative replication.
- Frequency leaderboard over replications: how often each model appears in the
  top 10 by BIC and prediction.
- Convergence/failure rate and median runtime.

Recommended examples:

- Real-valued heavy tail: `gaussian + student_t`.
- Positive continuous: `gamma + lognormal`.
- Count overdispersion/equivalence: `poisson + nb2`, reported with an explicit
  equivalence interpretation.
- Zero-inflated counts: `poisson + zip`, if strict recovery remains stable.

### Scenario B: Variable Selection

Goal: show that penalization is useful for sparse component regressions, without
overclaiming valid post-selection Wald inference.

Required outputs:

- TPR, FPR, FDR, support size, and false negative rate.
- Coefficient MSE after label alignment.
- Held-out log-likelihood or prediction loss.
- Lambda path or grid summary, not a single fixed lambda.
- Comparison of no penalty, lasso, elastic net, and optionally adaptive lasso.
- Optional oracle active-set refit to separate selection from estimation.

Current action: use the clean `v3_clean_louis` runner. Fixed `lam=20` is no
longer the simulation design; lasso and elastic-net are selected from a lambda
grid by active-set BIC, and the full path is saved.

### Scenario C: Inference

Goal: validate standard errors only in regimes where the theory supports them.

Required outputs:

- Bias, empirical SD, mean model-based SE, SE/SD ratio, CI length, and coverage.
- Separate tables for beta parameters and nuisance parameters.
- Comparison of numeric Hessian, Louis finite-difference, and analytic Louis
  once implemented.
- Penalized fits should not use naive Wald intervals. For penalized models,
  report bootstrap or post-selection oracle/refit intervals separately.

Initial paper-safe inference target:

- Unpenalized true-family model.
- Fixed K and fixed family tuple.
- Continuous regular examples first.
- Count examples only after the analytic/closed-form information path is stable.

## Louis Identity Implementation Plan

The current `src/mixglm/inference/louis.py` implements Louis' identity but still
uses finite differences for component log-density score and Hessian blocks.
That is useful as a baseline but not a closed-form contribution.

Tier 1 closed-form targets:

- Gaussian.
- Lognormal.
- Poisson.
- Gamma.
- Negative binomial NB2.
- Bernoulli/geometric/exponential if included in simulation spaces.

Tier 2 closed-form targets:

- Student-t, including scale and degrees-of-freedom derivatives.
- ZIP and ZINB, with separate zero and positive-count branches.

Tier 3 targets:

- Skew-normal, skew-t, and generalized hyperbolic families. These are much more
  algebraically delicate. Unless they are central to the paper story, they
  should remain supplementary or use numeric/autodiff derivatives without a
  strong closed-form claim.

Recommended API:

- Add a family method that returns per-observation score and Hessian with
  respect to the linear predictor and transformed nuisance parameters.
- Apply link-function chain rules centrally.
- Let Louis assembly consume these analytic blocks, falling back to finite
  differences only when a family does not implement the analytic method.
- Validate each analytic block against central finite differences on randomized
  parameter values before using it in simulation.

## Immediate Commands

Run derivative validation:

```bash
.venv/bin/python experiments/simulations/validate_louis_derivatives.py
```

Run a zero-rep output-root smoke check:

```bash
MIXGLM_N_REPS=0 \
MIXGLM_OUTPUT_ROOT=/tmp/mixglm_smoke_output \
.venv/bin/python experiments/simulations/master_run.py
```

Run a tiny pilot with reduced Scenario B size and lambda grid:

```bash
MIXGLM_N_REPS=1 \
MIXGLM_SAMPLE_SIZES=100 \
MIXGLM_SCENARIO_A_EXAMPLES=1 \
MIXGLM_SCENARIO_A_BEAM_WIDTH=2 \
MIXGLM_SCENARIO_B_EXAMPLES=1 \
MIXGLM_SCENARIO_C_EXAMPLES=1 \
MIXGLM_SCENARIO_B_N=100 \
MIXGLM_SCENARIO_B_P=10 \
MIXGLM_LAMBDA_GRID=10,50 \
MIXGLM_OUTPUT_ROOT=/tmp/mixglm_pilot_v3 \
.venv/bin/python experiments/simulations/master_run.py
```

Run the full master simulation after choosing final settings:

```bash
MIXGLM_OUTPUT_ROOT=paper_outputs/v3_clean_louis \
.venv/bin/python experiments/simulations/master_run.py
```

On HPC, use the existing submit scripts after confirming paths and environment.
