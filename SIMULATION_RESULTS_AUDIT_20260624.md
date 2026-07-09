# Simulation Results Audit: 2026-06-24 Fresh Louis Run

Fresh HPC job `22561285` completed successfully:

- SLURM state: `COMPLETED`, exit code `0:0`
- Runtime: `01:48:04`
- Output root: `paper_outputs/v3_clean_louis_fresh_20260624_b`
- Tasks evaluated: 2400 checkpointed simulation tasks
- Local copy: `experiments/simulations/paper_outputs/v3_clean_louis_fresh_20260624_b`
- Reconstructed raw aggregate: `raw_results_from_checkpoints.json`

## Bottom Line

These results should **not** be inserted into the manuscript as final top-journal
simulation evidence. The job ran cleanly, and several parts are encouraging, but
the simulation study is not yet publishable as a complete package.

The main reason is not a cluster failure. It is statistical: the positive-support
Gamma/lognormal scenario does not support the intended story. BIC overwhelmingly
selects identical Gamma/Gamma or Lognormal/Lognormal mixtures, strict selection
decreases with sample size, and the corresponding inference coverage is poor.

## Scenario A: Model Selection

Model-selection performance is strong for Examples 1 and 4, mixed but defensible
under equivalence for Example 3, and poor for Example 2.

| Example | Truth | n=500 strict/equiv | n=1000 strict/equiv | n=1500 strict/equiv | Main finding |
|---|---|---:|---:|---:|---|
| 1 | Gaussian + Student-t | 0.75 / 0.75 | 0.96 / 0.96 | 0.99 / 0.99 | Publishable |
| 2 | Gamma + Lognormal | 0.14 / 0.14 | 0.07 / 0.07 | 0.04 / 0.04 | Not publishable |
| 3 | Poisson + NB2 | 0.00 / 0.48 | 0.00 / 0.87 | 0.00 / 0.98 | Needs equivalence framing |
| 4 | Poisson + ZIP | 1.00 / 1.00 | 0.99 / 0.99 | 0.99 / 0.99 | Publishable |

For Example 2, the top BIC model is usually `gamma + gamma`:

- n=500: `gamma + gamma` in 50/100 replications
- n=1000: `gamma + gamma` in 66/100 replications
- n=1500: `gamma + gamma` in 65/100 replications

This suggests either the Gamma/lognormal data-generating setup is too weakly
separated, or the candidate families are too flexible/overlapping for BIC to
distinguish the intended non-identical truth.

## Scenario B: Variable Selection

Penalization helps, but the false-positive burden is still too high for a strong
variable-selection claim.

| Example | Penalty | Mean lambda | MSE | TPR | FPR | FDR | Selected variables |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | lasso | 19.9 | 0.006 | 1.000 | 0.222 | 0.388 | 16.66 |
| 1 | none | 0.0 | 0.005 | 1.000 | 0.998 | 0.750 | 39.95 |
| 2 | lasso | 44.58 | 0.004 | 0.893 | 0.487 | 0.513 | 23.55 |
| 2 | none | 0.0 | 0.008 | 1.000 | 0.998 | 0.750 | 39.95 |
| 4 | lasso | 43.6 | 0.006 | 0.975 | 0.297 | 0.452 | 18.67 |
| 4 | none | 0.0 | 0.006 | 1.000 | 0.997 | 0.749 | 39.92 |

This is usable as a diagnostic, but not yet a polished top-journal selection
result. We need either a stronger penalty-selection rule, an adaptive lasso
variant, stability selection, or a more carefully calibrated sparse simulation.

## Scenario C: Inference

The analytic Louis and numeric SEs agree closely and both have 100% SE-computation
success in the reported examples. This supports the implementation of Louis'
identity, but coverage is only acceptable for Examples 1 and 4.

| Example | n | Method | Mean absolute bias | Coverage | Mean SE | Mean CI length |
|---|---:|---|---:|---:|---:|---:|
| 1 | 500 | Louis | 0.075 | 0.937 | 0.093 | 0.363 |
| 1 | 1000 | Louis | 0.052 | 0.948 | 0.065 | 0.254 |
| 1 | 1500 | Louis | 0.044 | 0.952 | 0.054 | 0.213 |
| 2 | 500 | Louis | 0.150 | 0.499 | 0.040 | 0.157 |
| 2 | 1000 | Louis | 0.108 | 0.587 | 0.032 | 0.127 |
| 2 | 1500 | Louis | 0.104 | 0.530 | 0.024 | 0.094 |
| 4 | 500 | Louis | 0.060 | 0.943 | 0.071 | 0.280 |
| 4 | 1000 | Louis | 0.041 | 0.933 | 0.049 | 0.192 |
| 4 | 1500 | Louis | 0.032 | 0.938 | 0.039 | 0.151 |

Derivative source counts confirm that the run used analytic derivatives:

- `gaussian:analytic`: 3,000,000
- `student_t:analytic`: 3,000,000
- `poisson:analytic`: 3,000,000
- `zip:analytic`: 3,000,000
- `gamma:analytic`: 2,985,000
- `lognormal:analytic`: 2,985,000

## Manuscript Decision

Do **not** insert these tables into `manuscript_draft.tex` as final results.
The current manuscript simulation prose is also too optimistic relative to this
fresh run, especially for Example 2 and for the statement that BIC-selected
models always match the best predictive model.

Recommended next simulation revision:

1. Replace the positive-support Gamma/lognormal scenario with a better-separated
   positive-support design, or explicitly reframe it as a negative/overlap case.
2. For count Example 3, report equivalence-class recovery rather than strict
   family recovery, because NB2 can absorb Poisson-like behavior.
3. Strengthen the sparse variable-selection experiment using adaptive penalties,
   stability selection, or a tuning rule aimed at false discovery control.
4. Keep the Louis-vs-numeric inference comparison for Examples 1 and 4; those
   results are close to publishable.
5. Rerun the final simulation after the revised design is locked, then update
   the manuscript from a single fresh output directory only.
