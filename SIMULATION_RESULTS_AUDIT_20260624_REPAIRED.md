# Repaired Simulation Results Audit: 2026-06-24

Fresh repaired HPC job `22563194` completed successfully:

- SLURM state: `COMPLETED`, exit code `0:0`
- Runtime: `01:34:45`
- Stderr: empty
- Output root: `paper_outputs/v4_positive_repaired_20260624_b`
- Local copy: `experiments/simulations/paper_outputs/v4_positive_repaired_20260624_b`
- Raw aggregate: `raw_results.json`

## Bottom Line

The repaired simulation is now **usable for the manuscript**, but the paper must
describe the results honestly.

The main failure in the previous run, the positive-support Gamma/lognormal
scenario, is substantially fixed. BIC recovery now increases with sample size:

- n=500: 56% strict recovery
- n=1000: 68% strict recovery
- n=1500: 81% strict recovery

The true Gamma/lognormal model appears in the BIC top-10 leaderboard in 100%,
98%, and 95% of replications, respectively. This supports the top-10 leaderboard
story, but it is not a near-perfect top-1 selection result. We should present it
as a harder positive-support problem where overlapping tail families can compete,
not as a solved-every-time scenario.

## Scenario A: Model Selection

| Example | Truth | n=500 strict/equiv | n=1000 strict/equiv | n=1500 strict/equiv | Assessment |
|---|---|---:|---:|---:|---|
| 1 | Gaussian + Student-t | 0.75 / 0.75 | 0.96 / 0.96 | 0.99 / 0.99 | Strong |
| 2 | Gamma + Lognormal | 0.56 / 0.56 | 0.68 / 0.68 | 0.81 / 0.81 | Usable, but frame as hard |
| 3 | Poisson + NB2 | 0.00 / 0.48 | 0.00 / 0.88 | 0.00 / 0.98 | Use equivalence-class recovery |
| 4 | Poisson + ZIP | 1.00 / 1.00 | 0.99 / 0.99 | 0.99 / 0.99 | Strong |

The BIC-selected model is not always the top held-out likelihood model. The
rate ranges from 0.18 to 0.72 depending on the example and sample size. Do not
claim BIC always matches prediction. Instead, report BIC and prediction
leaderboards separately.

## Top-10 Leaderboard Findings

The repaired run creates:

- `top10_bic_frequency_summary.csv`
- `top10_pred_frequency_summary.csv`
- per-example files in `top10_freq/`
- first-replication case studies in `case_studies/`

For Example 2 at n=1500:

- BIC top-1 is the true Gamma/lognormal family tuple in 81/100 replications.
- The true tuple appears in the BIC top 10 in 95/100 replications.
- The true tuple appears in the prediction top 10 in 95/100 replications.
- In the showcased case-study replication, `gamma + lognormal` ranks first by
  both BIC and held-out log likelihood.

## Scenario B: Variable Selection

Penalization clearly reduces false positives relative to no penalty, but the
result should be phrased as a sparsity/parsimony trade-off rather than as
perfect support recovery.

| Example | Best sparse choice | TPR | FPR | FDR | Selected | Comment |
|---|---|---:|---:|---:|---:|---|
| 1 | Lasso | 1.000 | 0.222 | 0.388 | 16.66 | Good |
| 2 | Lasso | 0.717 | 0.152 | 0.289 | 11.74 | Sparse but loses some signals |
| 4 | Lasso | 0.975 | 0.297 | 0.452 | 18.67 | Useful reduction, not oracle |

For Example 2, the unpenalized model has lower coefficient MSE and better held-out
log likelihood but extremely high FPR/FDR. The lasso has worse prediction but
much better sparsity. This is a legitimate selection trade-off, not a pure
dominance result.

## Scenario C: Inference

Louis and numeric standard errors agree to numerical precision and both have
100% SE success. Analytic derivative blocks are used for all simulated families:

- `gaussian:analytic`
- `student_t:analytic`
- `gamma:analytic`
- `lognormal:analytic`
- `poisson:analytic`
- `zip:analytic`

Coverage:

- Example 1: 0.937, 0.948, 0.952 as n increases.
- Example 2: about 0.907 to 0.910.
- Example 4: about 0.933 to 0.943.

This is publishable if we describe it as finite-sample Wald/Louis behavior:
good in the regular continuous and zero-inflated count examples, mildly
under-covering in the harder positive-support mixture.

## Manuscript Recommendation

These results can be used in the paper, with careful language:

1. Replace the old simulation tables with the repaired `v4_positive_repaired`
   tables.
2. Add a leaderboard table or paragraph using the top-10 frequency summaries.
3. Do not claim BIC is always the best predictive model.
4. For Example 3, emphasize equivalence-class recovery because NB2 can mimic
   Poisson as dispersion approaches zero.
5. For variable selection, state that lasso/enet reduce false discovery and
   model size, not that they achieve exact support recovery.
6. For inference, emphasize analytic Louis stability and finite-sample coverage,
   while acknowledging mild undercoverage in the positive-support case.
