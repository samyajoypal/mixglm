# Real-data results audit, 2026-06-25

## Job status

The targeted real-data screen finished successfully on the cluster.

- Job id: `22563184`
- Job name: `mixglm_real_screen`
- State: `COMPLETED`
- Exit code: `0:0`
- Runtime: `07:42:25`
- Output root: `experiments/real_data/targeted_outputs/v1_k23_family_means`

The run used K=2,3 mixtures, two initializations (`kmeans_glm`, `quantile_glm`), lambda grid
`5,10,20,30,50,100`, `n_train=2000`, `n_test=1000`, `p_screen=40`, `max_iter=120`,
`tol=1e-3`, and two EM starts.

## Publishability decision

The screen is valuable, but the real-data section is not yet final-manuscript ready under the
strict criterion that the same non-identical model should be the leading model by BIC and prediction
with a meaningful component-specific variable-selection interpretation.

### Count example

The best current count candidate is `rand`.

- Best BIC model: `nb2+zinb`, K=2, lambda=50, `kmeans_glm`
- BIC: 8656.033
- Held-out log likelihood: -2146.576, rank 1 among finite converged fits
- RMSE: 3.9089, rank 2; the RMSE winner is `nb2+zip` with RMSE 3.9073
- Component weights: `[0.904, 0.096]`
- Active counts: `[11, 0]`; shared active variables: 0; symmetric difference: 11

This is close to publishable statistically: the exact `nb2+zinb` model wins BIC and held-out
log likelihood, and is essentially tied for RMSE. It supports a clear distributional story:
one overdispersed count component and one zero-inflated overdispersed component, with covariates
selected only in the larger component. The main weakness is interpretability: the saved RAND design
does not include human-readable names for all screened columns, so the current variable-selection
table would contain generic labels such as `x6` rather than scientific covariates.

`blog` should not be used as a primary count example. It has a non-identical BIC winner
(`nb2+zinb`), but the BIC-selected model has poor RMSE rank (171/244), and many nearby models show
extreme prediction instability.

### Continuous or bounded example

The best current continuous/bounded candidate is `crime_beta`, but it is not yet clean enough for
the final real-data story if the same exact model must win all criteria.

- Best BIC model: `beta+student_t+lognormal`, K=3, lambda=100, `quantile_glm`
- BIC: -2055.685
- BIC advantage over best identical-family competitor: about 113 BIC units
- Held-out log likelihood: 1022.697, rank 88
- RMSE: 0.1723, rank 518
- Component weights: `[0.041, 0.048, 0.911]`
- Active counts: `[0, 40, 10]`; shared active variables: 0; symmetric difference: 40

As a model-selection/density example, `crime_beta` is promising because non-identical mixtures lead
BIC, held-out log likelihood, and RMSE as model classes. However, the exact leading models differ:
the held-out log-likelihood winner is `beta+beta+lognormal`, and the RMSE winner is `beta+student_t`.
The BIC-selected model also has two small components, which may be scientifically meaningful tail
components but needs a stability check.

`parkinsons_log` should not be used as a primary example. Its BIC winner is non-identical, but the
BIC gap over an identical Student-t mixture is only about 3.4, prediction is better for simpler
two-component models, and held-out log likelihood is led by an identical Gaussian mixture.

`super_raw` should not be used as a primary example. Identical Gamma mixtures win BIC and RMSE.
Non-identical models only lead held-out log likelihood.

## What must be fixed before writing the real-data section

1. Add K=1 baselines to the confirmatory run. The current targeted screen used only K=2,3. For a
   top-journal submission, single GLMs must be reported as baselines even if the primary competition
   is among mixtures.

2. Recover and save real feature names for RAND and Communities and Crime before final variable
   selection tables are produced. The current targeted prepared files contain generic fallback names
   for these two datasets.

3. Store selected feature names and coefficients in each checkpoint. The present checkpoint format
   stores only active counts, shared active counts, and symmetric differences.

4. Run a confirmatory real-data analysis with repeated train/test splits or cross-validation. The
   current run is a single split and should be treated as screening evidence.

5. Use both a proper predictive score and a mean-prediction score. Recommended: held-out log
   likelihood as the primary predictive score, with RMSE/MAE as secondary summaries.

6. For `crime_beta`, add a sensitivity run with a minimum component-weight diagnostic or constraint
   and possibly a post-selection refit. The current BIC winner has two components below 5 percent.

## Current recommendation

Use `rand` as the leading count-data candidate, pending feature-name recovery and a confirmatory
baseline-inclusive run. Treat `crime_beta` as the leading continuous/bounded candidate, but do not
write it as final evidence yet. If the strict requirement is that the same exact non-identical
model wins BIC and RMSE, we need another targeted continuous-data screen or a refined `crime_beta`
run before the real-data section is publishable.

## Less-sparse rerun submitted

A revised confirmatory screen was submitted on 2026-06-25 because the first targeted run selected
overly sparse models under a lambda grid beginning at 5.

- Job id: `22564966`
- Output root: `experiments/real_data/targeted_outputs/v2_less_sparse_k123`
- Datasets: `rand,blog,super_raw,parkinsons_log,crime_beta`
- Mixture sizes: K=1,2,3
- Lambda grid: `0,0.1,0.25,0.5,1,2,5,10,20`
- Initializations: `kmeans_glm,quantile_glm`
- Maximum EM iterations: 160
- Active threshold: `1e-5`

The rerun also records selected feature indices, selected feature names, original-scale selected
coefficients, component intercepts, number of EM iterations, and minimum component weight in each
checkpoint/leaderboard row. The intent is to recover less aggressively sparse fits and make the
variable-selection evidence directly inspectable.
