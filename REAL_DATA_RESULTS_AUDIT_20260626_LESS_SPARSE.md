# Real-data results audit, less-sparse run, 2026-06-26

## Job status

The revised less-sparse real-data screen completed successfully and was fetched locally.

- Job id: `22564966`
- State: `COMPLETED`
- Exit code: `0:0`
- Runtime: `14:45:25`
- Output root: `experiments/real_data/targeted_outputs/v2_less_sparse_k123`
- Mixture sizes: K=1,2,3
- Lambda grid: `0,0.1,0.25,0.5,1,2,5,10,20`

The run records selected feature sets, original-scale selected coefficients, component intercepts,
number of EM iterations, and minimum component weight.

## Overall decision

The run is a major improvement for diagnosing the real-data story, but it is not yet final
manuscript evidence for a top statistics journal. We now have two plausible primary examples:

- Count example: `blog`
- Continuous example: `parkinsons_log`

The other datasets should not be primary examples in the paper without additional work:

- `rand`: K=1 NB2 wins BIC once K=1 baselines are included.
- `crime_beta`: non-identical mixtures are competitive, but the raw BIC winner is prediction-unstable
  and activates all 40 screened variables in every component.
- `super_raw`: the raw BIC winner is prediction-unstable, and stable BIC/RMSE choices are mostly
  identical Gamma mixtures.

## Candidate 1: BlogFeedback count response

The count example is promising but needs a final stability run and feature-name repair.

- Overall BIC winner: `nb2+zinb`, K=2, lambda=10, `quantile_glm`
- BIC: 3527.788
- BIC advantage over best K=1 model: 27.836
- BIC advantage over best identical-family mixture: 3.451
- Component weights: `[0.374, 0.626]`
- Active counts: `[2, 7]`
- Shared active variables: 1
- Symmetric-difference active variables: 7

The BIC winner has poor RMSE (`930.851`), so it should not be presented as a mean-prediction
winner. However, the best RMSE and MAE model is also a non-identical NB2/ZINB mixture:

- RMSE/MAE winner: `nb2+zinb`, K=2, lambda=20, `quantile_glm`
- BIC: 3564.781
- RMSE: 40.095
- MAE: 8.104
- Component weights: `[0.481, 0.519]`
- Active counts: `[5, 3]`
- Shared active variables: 2
- Symmetric-difference active variables: 4

This gives a defensible story if we present BIC and prediction leaderboards as complementary:
non-identical NB2/ZINB mixtures are selected by BIC and by mean-prediction metrics, but the
preferred penalty differs. The current obstacle is that BlogFeedback feature names are generic
(`x52`, `x51`, etc.) in the saved design.

## Candidate 2: Parkinsons telemonitoring log response

This is the strongest real-data example from the less-sparse run.

- BIC winner: `student_t+skew_normal+skew_normal`, K=3, lambda=20, `quantile_glm`
- BIC: 1452.180
- BIC advantage over best K=1 model: 416.380
- BIC advantage over best identical-family mixture: 43.378
- Component weights: `[0.475, 0.229, 0.296]`
- Active counts: `[14, 12, 16]`
- Shared active variables: 11
- Symmetric-difference active variables: 6
- RMSE: 0.392
- MAE: 0.305

The best held-out log-likelihood model is also non-identical:

- `gaussian+gaussian+skew_normal`, K=3, lambda=20, held-out log likelihood `-286.768`

The best RMSE model is also non-identical, though the RMSE margin over identical mixtures is tiny:

- `gaussian+student_t`, K=2, lambda=5, RMSE `0.381814`

Variable selection is meaningful because feature names are available. The selected sets involve
age, sex, test time, jitter, shimmer, NHR/HNR, RPDE, DFA, and PPE, with component-specific
differences rather than empty components.

## Datasets not recommended as primary examples

`rand` should not be primary: the best BIC model is K=1 NB2 with BIC 8768.717. The best
non-identical BIC model, NB2/ZINB, has better RMSE and MAE but is worse by about 22 BIC units.

`crime_beta` should not be primary in the current form: the raw BIC winner
`student_t+lognormal+lognormal` has active counts `[40, 40, 40]`, a small component, and explosive
RMSE. After filtering unstable/all-active fits, the best model is identical `beta+beta+beta`; the
best filtered non-identical competitor is `beta+lognormal`, but it does not win BIC.

`super_raw` should not be primary: the raw non-identical BIC winner has a tiny lognormal component,
all variables active, and explosive RMSE. After filtering, identical Gamma mixtures dominate BIC/RMSE.

## Recommended next step

Do a final focused real-data run rather than writing the section immediately.

1. Keep `blog` and `parkinsons_log` as the two candidate paper examples.
2. Reconstruct human-readable feature names for BlogFeedback.
3. Run repeated train/test splits or cross-validation only for focused model families and lambdas:
   - BlogFeedback: NB2, ZINB, ZIP, Poisson, NB2/ZINB, Poisson/NB2/ZINB, and nearby identical controls.
   - Parkinsons: Gaussian, Student-t, skew-normal, and the leading two- and three-component mixtures.
4. Report BIC, held-out log likelihood, RMSE/MAE, component weights, active feature sets, and selection
   frequencies over splits.
5. Use held-out log likelihood as the primary predictive score for model density; report RMSE/MAE as
   secondary mean-prediction summaries.
