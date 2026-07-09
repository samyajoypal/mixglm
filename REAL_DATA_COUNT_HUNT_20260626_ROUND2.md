# Count real-data hunt, round 2, 2026-06-26

Goal: continue the search for a publishable count-response example in which a non-identical mixture is selected by BIC after a full model search.

## New candidate pool

Prepared by `experiments/real_data/prepare_count_datasets.py`.

Health-utilization and zero-heavy count responses:

- `doctor_visits`: Australian doctor consultations, `n=5190`, zero fraction `0.798`.
- `doctor_nondoctor`: Australian non-doctor consultations, `n=5190`, zero fraction `0.909`.
- `doctor_hospdays`: Australian hospital days, `n=5190`, zero fraction `0.865`.
- `doctor_hospadmi`: Australian hospital admissions, `n=5190`, zero fraction `0.865`.
- `nmes_nvisits`: NMES non-physician office visits, `n=4406`, zero fraction `0.682`.
- `nmes_emergency`: NMES emergency visits, `n=4406`, zero fraction `0.818`.
- `nmes_hospital`: NMES hospital stays, `n=4406`, zero fraction `0.804`.
- `rwm5yr_docvis`: German health panel doctor visits, `n=19609`, zero fraction `0.386`.
- `rwm5yr_hospvis`: German health panel hospital visits, `n=19609`, zero fraction `0.914`.
- `vietnam_pharvis`: Vietnam pharmacy visits, `n=27765`, zero fraction `0.744`.

Other candidates:

- `biochem_articles`: PhD article counts, `n=915`, zero fraction `0.301`.
- `recreation_trips`: recreation-trip counts, `n=659`, zero fraction `0.633`.
- `badhealth_visits`: doctor visits among a bad-health subsample, `n=1127`, zero fraction `0.319`.
- `mdvis_visits`: medical visits, `n=2227`, zero fraction `0.299`.
- `insurance_car_claims`: car insurance claim counts, `n=67856`, zero fraction `0.932`.
- `insurance_singapore_claims`: Singapore auto claim counts, `n=7483`, zero fraction `0.935`.
- `insurance_ohlsson_claims`: motorcycle claim counts, `n=64548`, zero fraction `0.990`.
- `insurance_claims_long`: longitudinal claim counts, `n=120000`, zero fraction `0.857`.

## Local K=1,2 triage

Most candidates selected a single NB2, ZIP/ZINB, or identical mixture by BIC. Three datasets gave non-identical BIC winners in the cheap screen:

### `nmes_nvisits`

- BIC winner: `nb2+zinb`, K=2, `lambda=20`.
- BIC: `4079.932`; held-out log likelihood: `-947.882`; RMSE: `4.660`; MAE: `2.131`.
- Mixing weights: `[0.612, 0.388]`.
- Active counts: `[3, 3]`.
- BIC gap against best identical mixture: `-11.501` BIC units.
- BIC gap against best K=1 model: `-38.901` BIC units.
- Component-specific selected variables:
  - Component 1: `school`, `region_west`, `insurance_yes`.
  - Component 2: `school`, `region_other`, `employed_yes`.

This is the strongest local candidate because it is cross-sectional, has a credible health-utilization interpretation, has a nontrivial BIC margin, and produces sparse component-specific covariate sets.

### `rwm5yr_docvis`

- BIC winner: `nb2+zinb`, K=2, `lambda=50`.
- BIC: `6574.639`; held-out log likelihood: `-1474.243`; RMSE: `6.254`; MAE: `3.091`.
- Mixing weights: `[0.355, 0.645]`.
- Active counts: `[2, 3]`.
- BIC gap against best identical mixture: `-3.492` BIC units.
- Selected variables differ across components (`age`, `outwork` versus `age`, `female`, `kids`).

This is promising but less clean because the source is a repeated panel.

### `rwm5yr_hospvis`

- BIC winner: `poisson+zip`, K=2, `lambda=20`.
- BIC: `1084.286`; held-out log likelihood: `-215.449`; RMSE: `0.353`; MAE: `0.197`.
- Mixing weights: `[0.006, 0.994]`.
- Active counts: `[1, 1]`.
- BIC gap against best identical mixture: `-1.820` BIC units.

This is not a primary candidate unless the full run stabilizes it, because the small component weight is too close to degeneracy.

## Full cluster screen

Submitted job: `22565282`.

Settings:

- Output root: `experiments/real_data/targeted_outputs/count_hunt_v2_health`
- Datasets: `nmes_nvisits,rwm5yr_docvis,rwm5yr_hospvis`
- Families: all count-family combinations from K=1 to K=3.
- Initializations: `kmeans_glm,quantile_glm`
- Penalties: `0,1,2,5,10,20,50,100,200`
- Training/test split: `3000/1000`
- Screened predictors: `20`
- EM iterations: `180`
- Starts per fit: `2`

Decision rule: the count example is publishable only if the full screen selects a non-identical mixture by BIC with a stable mixing-weight vector, a meaningful margin over the best identical mixture, and an interpretable active-variable pattern.

## Full health-screen result

Job `22565282` completed successfully.  The full K=1,2,3 confirmation did not yield a publishable count example.

- `nmes_nvisits`: BIC selected identical `nb2+nb2`, K=2, `lambda=20`, BIC `8119.874`.  The best non-identical model was `nb2+zinb`, K=2, `lambda=20`, BIC `8160.715`, losing to the best identical mixture by `40.842` BIC units.
- `rwm5yr_docvis`: BIC selected identical `nb2+nb2`, K=2, `lambda=20`, BIC `13264.868`.  The best non-identical model was `nb2+nb2+zip`, K=3, `lambda=10`, BIC `13312.084`, losing by `47.216` BIC units.
- `rwm5yr_hospvis`: BIC selected identical `nb2+nb2`, K=2, `lambda=200`, BIC `2366.440`.  The best non-identical model was `zip+zinb`, K=2, `lambda=200`, BIC `2366.909`, losing by `0.469` BIC units.

The hospital-visit outcome came close, but the criterion still favors an identical mixture.  We should not use it as a positive result.

## Additional round-3 candidates

Prepared additional large or classical count datasets:

- `county_murders`: county-year murder counts, `n=37349`, zero fraction `0.419`.
- `randhealth_notmdvis`: RAND non-physician visits, `n=20190`, zero fraction `0.828`.
- `randhealth_mentvis`: RAND mental-health visits, `n=20190`, zero fraction `0.965`.
- `randhealth_totadm`: RAND hospital admissions, `n=20190`, zero fraction `0.910`.
- `webworms_count`: webworm counts in a beet-field experiment, `n=1300`, zero fraction `0.547`.
- `bird_counts`: bird counts, `n=18706`, zero fraction `0.621`.
- `crime1_arrests`: individual arrest counts, `n=2725`, zero fraction `0.723`.
- `crime1_felony_arrests`: felony arrest counts, `n=2725`, zero fraction `0.820`.
- `patents_1979`: firm patent counts, `n=346`, zero fraction `0.220`.

Local K=1,2 triage eliminated several datasets:

- `county_murders`: identical `nb2+nb2` won BIC.
- `webworms_count`: single `zip` won BIC.
- `crime1_arrests`: single `nb2` won BIC.
- `crime1_felony_arrests`: single `nb2` won BIC.
- `randhealth_mentvis`: available partial/narrow evidence favored single `nb2`.

Two candidates were retained for full confirmation:

- `randhealth_notmdvis`: local BIC winner `poisson+nb2`, K=2, `lambda=50`, BIC `2321.731`; it beats the best identical mixture by `22.793` BIC units.  Caveat: the selected fit is intercept-only in both components.
- `bird_counts`: local partial BIC winner `nb2+zinb`, K=2, `lambda=5`, BIC `7909.206`; it beats the best identical mixture by `23.403` BIC units.  Caveat: the best-BIC fit has unstable RMSE, so the full screen must be inspected for predictive stability.

Submitted full confirmation job `22565322`.

Settings:

- Output root: `experiments/real_data/targeted_outputs/count_hunt_v3_round3`
- Datasets: `randhealth_notmdvis,bird_counts`
- Families: all count-family combinations from K=1 to K=3.
- Initializations: `kmeans_glm,quantile_glm`
- Penalties: `0,1,2,5,10,20,50,100,200`
- Training/test split: `3000/1000`
- Screened predictors: `50`
- EM iterations: `180`
- Starts per fit: `2`

## Full round-3 result

Job `22565322` completed successfully on 2026-06-26 (exit code 0; elapsed time
5:43:22). All 1,224 fits converged without recorded errors. Results are in
`experiments/real_data/targeted_outputs/count_hunt_v3_round3`.

### `randhealth_notmdvis`

The BIC winner is a non-identical `poisson+nb2` mixture with K=2,
`lambda=50`, and `kmeans_glm` initialization:

- BIC `4358.112`, training log likelihood `-2143.027`;
- held-out log likelihood `-745.357`, RMSE `5.407`, and MAE `1.411`;
- mixing weights `[0.811, 0.189]`;
- active counts `[5,0]`;
- selected Poisson-component variables: age, race indicator, log
  coinsurance, sex, and one site indicator;
- the NB2 component is intercept-only.

The winner improves BIC by `287.936` units over the best K=1 model and by
`15.807` units over the best identical-family mixture. The same family,
penalty, and active-set size under `quantile_glm` has BIC `4365.174`, so the
structural result is not tied to only one initialization. Against the best
identical BIC model (`nb2+nb2`, BIC `4373.919`), however, held-out log
likelihood is slightly worse (`-745.357` versus `-744.228`). The non-identical
winner does improve held-out log likelihood over the best K=1 model
(`-778.717`).

This is a credible BIC result with stable weights and a sparse,
component-specific regression structure. It is not yet a final paper result:
the source contains repeated annual records for each RAND participant, but the
screen used an observation-level random split and removed the person ID.
Consequently, the reported predictive comparison does not respect the data's
clustering, and the likelihood analysis treats correlated rows as independent.

### `bird_counts`

The BIC winner is a non-identical `nb2+zinb` mixture with K=2, `lambda=1`,
and `quantile_glm` initialization:

- BIC `14777.406`, training log likelihood `-6964.365`;
- held-out log likelihood `-2311.036`, RMSE `2096.637`, and MAE `202.361`;
- mixing weights `[0.445,0.555]`;
- active counts `[50,50]`.

The winner improves BIC by `337.146` units over the best K=1 model and by
`24.134` units over the best identical-family mixture. It also improves all
three held-out scores relative to those BIC-selected comparators. Relative to
the best identical BIC model, the held-out log likelihood improves by `12.995`,
RMSE by `12.6%`, and MAE by `8.4%`. Relative to the best K=1 BIC model, the
corresponding improvements are `95.668`, `18.9%`, and `22.1%`.

This is the stronger numerical win, and the nearby `lambda=2` fit finds the
same `nb2+zinb` family with similar weights and scores. Nevertheless, it is not
ready for the manuscript. All 50 screened predictors are active in both
components, so the fit supplies no variable-selection result. More
importantly, feature screening was performed using the full response before
the train/test split, and the selected features are largely species indicators.
The data are repeated species-by-year counts with person-hours of observation;
the current analysis uses both hours and log-hours as estimated covariates
rather than treating log-hours as an exposure offset. The response is also
extreme (test maximum `65000`; test standard deviation `2482.9`), and 15 fits
produced infinite RMSE. These facts require a cleaner design and stability
analysis.

## Publication decision

The full screen has found two real non-identical BIC wins, rather than another
negative result. They should be treated as leads, not final evidence. No values
from this run should yet be inserted into the manuscript.

Before choosing the count example, rerun a confirmatory analysis in which:

1. feature screening is learned from the training data only;
2. RAND participants are kept wholly within training or test folds;
3. the bird analysis uses a defensible effort adjustment, preferably a
   log-hours offset, and a prespecified predictor set;
4. the complete available sample is used rather than a 4,000-row subsample;
5. selection and prediction are checked over repeated grouped splits, with
   initialization stability and finite-prediction diagnostics reported.

At this stage `randhealth_notmdvis` is the better variable-selection example,
whereas `bird_counts` is the better family-selection and prediction example.
The final count dataset should be selected only after the corrected analyses
show which conclusion is reproducible.

## Corrected publication run

The confirmatory pipeline was rebuilt on 2026-06-30 and submitted as Slurm job
`22570387`. Its output root is
`experiments/real_data/targeted_outputs/count_publication_v4`.

The earlier results are not reused as final evidence. The new run makes the
following corrections:

1. Predictor screening is performed separately within each training set. No
   held-out response is used for screening or model fitting.
2. The primary RAND analysis is a baseline cross-section containing one record
   for each of 5,912 participants. This avoids treating repeated annual records
   as independent likelihood contributions.
3. Bird records without recorded positive observation effort are excluded
   (`3,781` of `18,706` rows). The remaining `14,925` counts use log person-hours
   as a fixed offset. Hours and log-hours are no longer estimated or penalized
   as ordinary covariates.
4. Every split uses the complete available dataset. RAND splits contain
   4,730 training and 1,182 test participants. Bird splits hold out 15 complete
   years (2,985 rows) and train on 60 years (11,940 rows), with zero group
   overlap.
5. Three independently seeded grouped splits are analyzed. All count-family
   combinations for K=1,2,3 are fitted under both `kmeans_glm` and
   `quantile_glm` initialization, with two starts per fit.
6. The full-sample penalty grid is
   `0,2,5,10,25,50,100,250,500,1000`. This spans the effective penalty range
   used in the earlier subsample after accounting for the larger training
   samples.
7. Outputs include split-specific BIC and prediction leaderboards, best K=1,
   identical-family, and non-identical-family comparators, model and family
   win frequencies, BIC-gap stability, held-out log score per observation,
   RMSE/MAE skill, mixing weights, nuisance estimates, active sets, and explicit
   finite- and stable-prediction diagnostics.

The run contains 4,080 fits. A result will be admitted to the manuscript only
if a non-identical family structure wins reproducibly across splits, has
nondegenerate component weights, yields stable predictions, and retains a
scientifically interpretable component-specific active-set pattern.

## Corrected publication-run result

Slurm job `22570387` completed successfully on 2026-07-02 (exit code 0;
elapsed time 2 days 5:49:52). All 4,080 fits completed and were marked
converged. Results are stored in
`experiments/real_data/targeted_outputs/count_publication_v4`.

### RAND baseline cross-section: retain

The non-identical `poisson+nb2` mixture was the BIC winner in all three
independent participant splits. Its split-specific results were:

| Split | Lambda | BIC | Delta BIC vs K=1 | Delta BIC vs identical | Weights | Active counts |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50 | 6434.902 | -396.649 | -14.508 | (0.831, 0.169) | (12, 0) |
| 1 | 100 | 6472.066 | -483.083 | -11.317 | (0.831, 0.169) | (4, 0) |
| 2 | 100 | 6352.963 | -406.977 | -3.471 | (0.836, 0.164) | (5, 0) |

The family structure therefore won three of three splits, with mean BIC
advantages of `429.0` units over the best one-component model and `9.77` units
over the best identical-family mixture. All winning predictions passed the
stability diagnostic. Held-out log likelihood was substantially better than
the BIC-selected K=1 model in every split, but was essentially tied with the
best identical mixture. RMSE and MAE did not improve meaningfully; this example
supports distributional and latent-class modeling rather than superior point
prediction.

The mixing proportions and nuisance estimates were stable. The Poisson
component represented approximately 83% of participants and carried the
covariate effects. The approximately 17% NB2 component was intercept-only and
strongly overdispersed, with estimated alpha values `8.75`, `9.11`, and `11.19`.
Age, log coinsurance, and the site-3 indicator were selected in every split;
child status and the site-6 indicator appeared in two splits. The signs of the
recurrently selected coefficients were stable.

This is suitable as the primary count-response illustration, subject to a
final all-data fit and uncertainty analysis. A necessary qualification is that
Poisson is the zero-dispersion boundary of NB2. Indeed, the first dispersion
parameter in the best `nb2+nb2` comparators approached zero. Thus, the result
shows reproducible selection of a parsimonious Poisson component together with
a highly overdispersed NB2 component; it should not be presented as evidence
that two completely unrelated count families are required.

### Bird counts: reject

Bird-family selection was not reproducible. The three split winners were
`zinb+zinb+zinb`, `nb2+nb2+zinb`, and `nb2+nb2+nb2`; hence a non-identical
family structure won only one of three splits. The best non-identical model
lost by `356.7` BIC units in split 0 and `16.7` units in split 2. The leading
fits also retained nearly all 50 screened predictors, one split-0 component
had weight `0.030`, and many alternative fits produced explosive or overflowing
predictive means. This dataset should not appear as a positive example in the
paper.

### Publication decision

Use `randhealth_notmdvis_baseline` as the count example and retain
`parkinsons_log` as the continuous example. Do not use `bird_counts`. Before
writing the final real-data section, fit the selected RAND model to all 5,912
participants, report the complete-data parameter estimates and selected
variables, and obtain uncertainty using a selection-aware bootstrap or a
clearly labeled post-selection refit. Ordinary Louis/Wald standard errors from
the penalized fit alone are not sufficient for confirmatory inference.

## Final RAND estimation and inference job

Slurm job `22578683` was submitted on 2026-07-03. Its output root is
`experiments/real_data/targeted_outputs/rand_final_inference_v1`.

The job has two stages:

1. An exhaustive all-data leaderboard fits all 680 combinations generated by
   K=1,2,3, the four count families, ten penalty values, and two initialization
   schemes to all 5,912 baseline participants.
2. Conditional on the `poisson+nb2` family structure already validated in
   three independent train/test splits, 500 participant bootstrap samples each
   repeat predictor screening, penalty tuning, component-specific support
   selection, and an unpenalized constrained post-lasso refit.

The full-data constrained refit will also receive analytical Louis standard
errors after reducing the information matrix to the free coefficients. These
are conditional/model-based intervals. The bootstrap percentile intervals and
selection frequencies are the primary uncertainty analysis because they
include screening, tuning, and active-set uncertainty. Family-selection
uncertainty is documented separately by the completed three-split exhaustive
validation experiment; it is not repeated within every bootstrap sample.

## Final RAND inference result

Slurm job `22578683` completed successfully on 2026-07-04 (exit code 0;
elapsed time 12:08:48; maximum resident memory 2.08 GB). All 680 full-data fits
converged without recorded errors, and all 500 bootstrap replicates produced a
penalized fit and constrained refit.

### Full-data selection

The full-data BIC winner was again the non-identical `poisson+nb2` mixture,
initialized by `kmeans_glm` and fitted at `lambda=100` (log likelihood
`-3931.885`, BIC `7950.617`). It beat the best one-component model by `554.241`
BIC units and the best identical-family mixture, `nb2+nb2`, by `4.585` units.
The latter fitted its first dispersion parameter essentially at the Poisson
boundary (`log(alpha)=-8.518`), reinforcing the interpretation of a
parsimonious Poisson-like component rather than evidence against the broader
NB2 family. The second-ranked heterogeneous family structure was
`poisson+zinb`, `3.777` BIC units behind the winner. Thus, the full-data margin
among two-component count structures is moderate, while the evidence against a
single component is overwhelming. The three independent split analyses remain
the primary evidence for reproducibility of the selected family pair.

The penalized winner assigned weights `(0.841, 0.159)` to the Poisson and NB2
components, respectively. Six covariates were active in the Poisson component
(`xage`, `child`, `disea`, `black`, `logc`, and `site_3`), while the NB2
component was intercept-only. After an unpenalized refit conditional on this
support, the weights were `(0.841, 0.159)` and the NB2 dispersion was
`alpha=10.355`. The Poisson coefficients were `0.221` for age, `-0.191` for
child status, `0.127` for disease burden, `-0.136` for Black race, `-0.319` for
log coinsurance, and `0.210` for site 3.

### Conditional Louis information

The reduced analytical Louis information matrix had dimension and rank 10,
minimum eigenvalue `1.280`, and condition number `1439`. Both component
derivative blocks were analytical (`poisson:analytic` and `nb2:analytic` for
all 5,912 observations). Conditional on the selected support and family pair,
the Louis 95% intervals excluded zero for all six Poisson coefficients. These
intervals are model-based post-refit summaries; they do not account for
screening, penalty tuning, or active-set selection and must not be presented as
unconditional post-selection confidence intervals.

### Selection-aware bootstrap

All 500 participant bootstrap samples repeated screening, penalty tuning,
support selection, and the constrained unpenalized refit with the family pair
held fixed. The NB2-component weight was `0.159` in the full data, with
bootstrap standard error `0.037` and percentile interval `(0.132, 0.191)`.
The NB2 dispersion estimate was `10.355`, with bootstrap standard error `2.285`
and percentile interval `(4.330, 12.159)`. Three replicates reached a distinct
mode with a majority NB2 component, but these isolated cases did not affect the
central 95% intervals.

Covariate stability was heterogeneous:

| Poisson covariate | Selection frequency | Full-data estimate | Bootstrap SE | Percentile interval |
|---|---:|---:|---:|---:|
| `xage` | 1.000 | 0.221 | 0.079 | (0.094, 0.400) |
| `site_3` | 0.988 | 0.210 | 0.058 | (0.003, 0.247) |
| `logc` | 0.886 | -0.319 | 0.148 | (-0.369, 0.147) |
| `disea` | 0.868 | 0.127 | 0.045 | (0.000, 0.152) |
| `child` | 0.796 | -0.191 | 0.173 | (-0.288, 0.000) |
| `black` | 0.682 | -0.136 | 0.084 | (-0.254, 0.000) |

The intervals include a point mass at zero whenever a covariate is omitted by
the resampled selection pipeline. Age and site 3 are therefore the two most
robust covariate findings. The signs of child status and Black race were stable
when selected, disease burden was predominantly positive, and log coinsurance
showed appreciable sign instability.

Penalty and support selection were less stable than the family structure.
The bootstrap chose `lambda=50` in 67.4% of samples, `lambda=100` in 19.6%, and
`lambda=25` in 13.0%. Consequently, the median active-set sizes were 16 and 2
for the Poisson and NB2 components, compared with 6 and 0 in the full-data
fit. Several variables absent from the full-data support had substantial
selection frequencies. The six-variable support should therefore be described
as the full-data BIC-selected representation, not as a uniquely determined
scientific model.

### Publication assessment

The RAND example is suitable for the manuscript as evidence that allowing
component-specific count families can identify a reproducible, parsimonious
Poisson-like majority component and a strongly overdispersed NB2 minority
component. The family result is supported by three held-out split analyses and
the exhaustive full-data leaderboard. The variable-selection analysis is
publishable when reported through selection frequencies and bootstrap
intervals, with age and site 3 emphasized as stable and the remaining effects
described cautiously. Claims of uniformly stable support or unconditional
Wald inference would not be supported by these results.
