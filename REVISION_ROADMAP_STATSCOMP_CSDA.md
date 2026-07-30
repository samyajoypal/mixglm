# Revision Roadmap for Statistics and Computing / CSDA

## Target

The current project should be revised for a computational-statistics journal,
with Statistics and Computing as the primary target and Computational Statistics
and Data Analysis (CSDA) as the realistic fallback. The paper should not be
positioned as a JASA Theory and Methods submission unless we add a major new
theoretical contribution on heterogeneous-family identifiability or
post-selection inference.

The main story should be:

1. A general regression-mixture framework allowing component-specific response
   families, links, nuisance parameters, and sparse covariate effects.
2. A reproducible computational search procedure over support-compatible
   heterogeneous family tuples.
3. Closed-form Louis information blocks for the implemented family library.
4. Empirical evidence that heterogeneous-family mixtures improve distributional
   fit and can reveal component-specific covariate structures.

## Working Principles

- Be honest about the fixed-model nature of the main asymptotic theory.
- Separate penalized screening, active support selection, unpenalized refitting,
  and post-selection uncertainty.
- Report unrestricted model-selection results; do not discard models solely
  because they are inconvenient.
- For real-data examples included in the main paper, require at least one
  non-intercept active covariate in every selected component unless the
  intercept-only component is itself the scientific point.
- Prefer lower and moderate penalty grids in the main real-data searches,
  because very large common penalties can overshrink small components.
- Use the same result template for both real-data applications.

## Phase 1: Correctness and Scoring

Deliverable: a corrected, tested core pipeline.

- Audit `mean_from_mu()` for every implemented family and add tests showing that
  `MixtureGLM.predict_mean()` returns the response mean, not merely the linked
  location parameter.
- Recompute all root mean squared error (RMSE) and mean absolute error (MAE)
  summaries after the predictive-mean audit.
- Generalize the post-lasso active-set refit workflow beyond the RAND-specific
  script:
  penalized path -> selected support -> unpenalized masked refit -> refit BIC.
- Keep ridge separate from lasso/elastic net in model-selection summaries,
  because ridge does not select an active set.
- Record, for every candidate, the penalized objective/log likelihood,
  unpenalized log likelihood, refit log likelihood when applicable, active set,
  active counts, BIC, integrated completed likelihood (ICL), held-out log score,
  RMSE, and MAE.

Gate to proceed: local smoke tests pass and the same candidate can be scored
with both penalized and refit-based criteria.

## Phase 2: Penalty Strategy for Real Data

Deliverable: a pre-specified real-data penalty policy.

Current broad real-data screening uses relatively sparse-leaning grids such as
`5,10,20,30,50,100`, and the RAND inference script has included values up to
`1000`. For the publication reruns, use a lower grid first:

`0,0.25,0.5,1,2,5,10,20`

Use the larger penalties only as sensitivity analyses, not as the default main
screen. For each selected model, report active counts by component and flag
intercept-only components.

Gate to proceed: the selected main-paper real-data models should have
non-identical families, stable BIC support, and non-intercept active covariates
in every component.

## Phase 3: Louis Information Validation

Deliverable: a table and figure suitable for the main paper or supplement.

- Run family-by-family derivative validation for all analytical Louis blocks:
  Gaussian, Student-t, Poisson, negative binomial (NB2), Gamma, exponential,
  lognormal, inverse Gaussian, Bernoulli, geometric, beta, zero-inflated
  Poisson (ZIP), zero-inflated negative binomial (ZINB), and skew normal.
- Compare analytical score/Hessian blocks with finite-difference derivatives.
- Compare assembled Louis observed information with a numerical Hessian for
  representative mixtures.
- Report maximum absolute error, relative error, minimum eigenvalue, condition
  number, failure rate, and runtime.

Gate to proceed: analytical derivatives match numerical checks to a documented
tolerance except in clearly explained boundary cases.

## Phase 4: Simulation Rerun

Deliverable: final simulation tables and figures.

- Rerun Scenario A with corrected scoring and report top-1, top-3, top-5,
  top-10, median rank, and BIC gap for true/equivalent models.
- Make equivalence definitions parameter-aware. Report strict recovery as the
  primary metric and boundary-equivalence recovery as secondary.
- Resolve the Poisson-NB2 anomaly by checking larger sample sizes, stronger
  overdispersion, more starts, and refit-based BIC.
- Rerun Scenario B under the corrected penalty policy and report true-positive
  rate, false-positive rate, F1 score, active-set size, BIC, and prediction.
- Rerun Scenario C with enough replications to support coverage claims, ideally
  at least 500 when computationally feasible, and report Monte Carlo standard
  errors.
- Generate final plots: RMSE versus sample size, family recovery versus sample
  size, variable-selection error by penalty, and Louis analytical versus
  numerical standard errors.

Gate to proceed: simulation conclusions remain positive after corrected scoring
and Monte Carlo uncertainty is visibly quantified.

## Phase 5: Real-Data Applications

Deliverable: two matched, publication-grade applications.

For both applications, use the same reporting template:

- data description and scientific question;
- train/test or grouped split design;
- candidate family space and penalty grid;
- BIC/refit-BIC leaderboard;
- held-out log score, RMSE, and MAE;
- mixture weights;
- component-specific active covariates;
- entropy or classification sharpness;
- split/bootstrap stability;
- comparison with best one-component and best identical-family mixtures.

For the continuous application, Parkinson's telemonitoring remains promising but
must be strengthened with grouped participant-level validation or replaced if
the grouped analysis is weak. The nonconverged Student-t competitor must be
resolved with longer runs and more starts.

For the count application, RAND remains the strongest candidate. The next run
should use the lower penalty grid and report whether the NB2 component keeps
non-intercept covariates. If it remains intercept-only under lower penalties,
that result should be treated as a limitation rather than hidden.

Gate to proceed: both applications have non-identical selected models, coherent
leaderboards, meaningful component-specific covariate patterns, and stable
split/bootstrap behavior.

## Phase 6: Computational Benchmarking

Deliverable: the material that makes the paper fit Statistics and Computing.

- Compare exhaustive search and beam search on runtime, memory, convergence,
  selected model, and top-rank stability.
- Compare initialization schemes: random, quantile, k-means, and k-means plus
  one-component GLM warm starts.
- Compare analytical Louis information with numerical Hessian inference on
  runtime, numerical failures, and standard-error agreement.
- Add external comparators where feasible, such as homogeneous-family mixture
  GLMs, single-family GLMs, zero-inflated/NB count regressions, and available
  mixture-regression software.

Gate to proceed: the paper can make a credible computational contribution, not
only an applied modeling claim.

## Phase 7: Manuscript Revision

Deliverable: submission-ready manuscript and supplement.

- Reframe the abstract and introduction for Statistics and Computing/CSDA.
- Add a compact literature-positioning table comparing the proposed framework
  with FlexMix, finite mixtures of GLMs, GAMLSS/distributional-regression
  mixtures, penalized mixture regression, and zero-inflated count models.
- Keep the main fixed-model theory, but avoid overclaiming search-valid
  inference.
- Move lengthy algebra to the appendix/supplement while keeping the main Louis
  result visible.
- Ensure every section and subsection begins with orientation text.
- Use consistent abbreviations and define each abbreviation at first use.
- Use matched real-data tables and plots for both applications.
- Update the reproducibility files and repository reference after final reruns.

Gate to submit: PDFs compile without unresolved references or warnings, all
reported tables/figures are reproducible from committed code, and the main
claims are aligned with the validated results.
