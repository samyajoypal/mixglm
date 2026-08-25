# Reproducibility Guide

This repository contains the Python implementation, experiment drivers,
compact frozen results, and figure-generation tools for the computational
study.

## Canonical Numerical Record

The small tracked files in `experiments/paper_figures/csda_data/` are the
frozen numerical record for the reported analyses. They include:

- family recovery, top-ten inclusion, criterion agreement, sparsity, and inference summaries;
- unique-family full-data leaderboards and five-split class winners;
- matched log-score, RMSE, and MAE class means for both applications;
- bootstrap tuning, active-count, parameter, and support summaries;
- final model parameters, run metadata, and Louis diagnostics;
- all sixteen analytical-derivative validation checks.

This snapshot is deterministic. Rebuild it from the completed raw output
directories without fitting any model:

```bash
.venv/bin/python experiments/paper_figures/freeze_csda_results.py
```

The raw sources used by the freezing script are the final directories named in
the script. They include the repaired simulation run dated `20260817`, the
Parkinson grouped and final runs dated `20260820`, and the RAND lower-penalty
split and final runs. Older similarly named exploratory directories are not
part of the final analysis record.

Both application searches use `LassoPenalty` for every positive tuning value,
equivalent to a penalty mixing parameter of one. The same lambda is applied to
all components, and lambda zero is the unpenalized endpoint. Ridge and elastic
net are implemented in the package but are not part of the Parkinson or RAND
candidate counts.

## Figures

Generate the five sequential figures from the compact snapshot:

```bash
.venv/bin/python experiments/paper_figures/make_csda_figures.py \
  --outdir paper_outputs/csda_figures
```

The command writes only `fig1.pdf` through `fig5.pdf`. It performs no fitting or
selection.

## Analytical Louis Validation

Run the independent central finite-difference checks with:

```bash
.venv/bin/python experiments/simulations/validate_louis_derivatives.py
```

The validation covers fourteen analytical family implementations in sixteen
branches because ZIP and ZINB are each checked at zero and at a positive count.
The frozen summary is
`experiments/paper_figures/csda_data/louis_validation_summary.csv`.

## Full Experiments

The principal simulation driver is `experiments/simulations/master_run.py`.
The application drivers are:

- `experiments/real_data/run_targeted_hpc_screen.py`
- `experiments/real_data/run_final_inference.py`
- `experiments/real_data/run_rand_final_inference.py`

Matching cluster wrappers are retained beside those drivers. Every final run
records its seeds, tuning grid, iteration limits, initialization budget,
bootstrap size, and selection rule in `run_metadata.json`.

Three reproduction levels should be distinguished:

1. Analytical-derivative validation runs locally from the implementation.
2. Numerical-summary and figure reproduction uses the compact frozen snapshot and takes minutes.
3. Full stochastic reproduction reruns thousands of fits and two 500-sample bootstraps on a cluster.

A fresh stochastic run should agree within Monte Carlo error; it need not be
bit-identical across numerical libraries or parallel schedules.
