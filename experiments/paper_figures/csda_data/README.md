# Frozen CSDA Result Record

These files are the compact numerical record used by `csda_submission/main.tex`,
`csda_submission/supplementary.tex`, and `make_csda_figures.py`. They contain
only processed summaries, not exploratory runs or model checkpoints.

Regenerate them from the completed experiment directories with:

```bash
.venv/bin/python experiments/paper_figures/freeze_csda_results.py
```

Then regenerate the five article figures with:

```bash
.venv/bin/python experiments/paper_figures/make_csda_figures.py \
  --outdir csda_submission
```

The freezing script applies the declared convergence, finite-prediction, and
active-component rules before retaining full-data unique-family leaders and the
three model-class winners in each split. The JSON files preserve final model,
Louis-diagnostic, and run-metadata details for both applications.
