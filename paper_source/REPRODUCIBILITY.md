# Reproducibility Notes

This repository contains the manuscript source, Python implementation, and saved
artifacts used for the current paper draft.

## Main LaTeX sources

- `manuscript_draft.tex`
- `methods_theory_rewrite.tex`
- `appendix_louis_mstep.tex`
- `supplementary_material.tex`
- `references.bib`
- `figures/*.pdf`

Compile from the repository root with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_draft.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

## Figure generation

The paper figures are generated from saved CSV artifacts, not from a fresh model
fit:

```bash
.venv/bin/python experiments/paper_figures/make_paper_figures.py --outdir figures
```

The script reads:

- `experiments/simulations/paper_outputs/v4_positive_repaired_20260624_b/`
- `experiments/real_data/targeted_outputs/v2_less_sparse_k123/`
- `experiments/real_data/targeted_outputs/count_publication_v4/`
- `experiments/real_data/targeted_outputs/rand_final_inference_v1/`

Scenario A saved held-out log score by sample size, not RMSE by sample size.
The current consistency-style figure therefore reports the saved held-out log
score.  A future simulation rerun can add RMSE by sample size without changing
the plotting interface.

## Python implementation

Core implementation files live under `src/mixglm/`:

- `model/mixture_glm.py` and `model/component.py`
- `em/` for responsibilities, initialization, and M-step updates
- `families/` for response-family likelihoods
- `inference/louis.py` and `inference/analytic_blocks.py`
- `selection/` for model spaces, screening, criteria, and beam search
- `penalties/` and `links/`

The analytical Louis validation script is:

```bash
.venv/bin/python experiments/simulations/validate_louis_derivatives.py
```

## Simulation artifacts

The simulation tables in the manuscript use:

```text
experiments/simulations/paper_outputs/v4_positive_repaired_20260624_b/
```

Key files are:

- `table_scenario_A.csv`
- `table_scenario_B.csv`
- `table_scenario_C_main.csv`
- `top10_bic_all_reps.csv`
- `top10_pred_all_reps.csv`
- `scenario_B_lambda_paths.csv`

The simulation driver and aggregation scripts are:

- `experiments/simulations/master_run.py`
- `experiments/simulations/run_sim.py`
- `experiments/simulations/aggregate.py`
- `experiments/simulations/submit_master.sh`

## Real-data artifacts

The Parkinson application uses:

```text
experiments/real_data/targeted_outputs/v2_less_sparse_k123/
```

The RAND split-validation count application uses:

```text
experiments/real_data/targeted_outputs/count_publication_v4/
```

The final RAND full-data refit and bootstrap use:

```text
experiments/real_data/targeted_outputs/rand_final_inference_v1/
```

The real-data preparation and run scripts are:

- `experiments/real_data/fetch_datasets.py`
- `experiments/real_data/prepare_count_datasets.py`
- `experiments/real_data/prepare_openml_datasets.py`
- `experiments/real_data/run_targeted_hpc_screen.py`
- `experiments/real_data/run_rand_final_inference.py`
- `experiments/real_data/submit_publication_count.sh`
- `experiments/real_data/submit_rand_final_inference.sh`

Large checkpoint folders are useful for audit trails but are not required to
compile the manuscript.  They can be regenerated from the scripts above.
