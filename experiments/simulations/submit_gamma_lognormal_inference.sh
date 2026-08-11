#!/bin/bash
#SBATCH --job-name=mixglm_gamma_c
#SBATCH --partition=epyc-256
#SBATCH --output=logs/gamma_c_%j.out
#SBATCH --error=logs/gamma_c_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

module load python/3.10.12-extended 2>/dev/null || true

cd ~/mixglm
source .venv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MIXGLM_OUTPUT_ROOT=paper_outputs/v6_gamma_lognormal_multistart_20260811
export MIXGLM_N_REPS_A=0
export MIXGLM_N_REPS_B=0
export MIXGLM_N_REPS_C=500
export MIXGLM_SAMPLE_SIZES=500,1000,1500,2500
export MIXGLM_SCENARIO_A_EXAMPLES=
export MIXGLM_SCENARIO_B_EXAMPLES=
export MIXGLM_SCENARIO_C_EXAMPLES=2
export MIXGLM_SCENARIO_C_N_STARTS=10
export MIXGLM_SCENARIO_C_MAX_ITER=250
export MIXGLM_INFERENCE_METHODS=louis

mkdir -p logs "$MIXGLM_OUTPUT_ROOT/checkpoints"

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Output: $MIXGLM_OUTPUT_ROOT"
echo "Scenario: Gamma-lognormal Louis inference"
echo "Replicates: $MIXGLM_N_REPS_C per sample size"
echo "Optimization: $MIXGLM_SCENARIO_C_N_STARTS starts, $MIXGLM_SCENARIO_C_MAX_ITER iterations"

python -u experiments/simulations/master_run.py
