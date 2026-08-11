#!/bin/bash
#SBATCH --job-name=mixglm_park_grouped
#SBATCH --partition=epyc-256
#SBATCH --output=logs/park_grouped_%j.out
#SBATCH --error=logs/park_grouped_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

set -euo pipefail

module load python/3.10.12-extended 2>/dev/null || true

cd ~/mixglm
source .venv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MIXGLM_REAL_OUTPUT_ROOT=experiments/real_data/targeted_outputs/v4_parkinsons_grouped
export MIXGLM_REAL_DATASETS=parkinsons_log
export MIXGLM_REAL_INITS=kmeans_glm,quantile_glm
export MIXGLM_REAL_LAMBDAS=0,0.1,0.25,0.5,1,2,5,10,20
export MIXGLM_REAL_K_MIN=1
export MIXGLM_REAL_K_MAX=3
export MIXGLM_REAL_N_TRAIN=3000
export MIXGLM_REAL_N_TEST=1000
export MIXGLM_REAL_P_SCREEN=50
export MIXGLM_REAL_MAX_ITER=180
export MIXGLM_REAL_TOL=1e-3
export MIXGLM_REAL_N_STARTS=2
export MIXGLM_REAL_ACTIVE_THRESHOLD=1e-5
export MIXGLM_REAL_N_SPLITS=5
export MIXGLM_REAL_TEST_FRACTION=0.2
export MIXGLM_REAL_USE_FULL_SAMPLE=1
export MIXGLM_REAL_REFIT_ACTIVE=1
export MIXGLM_REAL_REFIT_MAX_ITER=220
export MIXGLM_REAL_REFIT_N_STARTS=3
export MIXGLM_REAL_MIN_ACTIVE_PER_COMPONENT=1
export MIXGLM_REAL_SEED=20260811

mkdir -p logs "$MIXGLM_REAL_OUTPUT_ROOT/checkpoints"

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Output: $MIXGLM_REAL_OUTPUT_ROOT"
echo "Dataset: $MIXGLM_REAL_DATASETS"
echo "Validation: five participant-grouped splits plus full family leaderboard"
echo "Selection: active-set refit BIC; at least one slope per component"

python -u experiments/real_data/run_targeted_hpc_screen.py
