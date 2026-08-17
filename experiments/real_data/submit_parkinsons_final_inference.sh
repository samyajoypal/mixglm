#!/bin/bash
#SBATCH --job-name=mixglm_park_infer
#SBATCH --partition=epyc-256
#SBATCH --output=logs/park_infer_%j.out
#SBATCH --error=logs/park_infer_%j.err
#SBATCH --time=20-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

module load python/3.10.12-extended 2>/dev/null || true

cd ~/mixglm
source .venv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MIXGLM_INFERENCE_OUTPUT_ROOT=experiments/real_data/targeted_outputs/v5_parkinsons_final_inference
export MIXGLM_INFERENCE_DATASET=parkinsons_log
export MIXGLM_INFERENCE_P_SCREEN=50
export MIXGLM_INFERENCE_LAMBDAS=0,0.1,0.25,0.5,1,2,5,10,20
export MIXGLM_INFERENCE_INITS=kmeans_glm,quantile_glm
export MIXGLM_INFERENCE_MAX_ITER=220
export MIXGLM_INFERENCE_TOL=1e-3
export MIXGLM_INFERENCE_LEADERBOARD_STARTS=2
export MIXGLM_INFERENCE_BOOTSTRAP_STARTS=1
export MIXGLM_INFERENCE_REFIT_STARTS=2
export MIXGLM_INFERENCE_ACTIVE_THRESHOLD=1e-5
export MIXGLM_INFERENCE_MIN_ACTIVE_PER_COMPONENT=1
export MIXGLM_INFERENCE_BOOTSTRAP_REPS=500
export MIXGLM_INFERENCE_BOOTSTRAP_FAMILIES=auto
export MIXGLM_INFERENCE_LEADERBOARD_FAMILIES=gaussian,student_t,skew_normal
export MIXGLM_INFERENCE_LEADERBOARD_K_MAX=3
export MIXGLM_INFERENCE_SEED=20260817

mkdir -p logs "$MIXGLM_INFERENCE_OUTPUT_ROOT/full_checkpoints" \
  "$MIXGLM_INFERENCE_OUTPUT_ROOT/bootstrap_checkpoints"

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Output: $MIXGLM_INFERENCE_OUTPUT_ROOT"
echo "Dataset: $MIXGLM_INFERENCE_DATASET"
echo "Lambda grid: $MIXGLM_INFERENCE_LAMBDAS"
echo "Bootstrap: $MIXGLM_INFERENCE_BOOTSTRAP_REPS participant-cluster replicates"
echo "Family rule: best full-data nonidentical model with slopes in every component"

python -u experiments/real_data/run_final_inference.py
