#!/bin/bash
#SBATCH --job-name=mixglm_count_final
#SBATCH --partition=downtime-24c
#SBATCH --output=logs/count_final_%j.out
#SBATCH --error=logs/count_final_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=20
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

export MIXGLM_REAL_OUTPUT_ROOT=experiments/real_data/targeted_outputs/count_publication_v4
export MIXGLM_REAL_DATASETS=randhealth_notmdvis_baseline,bird_counts
export MIXGLM_REAL_INITS=kmeans_glm,quantile_glm
export MIXGLM_REAL_LAMBDAS=0,2,5,10,25,50,100,250,500,1000
export MIXGLM_REAL_K_MIN=1
export MIXGLM_REAL_K_MAX=3
export MIXGLM_REAL_P_SCREEN=50
export MIXGLM_REAL_MAX_ITER=200
export MIXGLM_REAL_TOL=1e-3
export MIXGLM_REAL_N_STARTS=2
export MIXGLM_REAL_ACTIVE_THRESHOLD=1e-5
export MIXGLM_REAL_N_SPLITS=3
export MIXGLM_REAL_TEST_FRACTION=0.2
export MIXGLM_REAL_USE_FULL_SAMPLE=1
export MIXGLM_REAL_SEED=20260630

mkdir -p logs "$MIXGLM_REAL_OUTPUT_ROOT/checkpoints"

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Python: $(python --version)"
echo "Output: $MIXGLM_REAL_OUTPUT_ROOT"
echo "Datasets: $MIXGLM_REAL_DATASETS"
echo "Lambdas: $MIXGLM_REAL_LAMBDAS"
echo "Repeated grouped splits: $MIXGLM_REAL_N_SPLITS"
echo "Full sample: $MIXGLM_REAL_USE_FULL_SAMPLE"

python -u experiments/real_data/run_targeted_hpc_screen.py
