#!/bin/bash
#SBATCH --job-name=mixglm_master
#SBATCH --partition=downtime-24c
#SBATCH --output=logs/master_%j.out
#SBATCH --error=logs/master_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G

# LRZ specific setup
module load python/3.10.12-extended 2>/dev/null || true

cd ~/mixglm || exit 1

source .venv/bin/activate
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# CRITICAL: Since master_run.py uses joblib to spawn 20 independent Python workers,
# we MUST restrict NumPy/OpenBLAS to 1 thread per worker to prevent CPU thrashing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MIXGLM_OUTPUT_ROOT=${MIXGLM_OUTPUT_ROOT:-paper_outputs/v3_clean_louis}
export MIXGLM_INFERENCE_METHODS=${MIXGLM_INFERENCE_METHODS:-louis,numeric}

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "SLURM CPUs: $SLURM_CPUS_PER_TASK"
echo "MIXGLM_OUTPUT_ROOT: $MIXGLM_OUTPUT_ROOT"
echo "MIXGLM_INFERENCE_METHODS: $MIXGLM_INFERENCE_METHODS"

mkdir -p logs
mkdir -p "$MIXGLM_OUTPUT_ROOT/checkpoints"

# Automatically loop the script using SLURM if it hits a wall clock limit (or just run once)
python experiments/simulations/master_run.py
