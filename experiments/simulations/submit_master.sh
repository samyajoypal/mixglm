#!/bin/bash
#SBATCH --job-name=mixglm_master
#SBATCH --partition=epyc-256
#SBATCH --output=logs/master_%j.out
#SBATCH --error=logs/master_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=16
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
export MIXGLM_OUTPUT_ROOT=${MIXGLM_OUTPUT_ROOT:-paper_outputs/v5_statscomp_corrected_20260730}
export MIXGLM_N_REPS_A=${MIXGLM_N_REPS_A:-200}
export MIXGLM_N_REPS_B=${MIXGLM_N_REPS_B:-200}
export MIXGLM_N_REPS_C=${MIXGLM_N_REPS_C:-500}
export MIXGLM_SAMPLE_SIZES=${MIXGLM_SAMPLE_SIZES:-500,1000,1500,2500}
export MIXGLM_LAMBDA_GRID=${MIXGLM_LAMBDA_GRID:-0.25,0.5,1,2,5,10,20,50}
export MIXGLM_INFERENCE_METHODS=${MIXGLM_INFERENCE_METHODS:-louis}

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "SLURM CPUs: $SLURM_CPUS_PER_TASK"
echo "MIXGLM_OUTPUT_ROOT: $MIXGLM_OUTPUT_ROOT"
echo "MIXGLM_N_REPS_A: $MIXGLM_N_REPS_A"
echo "MIXGLM_N_REPS_B: $MIXGLM_N_REPS_B"
echo "MIXGLM_N_REPS_C: $MIXGLM_N_REPS_C"
echo "MIXGLM_SAMPLE_SIZES: $MIXGLM_SAMPLE_SIZES"
echo "MIXGLM_LAMBDA_GRID: $MIXGLM_LAMBDA_GRID"
echo "MIXGLM_INFERENCE_METHODS: $MIXGLM_INFERENCE_METHODS"

mkdir -p logs
mkdir -p "$MIXGLM_OUTPUT_ROOT/checkpoints"

# Automatically loop the script using SLURM if it hits a wall clock limit (or just run once)
python experiments/simulations/master_run.py
