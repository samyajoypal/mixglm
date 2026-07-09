#!/bin/bash
#SBATCH --job-name=mixglm_real_screen
#SBATCH --partition=downtime-24c
#SBATCH --output=logs/real_screen_%j.out
#SBATCH --error=logs/real_screen_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G

module load python/3.10.12-extended 2>/dev/null || true

cd ~/mixglm || exit 1
source .venv/bin/activate
export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MIXGLM_REAL_OUTPUT_ROOT=${MIXGLM_REAL_OUTPUT_ROOT:-experiments/real_data/targeted_outputs/v2_less_sparse_k123}
export MIXGLM_REAL_DATASETS=${MIXGLM_REAL_DATASETS:-rand,blog,super_raw,parkinsons_log,crime_beta}
export MIXGLM_REAL_INITS=${MIXGLM_REAL_INITS:-kmeans_glm,quantile_glm}
export MIXGLM_REAL_LAMBDAS=${MIXGLM_REAL_LAMBDAS:-0,0.1,0.25,0.5,1,2,5,10,20}
export MIXGLM_REAL_K_MIN=${MIXGLM_REAL_K_MIN:-1}
export MIXGLM_REAL_K_MAX=${MIXGLM_REAL_K_MAX:-3}
export MIXGLM_REAL_N_TRAIN=${MIXGLM_REAL_N_TRAIN:-2000}
export MIXGLM_REAL_N_TEST=${MIXGLM_REAL_N_TEST:-1000}
export MIXGLM_REAL_P_SCREEN=${MIXGLM_REAL_P_SCREEN:-40}
export MIXGLM_REAL_MAX_ITER=${MIXGLM_REAL_MAX_ITER:-160}
export MIXGLM_REAL_TOL=${MIXGLM_REAL_TOL:-1e-3}
export MIXGLM_REAL_N_STARTS=${MIXGLM_REAL_N_STARTS:-2}
export MIXGLM_REAL_ACTIVE_THRESHOLD=${MIXGLM_REAL_ACTIVE_THRESHOLD:-1e-5}
export MIXGLM_REAL_SEED=${MIXGLM_REAL_SEED:-20260624}

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "SLURM CPUs: $SLURM_CPUS_PER_TASK"
echo "MIXGLM_REAL_OUTPUT_ROOT: $MIXGLM_REAL_OUTPUT_ROOT"
echo "MIXGLM_REAL_DATASETS: $MIXGLM_REAL_DATASETS"
echo "MIXGLM_REAL_INITS: $MIXGLM_REAL_INITS"
echo "MIXGLM_REAL_LAMBDAS: $MIXGLM_REAL_LAMBDAS"
echo "MIXGLM_REAL_K_MIN: $MIXGLM_REAL_K_MIN"
echo "MIXGLM_REAL_K_MAX: $MIXGLM_REAL_K_MAX"
echo "MIXGLM_REAL_N_TRAIN: $MIXGLM_REAL_N_TRAIN"
echo "MIXGLM_REAL_N_TEST: $MIXGLM_REAL_N_TEST"
echo "MIXGLM_REAL_P_SCREEN: $MIXGLM_REAL_P_SCREEN"
echo "MIXGLM_REAL_MAX_ITER: $MIXGLM_REAL_MAX_ITER"
echo "MIXGLM_REAL_TOL: $MIXGLM_REAL_TOL"
echo "MIXGLM_REAL_N_STARTS: $MIXGLM_REAL_N_STARTS"
echo "MIXGLM_REAL_ACTIVE_THRESHOLD: $MIXGLM_REAL_ACTIVE_THRESHOLD"

mkdir -p logs
mkdir -p "$MIXGLM_REAL_OUTPUT_ROOT/checkpoints"

python -u experiments/real_data/run_targeted_hpc_screen.py
