#!/bin/bash
#SBATCH --job-name=mixglm_real
#SBATCH --partition=cm4_inter
#SBATCH --output=logs/real_%j.out
#SBATCH --error=logs/real_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G

module purge
module load python/3.10.12-extended

cd ~/mixglm || exit 1

source .venv/bin/activate
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Ensure underlying libraries don't multi-thread since joblib uses process parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "SLURM CPUs: $SLURM_CPUS_PER_TASK"

mkdir -p logs
mkdir -p data


# 1. Fetch datasets directly on the cluster
echo "Fetching real datasets..."
python experiments/real_data/fetch_datasets.py

# 2. Run the main analysis pipeline
DATASET=${1:-all}
echo "Running full beam search pipeline for dataset: $DATASET..."
python experiments/real_data/run_real_data.py $DATASET
