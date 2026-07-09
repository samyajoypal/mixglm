#!/bin/bash
#SBATCH --job-name=real_data_K23
#SBATCH --partition=cm4_inter
#SBATCH --output=logs/real_K23_%j.out
#SBATCH --error=logs/real_K23_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G

# LRZ specific setup
module load python/3.10.12-extended 2>/dev/null || true

cd ~/mixglm || exit 1
source .venv/bin/activate
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "SLURM CPUs: $SLURM_CPUS_PER_TASK"

mkdir -p logs
mkdir -p experiments/real_data/real_outputs/checkpoints

# Run the checkpointed script!
python -u experiments/real_data/run_real_data_checkpointed.py
