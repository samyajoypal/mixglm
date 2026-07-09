#!/bin/bash
#SBATCH --job-name=mixglm_sim
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --array=1-100
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00

# Usage: sbatch submit_slurm.sh <scenario> <example> <n_samples>
# Example: sbatch submit_slurm.sh A 1 1000

SCENARIO=$1
EXAMPLE=$2
N_SAMPLES=$3

mkdir -p logs
mkdir -p results

# Load environment if necessary (assume venv is two levels up or python is in PATH)
# source ../../.venv/bin/activate

python run_sim.py --scenario ${SCENARIO} --example ${EXAMPLE} --n_samples ${N_SAMPLES} --task_id ${SLURM_ARRAY_TASK_ID}
