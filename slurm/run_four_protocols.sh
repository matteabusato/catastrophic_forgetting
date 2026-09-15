#!/bin/bash -l

#SBATCH --job-name=lora-protocols
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-3

#SBATCH --output=job_output/lora_%A_%a.out
#SBATCH --error=job_output/lora_%A_%a.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@epfl.ch

set -euo pipefail

PROJECT_ROOT="$HOME/catastrophic_forgetting"
cd "$PROJECT_ROOT"

mkdir -p job_output
mkdir -p results/training_dynamics

module purge
module load gcc
module load python

source "$PROJECT_ROOT/.venv/bin/activate"

EXPERIMENTS=(
    "joint"
    "W_then_w"
    "w_then_W"
    "matched_two_stage"
)

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "Experiment: ${EXPERIMENT}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"

python src/lora_dynamics/train.py \
    --experiment "${EXPERIMENT}" \
    --d 200 \
    --T 5 \
    --kappa-star 1.0 \
    --kappa 1.0 \
    --alpha 2.0 \
    --alpha-prime 2.0 \
    --lr-W 1e-3 \
    --lr-w 1e-3 \
    --log-every 500 \
    --seed 0 \
    --device cuda \
    --output-root results/training_dynamics