#!/bin/bash -l

#SBATCH --job-name=lora-matched
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00

#SBATCH --output=job_output/lora_matched_%j.out
#SBATCH --error=job_output/lora_matched_%j.err

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

python src/lora_dynamics/train.py \
    --experiment matched_two_stage \
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