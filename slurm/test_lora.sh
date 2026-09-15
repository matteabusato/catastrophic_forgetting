#!/bin/bash -l

#SBATCH --job-name=lora-test
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

#SBATCH --output=job_output/lora_test_%j.out
#SBATCH --error=job_output/lora_test_%j.err

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

echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "Python:       $(which python)"
echo "=========================================="

nvidia-smi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python src/lora_dynamics/train.py \
    --experiment matched_two_stage \
    --d 200 \
    --T 5 \
    --kappa-star 1.0 \
    --kappa 1.0 \
    --alpha 0.05 \
    --alpha-prime 0.5 \
    --lr-W 1e-3 \
    --lr-w 1e-3 \
    --log-every 100 \
    --test-size 200 \
    --gradient-probe-size 1 \
    --seed 0 \
    --device cuda \
    --output-root results/training_dynamics