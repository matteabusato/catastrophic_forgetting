#!/bin/bash

#SBATCH --job-name=task1_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/task1_ft_%j.out
#SBATCH --error=logs/task1_ft_%j.err

set -euo pipefail


# ============================================================
# Usage
# ============================================================
#
# Required:
#
#   sbatch scripts/run_task1_finetuning.sh \
#       PRETRAIN_RUN_DIR \
#       PRETRAIN_STEP \
#       TARGET_STEPS
#
# Optional:
#
#   sbatch scripts/run_task1_finetuning.sh \
#       PRETRAIN_RUN_DIR \
#       PRETRAIN_STEP \
#       TARGET_STEPS \
#       LR \
#       INIT_SCALE
#
#
# Example:
#
#   sbatch scripts/run_task1_finetuning.sh \
#       results/quality_pretraining/scaling_d200_T5 \
#       1600000 \
#       10000 \
#       0.01 \
#       1.0
#
# This means:
#
#   - load W at pretraining step 1,600,000
#   - fine-tune only w
#   - run Task-1 SGD for 10,000 updates
#   - lr = 0.01
#   - w_i(0) ~ N(0, 1)
#
# ============================================================


PRETRAIN_RUN_DIR=${1:?Please provide PRETRAIN_RUN_DIR}

PRETRAIN_STEP=${2:?Please provide PRETRAIN_STEP}

TARGET_STEPS=${3:?Please provide TARGET_STEPS}

LR=${4:-0.01}

INIT_SCALE=${5:-1.0}


# ============================================================
# Project setup
# ============================================================

cd /home/busato/catastrophic_forgetting/quality_pretraining

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

source ../.venv/bin/activate

mkdir -p logs


# ============================================================
# Print setup
# ============================================================

echo
echo "======================================================"
echo "Task-1 fine-tuning"
echo "======================================================"
echo "Pretraining run:   ${PRETRAIN_RUN_DIR}"
echo "Pretraining step:  ${PRETRAIN_STEP}"
echo "Target FT steps:   ${TARGET_STEPS}"
echo "Learning rate:     ${LR}"
echo "w init scale:      ${INIT_SCALE}"
echo "======================================================"
echo

nvidia-smi || true


# ============================================================
# Run
# ============================================================

python -u scripts/run_task1_finetuning.py \
    --pretrain-run-dir "${PRETRAIN_RUN_DIR}" \
    --pretrain-step "${PRETRAIN_STEP}" \
    --target-steps "${TARGET_STEPS}" \
    --lr "${LR}" \
    --low-rank-init-scale "${INIT_SCALE}" \
    --w-init-seed 21 \
    --sgd-seed 22 \
    --task1-test-seed 31 \
    --device cuda


echo
echo "======================================================"
echo "Task-1 fine-tuning finished"
echo "======================================================"