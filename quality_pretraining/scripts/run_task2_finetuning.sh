#!/bin/bash

#SBATCH --job-name=task2_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/task2_ft_%j.out
#SBATCH --error=logs/task2_ft_%j.err

set -euo pipefail


# ============================================================
# Usage
# ============================================================
#
# Required:
#
#   sbatch scripts/run_task2_finetuning.sh \
#       PRETRAIN_RUN_DIR \
#       PRETRAIN_STEP \
#       TASK1_RUN_DIR \
#       TASK1_STEP \
#       TARGET_STEPS
#
#
# Optional:
#
#   sbatch scripts/run_task2_finetuning.sh \
#       PRETRAIN_RUN_DIR \
#       PRETRAIN_STEP \
#       TASK1_RUN_DIR \
#       TASK1_STEP \
#       TARGET_STEPS \
#       LR
#
#
# Example for d = 200:
#
#   alpha_pre = 40
#
#       pretraining step
#       = 40 d^2
#       = 40 * 200^2
#       = 1,600,000
#
#   Task-1 endpoint
#       = 500 d
#       = 100,000
#
#   Task-2 target
#       = 500 d
#       = 100,000
#
#
#   sbatch scripts/run_task2_finetuning.sh \
#       results/quality_pretraining/scaling_d200_T5 \
#       1600000 \
#       results/quality_pretraining/finetunetask1_d200_T5/prestep_1600000_lr0p01_init1_wseed21_sgdseed22 \
#       100000 \
#       100000 \
#       0.01
#
#
# This means:
#
#   - load W from pretraining step 1,600,000
#   - load w from Task-1 step 100,000
#   - DO NOT reinitialize w
#   - freeze W
#   - continue optimizing the same w on Task 2
#   - run Task-2 SGD for 100,000 updates
#   - Task-2 lr = 0.01
#
# ============================================================


PRETRAIN_RUN_DIR=${1:?Please provide PRETRAIN_RUN_DIR}

PRETRAIN_STEP=${2:?Please provide PRETRAIN_STEP}

TASK1_RUN_DIR=${3:?Please provide TASK1_RUN_DIR}

TASK1_STEP=${4:?Please provide TASK1_STEP}

TARGET_STEPS=${5:?Please provide TARGET_STEPS}

LR=${6:-0.01}


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
echo "Task-2 fine-tuning"
echo "======================================================"

echo "Pretraining run:       ${PRETRAIN_RUN_DIR}"
echo "Pretraining step:      ${PRETRAIN_STEP}"

echo "Task-1 run:            ${TASK1_RUN_DIR}"
echo "Task-1 checkpoint:     ${TASK1_STEP}"

echo "Target Task-2 steps:   ${TARGET_STEPS}"
echo "Task-2 learning rate:  ${LR}"

echo "======================================================"
echo

nvidia-smi || true


# ============================================================
# Run
# ============================================================

python -u scripts/run_task2_finetuning.py \
    --pretrain-run-dir "${PRETRAIN_RUN_DIR}" \
    --pretrain-step "${PRETRAIN_STEP}" \
    --task1-run-dir "${TASK1_RUN_DIR}" \
    --task1-step "${TASK1_STEP}" \
    --target-steps "${TARGET_STEPS}" \
    --lr "${LR}" \
    --sgd-seed 42 \
    --task2-test-seed 41 \
    --device cuda


echo
echo "======================================================"
echo "Task-2 fine-tuning finished"
echo "======================================================"