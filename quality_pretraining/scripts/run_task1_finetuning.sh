#!/bin/bash

#SBATCH --job-name=task1_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0
#SBATCH --output=logs/task1_ft_%j.out
#SBATCH --error=logs/task1_ft_%j.err

set -euo pipefail


# ============================================================
# Usage
# ============================================================
#
# Basic:
#
# sbatch scripts/run_task1_finetuning.sh \
#     PRETRAIN_RUN_DIR \
#     PRETRAIN_STEP
#
#
# Optional:
#
# sbatch scripts/run_task1_finetuning.sh \
#     PRETRAIN_RUN_DIR \
#     PRETRAIN_STEP \
#     TARGET_STEPS \
#     LR
#
#
# Example:
#
# sbatch scripts/run_task1_finetuning.sh \
#     results/scaling_d200_T5_k1_ks1_lr0p01_teacher200 \
#     40000
#
# By default this trains until 5d.
#
# ============================================================


PRETRAIN_RUN_DIR=${1:?Please provide the pretraining run directory}
PRETRAIN_STEP=${2:?Please provide the pretraining checkpoint step}

TARGET_STEPS=${3:-}
LR=${4:-0.01}


# ============================================================
# Project root
# ============================================================

cd /home/busato/catastrophic_forgetting/quality_pretraining

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

source ../.venv/bin/activate

mkdir -p logs


# ============================================================
# Print information
# ============================================================

echo
echo "======================================================"
echo "Task-1 fine-tuning"
echo "======================================================"
echo "Pretraining run:   ${PRETRAIN_RUN_DIR}"
echo "Pretraining step:  ${PRETRAIN_STEP}"
echo "Learning rate:     ${LR}"

if [ -n "${TARGET_STEPS}" ]; then
    echo "Target steps:      ${TARGET_STEPS}"
else
    echo "Target steps:      5d (default)"
fi

echo "======================================================"
echo

nvidia-smi || true


# ============================================================
# Build command
# ============================================================

CMD=(
    python -u scripts/run_task1_finetuning.py

    --pretrain-run-dir "${PRETRAIN_RUN_DIR}"

    --pretrain-step "${PRETRAIN_STEP}"

    --results-root "results/task1_finetuning"

    --lr "${LR}"

    --device cuda
)


# Only specify target if provided.
if [ -n "${TARGET_STEPS}" ]; then
    CMD+=(
        --target-steps "${TARGET_STEPS}"
    )
fi


# ============================================================
# Run
# ============================================================

"${CMD[@]}"


echo
echo "======================================================"
echo "Task-1 fine-tuning finished"
echo "======================================================"