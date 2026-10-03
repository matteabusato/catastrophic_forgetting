#!/bin/bash

#SBATCH --job-name=pretrain_scaling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/pretrain_scaling_%A_%a.out
#SBATCH --error=logs/pretrain_scaling_%A_%a.err


# Exactly three jobs:
#   array 0 -> d=100
#   array 1 -> d=200
#   array 2 -> d=500
#SBATCH --array=0-2


set -euo pipefail


# ============================================================
# Move to project root
# ============================================================

cd "${SLURM_SUBMIT_DIR}"


# ============================================================
# Activate environment
# ============================================================

# Change this if needed.
# You said the venv is outside the project directory.
source ../.venv/bin/activate


# ============================================================
# Shared experiment settings
# ============================================================

T=5

KAPPA=1.0
KAPPA_STAR=1.0

LR=0.01

N_TEST=1000

DTYPE="float64"

# Keep these fixed across dimensions.
STUDENT_SEED=1
SGD_SEED=2
TEST_SEED=11


# ============================================================
# Dimension-specific settings
# ============================================================

D_VALUES=(100 200 500)

# Explicitly use different teacher realizations.
TEACHER_SEEDS=(100 200 500)

D=${D_VALUES[$SLURM_ARRAY_TASK_ID]}
TEACHER_SEED=${TEACHER_SEEDS[$SLURM_ARRAY_TASK_ID]}


# ============================================================
# Derived quantities
# ============================================================

D2=$((D * D))

# Train each model until 2 d^2 samples / SGD updates.
TARGET_STEPS=$((40 * D2))

# Metrics every d/2.
EVAL_EVERY=$((D / 2))


# Make run names explicit and unique.
RUN_NAME="scaling_d${D}_T${T}"


# ============================================================
# Print job information
# ============================================================

echo "======================================================"
echo "Pre-training scaling experiment"
echo "======================================================"
echo "SLURM job ID:       ${SLURM_JOB_ID}"
echo "SLURM array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Host:               $(hostname)"
echo
echo "Run name:           ${RUN_NAME}"
echo "d:                  ${D}"
echo "d^2:                ${D2}"
echo "target steps:       ${TARGET_STEPS}"
echo "target / d^2:       40"
echo "eval every:         ${EVAL_EVERY}"
echo "learning rate:      ${LR}"
echo "teacher seed:       ${TEACHER_SEED}"
echo "student seed:       ${STUDENT_SEED}"
echo "SGD seed:           ${SGD_SEED}"
echo "test seed:          ${TEST_SEED}"
echo "n_test:             ${N_TEST}"
echo "======================================================"
echo

nvidia-smi || true


# ============================================================
# Run
# ============================================================

python -u scripts/run_pretraining_quality.py \
    --run-name "${RUN_NAME}" \
    --d "${D}" \
    --T "${T}" \
    --kappa "${KAPPA}" \
    --kappa-star "${KAPPA_STAR}" \
    --lr "${LR}" \
    --dtype "${DTYPE}" \
    --teacher-seed "${TEACHER_SEED}" \
    --student-seed "${STUDENT_SEED}" \
    --sgd-seed "${SGD_SEED}" \
    --test-seed "${TEST_SEED}" \
    --n-test "${N_TEST}" \
    --eval-every "${EVAL_EVERY}" \
    --target-steps "${TARGET_STEPS}" \
    --device cuda


echo
echo "======================================================"
echo "Finished ${RUN_NAME}"
echo "End time: $(date)"
echo "======================================================"