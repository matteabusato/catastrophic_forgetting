#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattea.busato@epfl.ch

set -euo pipefail

TARGET_STEPS=${1:?Please provide TARGET_STEPS}
MODE=${2:-fresh}

D=200
T=5

KAPPA=1.0
KAPPA_STAR=1.0

LR=0.001

N_TEST=1000

TEACHER_SEED=0
STUDENT_SEED=1
SGD_SEED=2
TEST_SEED=11

DTYPE="float64"

RUN_NAME="d200_T5_k1_ks1_seed0"

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p logs

module purge
module load gcc/11.3.0
module load python/3.10.4
module load cuda/11.8.0

source ../.venv/bin/activate


echo "============================================"
echo "Pre-training quality"
echo "============================================"

echo "Job ID:        ${SLURM_JOB_ID}"
echo "Host:          $(hostname)"
echo "Start time:    $(date)"

echo
echo "Run name:      ${RUN_NAME}"
echo "Mode:          ${MODE}"
echo "Target steps:  ${TARGET_STEPS}"

echo
echo "d:             ${D}"
echo "T:             ${T}"
echo "kappa:         ${KAPPA}"
echo "kappa_star:    ${KAPPA_STAR}"
echo "learning rate: ${LR}"
echo "n_test:        ${N_TEST}"

echo
nvidia-smi || true
echo


if [ "${MODE}" = "fresh" ]; then

    echo "Starting NEW pre-training trajectory."

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
        --target-steps "${TARGET_STEPS}" \
        --device cuda

elif [ "${MODE}" = "resume" ]; then

    echo "Resuming existing pre-training trajectory."

    python -u  scripts/run_pretraining_quality.py \
        --run-name "${RUN_NAME}" \
        --target-steps "${TARGET_STEPS}" \
        --resume \
        --device cuda

else

    echo "ERROR: MODE must be either 'fresh' or 'resume'."
    exit 1

fi

echo
echo "============================================"
echo "Finished"
echo "============================================"

echo "End time: $(date)"