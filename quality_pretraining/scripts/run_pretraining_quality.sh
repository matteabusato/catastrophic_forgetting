#!/bin/bash

#SBATCH --job-name=resume_pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0
#SBATCH --output=logs/resume_pretrain_%j.out
#SBATCH --error=logs/resume_pretrain_%j.err

set -euo pipefail


# ============================================================
# Usage
# ============================================================
#
# Default:
#   resume for another 1 * d^2 steps
#
#   sbatch scripts/run_pretraining_quality.sh \
#       results/scaling_d100_T5_...
#
#
# Optionally specify how many additional d^2 units:
#
#   sbatch scripts/run_pretraining_quality.sh \
#       results/scaling_d100_T5_... 3
#
# means:
#
#   new_target = latest_step + 3 * d^2
#
# ============================================================


RUN_DIR=${1:?Please provide the run directory}

EXTRA_D2=${2:-1}


# ============================================================
# Project setup
# ============================================================

cd /home/busato/catastrophic_forgetting/quality_pretraining

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

source ../.venv/bin/activate

mkdir -p logs


# ============================================================
# Normalize the run path
# ============================================================

# Remove trailing slash if present
RUN_DIR="${RUN_DIR%/}"

# Example:
#
# RUN_DIR =
# results/quality_pretraining/scaling_d100_T5_...
#
# becomes
#
# RESULTS_ROOT =
# results/quality_pretraining
#
# RUN_NAME =
# scaling_d100_T5_...

RESULTS_ROOT=$(dirname "${RUN_DIR}")
RUN_NAME=$(basename "${RUN_DIR}")


CONFIG_PATH="${RUN_DIR}/config.json"


if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: config.json not found:"
    echo "  ${CONFIG_PATH}"
    exit 1
fi


# ============================================================
# Read d from config.json
# ============================================================

D=$(python - <<PY
import json

with open("${CONFIG_PATH}", "r") as f:
    config = json.load(f)

print(config["d"])
PY
)


D2=$((D * D))


# ============================================================
# Find latest checkpoint
# ============================================================

LATEST_STEP=$(python - <<PY
import json
from pathlib import Path

run_dir = Path("${RUN_DIR}")

latest_file = run_dir / "latest.json"

if latest_file.exists():

    with open(latest_file, "r") as f:
        latest = json.load(f)

    print(int(latest["step"]))

else:

    checkpoint_root = run_dir / "checkpoints"

    steps = []

    if checkpoint_root.exists():
        for p in checkpoint_root.iterdir():

            if not p.is_dir():
                continue

            if not p.name.startswith("step_"):
                continue

            try:
                steps.append(
                    int(p.name.replace("step_", ""))
                )
            except ValueError:
                pass

    if not steps:
        raise RuntimeError(
            f"No checkpoints found in {checkpoint_root}"
        )

    print(max(steps))
PY
)


# ============================================================
# Decide new target
# ============================================================

ADDITIONAL_STEPS=$((EXTRA_D2 * D2))

TARGET_STEPS=$((LATEST_STEP + ADDITIONAL_STEPS))


# ============================================================
# Print summary
# ============================================================

echo
echo "======================================================"
echo "Resume pre-training"
echo "======================================================"
echo "Run directory:       ${RUN_DIR}"
echo "Results root:        ${RESULTS_ROOT}"
echo "Run name:            ${RUN_NAME}"
echo
echo "d:                   ${D}"
echo "d^2:                 ${D2}"
echo
echo "Latest checkpoint:   ${LATEST_STEP}"
echo "Additional d^2:      ${EXTRA_D2}"
echo "Additional steps:    ${ADDITIONAL_STEPS}"
echo "New target:          ${TARGET_STEPS}"
echo
echo "Start t/d^2:         $(python - <<PY
print(${LATEST_STEP} / (${D} ** 2))
PY
)"
echo "Target t/d^2:        $(python - <<PY
print(${TARGET_STEPS} / (${D} ** 2))
PY
)"
echo "======================================================"
echo


nvidia-smi || true


# ============================================================
# Resume
# ============================================================

python -u -m scripts.run_pretraining_quality \
    --results-root "${RESULTS_ROOT}" \
    --run-name "${RUN_NAME}" \
    --target-steps "${TARGET_STEPS}" \
    --resume \
    --device cuda


echo
echo "======================================================"
echo "Finished"
echo "Run: ${RUN_NAME}"
echo "Reached step: ${TARGET_STEPS}"
echo "======================================================"