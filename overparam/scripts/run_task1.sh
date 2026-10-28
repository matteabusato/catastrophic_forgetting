#!/usr/bin/env bash
#SBATCH --job-name=overparam_task1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/overparam_task1_%A_%a.out
#SBATCH --error=logs/overparam_task1_%A_%a.err

# Submit from the repository root. All command-line arguments reach Python.
# Override resource directives with sbatch flags; set OVERPARAM_PYTHON if needed.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    OVERPARAM_ROOT="${OVERPARAM_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
else
    OVERPARAM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    OVERPARAM_ROOT="${OVERPARAM_REPO_ROOT:-$(cd -- "${OVERPARAM_SCRIPT_DIR}/../.." && pwd)}"
fi
cd "${OVERPARAM_ROOT}"
if [[ ! -f overparam/scripts/run_task1.py ]]; then
    echo "Submit from the repository root, or set OVERPARAM_REPO_ROOT." >&2
    exit 2
fi

if [[ -n "${OVERPARAM_PYTHON:-}" ]]; then
    OVERPARAM_INTERPRETER="${OVERPARAM_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    OVERPARAM_INTERPRETER="${VIRTUAL_ENV}/bin/python"
elif [[ -x .venv/bin/python ]]; then
    OVERPARAM_INTERPRETER="${OVERPARAM_ROOT}/.venv/bin/python"
elif [[ -x ../.venv/bin/python ]]; then
    OVERPARAM_INTERPRETER="${OVERPARAM_ROOT}/../.venv/bin/python"
else
    OVERPARAM_INTERPRETER="python3"
fi

read -r -a OVERPARAM_WIDTHS <<< "${OVERPARAM_KAPPAS:-1 1.2 1.5 2 3 4 5 10 20}"
OVERPARAM_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( OVERPARAM_INDEX < 0 || OVERPARAM_INDEX >= ${#OVERPARAM_WIDTHS[@]} )); then
    echo "Array index ${OVERPARAM_INDEX} has no kappa in OVERPARAM_KAPPAS." >&2
    exit 2
fi
echo "Task 1: array=${OVERPARAM_INDEX} kappa=${OVERPARAM_WIDTHS[$OVERPARAM_INDEX]} python=${OVERPARAM_INTERPRETER}"
exec "${OVERPARAM_INTERPRETER}" -u overparam/scripts/run_task1.py \
    --kappa "${OVERPARAM_WIDTHS[$OVERPARAM_INDEX]}" \
    --device "${OVERPARAM_DEVICE:-cuda}" --threads "${SLURM_CPUS_PER_TASK:-1}" "$@"
