#!/usr/bin/env bash
#SBATCH --job-name=overparam_task2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/overparam_task2_%j.out
#SBATCH --error=logs/overparam_task2_%j.err

# Independent single-width job. Supply --task1-run-dir and selected --task1-step.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    OVERPARAM_ROOT="${OVERPARAM_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
else
    OVERPARAM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    OVERPARAM_ROOT="${OVERPARAM_REPO_ROOT:-$(cd -- "${OVERPARAM_SCRIPT_DIR}/../.." && pwd)}"
fi
cd "${OVERPARAM_ROOT}"
if [[ ! -f overparam/scripts/run_task2.py ]]; then
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
exec "${OVERPARAM_INTERPRETER}" -u overparam/scripts/run_task2.py \
    --device "${OVERPARAM_DEVICE:-cuda}" --threads "${SLURM_CPUS_PER_TASK:-1}" "$@"
