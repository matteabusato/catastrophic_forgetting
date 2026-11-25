#!/usr/bin/env bash
#SBATCH --job-name=ws-long-run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=warm-start-long-%j.out
#SBATCH --error=warm-start-long-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"

VENV="${VENV:-$PWD/.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "Missing $VENV; set VENV=/path/to/venv before sbatch" >&2
    exit 1
fi
source "$VENV/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

SEEDS="${SEEDS:-0 1 2 3 4}"
TARGET_FULL_DATA_EPOCHS="${TARGET_FULL_DATA_EPOCHS:-3500}"
read -r -a SEED_LIST <<< "$SEEDS"

python -u warm_start/continue_long_run.py \
    --data-root "${CIFAR10_ROOT:-$PWD/warm_start/data}" \
    --seeds "${SEED_LIST[@]}" \
    --target-full-data-epochs "$TARGET_FULL_DATA_EPOCHS"
