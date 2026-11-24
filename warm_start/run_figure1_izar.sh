#!/usr/bin/env bash
#SBATCH --job-name=warm-start-fig1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=warm-start-fig1-%j.out
#SBATCH --error=warm-start-fig1-%j.err

# Submit from the catastrophic_forgetting repository root:
#   sbatch warm_start/izar.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

VENV="${VENV:-$PWD/.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "Missing $VENV; set VENV=/path/to/venv before sbatch" >&2
    exit 1
fi
source "$VENV/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u warm_start/replicate_figure1.py \
    --data-root "${CIFAR10_ROOT:-$PWD/warm_start/data}" \
    --output-dir "${WARM_START_OUTPUT_DIR:-$PWD/warm_start/outputs}"
