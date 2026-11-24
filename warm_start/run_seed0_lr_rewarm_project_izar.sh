#!/usr/bin/env bash
#SBATCH --job-name=ws0-elr-proj
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=warm-start-elr-proj-%j.out
#SBATCH --error=warm-start-elr-proj-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
VENV="${VENV:-$PWD/.venv}"
source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u warm_start/continue_seed0_rewarming.py \
    --variant lr_rewarm_project \
    --checkpoint warm_start/outputs/seed_00/warm_after_half.pt \
    --data-root "${CIFAR10_ROOT:-$PWD/warm_start/data}" \
    --output-dir warm_start/outputs/seed_00/lr_rewarm_project
