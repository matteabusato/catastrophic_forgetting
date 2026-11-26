#!/usr/bin/env bash
#SBATCH --job-name=ws-tanh-mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=warm-start-tanh-mlp-%j.out
#SBATCH --error=warm-start-tanh-mlp-%j.err

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
SEED="${SEED:-0}"
SEED_DIR="$(printf 'seed_%02d' "$SEED")"
python -u warm_start/run_tanh_mlp_experiment.py \
    --seed "$SEED" \
    --data-root "${CIFAR10_ROOT:-$PWD/warm_start/data}" \
    --output-dir "warm_start/outputs/architecture_study/three_layer_tanh_mlp/$SEED_DIR"
