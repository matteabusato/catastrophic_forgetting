#!/bin/bash
#SBATCH --job-name=grokking
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

# Submit from the catastrophic_forgetting repository root.
# Any options following this script are passed straight to train.py.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Must submit with sbatch}"
VENV="${VENV:-$PWD/.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing $VENV; set VENV=/path/to/venv before sbatch" >&2
  exit 1
fi
source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u grokking/train.py --device cuda "$@"
