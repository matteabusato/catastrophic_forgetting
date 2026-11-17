#!/bin/bash
#SBATCH --job-name=grok-rescale
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=grok-rescale-%A_%a.out

# Submit FROM the catastrophic_forgetting repository root:
#   sbatch --array=0-4%2 grokking/rescale_transfer.sh
# The five array tasks are a matched sweep from the ORIGINAL A step-1000:
#   0  c=1.0, no rewarm (baseline, reset AdamW)
#   1  c=0.5, no rewarm
#   2  c=2.0, no rewarm
#   3  c=1.0, rewarm to 0.003, return to 0.001 by B step 2000
#   4  c=0.5, rewarm to 0.003, return to 0.001 by B step 2000
# All use B=(a+b+1) mod 113, no A replay, and 20,000 B steps.
#
# To submit an individual custom run, WITHOUT --array:
#   sbatch grokking/rescale_transfer.sh --checkpoint ... --run-dir ... \
#      --mlp-scale 0.5 --steps 20000 --b-operation sum_plus_one
#   (add --rewarm to enable LR rewarming)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
VENV="${VENV:-$PWD/.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Python virtualenv not found: $VENV" >&2
  exit 1
fi
source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  case "${SLURM_ARRAY_TASK_ID}" in
    0) name="control";       scale="1.0"; rewarm=() ;;
    1) name="scale_0p5";     scale="0.5"; rewarm=() ;;
    2) name="scale_2";       scale="2.0"; rewarm=() ;;
    3) name="rewarm";        scale="1.0"; rewarm=(--rewarm) ;;
    4) name="scale_0p5_rewarm"; scale="0.5"; rewarm=(--rewarm) ;;
    *) echo "Unknown array task ID $SLURM_ARRAY_TASK_ID (expected 0..4)" >&2; exit 2 ;;
  esac
  CHECKPOINT="grokking/runs/a_nanda3_seed0/checkpoints/step_001000.pt"
  [[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint $CHECKPOINT" >&2; exit 3; }
  exec python -u grokking/rescale_transfer.py \
    --checkpoint "$CHECKPOINT" \
    --run-dir "grokking/runs/rescale_shift1_from_1000/$name" \
    --b-operation sum_plus_one \
    --mlp-scale "$scale" "${rewarm[@]}" \
    --peak-lr 0.003 --warmup-steps 200 --rewarm-end 2000 \
    --steps 20000 --eval-every 100 --checkpoint-every 1000 \
    --device cuda \
    --wandb-project grokking-continual \
    --wandb-entity busatomattea-epfl \
    --wandb-name "rescale_shift1_1000_${name}"
else
  # Flexible invocation (e.g., manual resume or different checkpoint).
  exec python -u grokking/rescale_transfer.py --device cuda "$@"
fi
