#!/usr/bin/env bash
# Preset: kappa=P/d=1, N=20*d^2, one sequence per SGD update.
# Trailing arguments override these defaults (e.g. --device cuda or --output PATH).
set -euo pipefail

WARM_START_PRESET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${WARM_START_PRESET_DIR}/run_warm_start.sh" \
    --seed 0 \
    --d 100 \
    --tokens 5 \
    --rank 100 \
    --teacher-rank 100 \
    --n-train 200000 \
    --initial-fraction 0.5 \
    --n-test 10000 \
    --init-scale 1.0 \
    --teacher-scale 1.0 \
    --optimizer sgd \
    --batch-size 1 \
    --pretrain-steps 100000 \
    --train-steps 200000 \
    --lr 0.01 \
    --pretrain-lr 0.01 \
    --log-every 50 \
    "$@"
