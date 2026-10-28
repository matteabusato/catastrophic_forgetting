#!/usr/bin/env bash
# Local launcher. Pass Python-runner options unchanged; defaults use one seed (0).
# For a cluster, activate the desired environment and submit this launcher with
# site-specific sbatch resource flags, adding --device cuda for a GPU.
set -euo pipefail

WARM_START_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WARM_START_REPO_ROOT="$(cd -- "${WARM_START_SCRIPT_DIR}/../.." && pwd)"

# Respect an explicit interpreter, then an active environment, then repo .venv.
if [[ -n "${WARM_START_PYTHON:-}" ]]; then
    WARM_START_INTERPRETER="${WARM_START_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    WARM_START_INTERPRETER="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${WARM_START_REPO_ROOT}/.venv/bin/python" ]]; then
    WARM_START_INTERPRETER="${WARM_START_REPO_ROOT}/.venv/bin/python"
else
    WARM_START_INTERPRETER="python3"
fi

exec "${WARM_START_INTERPRETER}" -u "${WARM_START_SCRIPT_DIR}/run_warm_start.py" "$@"
