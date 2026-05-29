#!/usr/bin/env bash

# Usage:
#   bash plots/run_plotting.sh [plot_settings.json]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SETTINGS="${1:-${SCRIPT_DIR}/plot_settings.json}"

if [[ ! -f "${SETTINGS}" ]]; then
    echo "ERROR: plot settings file not found: ${SETTINGS}" >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
fi

python "${SCRIPT_DIR}/main_plotting.py" "${SETTINGS}"