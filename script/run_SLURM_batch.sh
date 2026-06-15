#!/usr/bin/env bash
#SBATCH --job-name=mssim
#SBATCH --output=logs/mssim_%A_%a.out     # %A = job id, %a = array task id
#SBATCH --error=logs/mssim_%A_%a.err
#SBATCH --time=00:30:00                   # wall-clock limit per task
#SBATCH --mem=200M                        # memory per task
#SBATCH --cpus-per-task=1                 # CPUs per task

set -euo pipefail

# Parse arguments
SETTINGS="${1:?Usage: run_batch.sh <settings.json>}"

if [[ ! -f "$SETTINGS" ]]; then
    echo "ERROR: settings file not found: $SETTINGS" >&2
    exit 1
fi

# Environment Setup (Python & Environment)
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${VENV_PATH:-./.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
fi

python batch.py --config "${SETTINGS}" --parallel