#!/usr/bin/env bash

# Usage:
#   sbatch run_parallel.sh <settings.json>
#
# The script reads the "sweep" block of the settings JSON and creates one
# SLURM array task per combination of (n_qubits × depth × engine).

#SBATCH --job-name=mssim
#SBATCH --output=logs/mssim_%A_%a.out     # %A = job id, %a = array task id
#SBATCH --error=logs/mssim_%A_%a.err
#SBATCH --time=00:30:00                   # ⚠ wall-clock limit per task
#SBATCH --mem=8G                          # ⚠ memory per task
#SBATCH --cpus-per-task=4                 # ⚠ CPUs per task

set -euo pipefail

# 1. Load jq early to parse JSON parameters
module load jq/1.6-GCCcore-12.2.0

# Parse arguments
SETTINGS="${1:?Usage: sbatch run_parallel.sh <settings.json>}"

if [[ ! -f "$SETTINGS" ]]; then
    echo "ERROR: settings file not found: $SETTINGS" >&2
    exit 1
fi

# 2. Read sweep parameters using the new array syntax
N_QUBITS_LIST=($(jq -r '.sweep.n_qubits[]' "$SETTINGS"))
DEPTH_LIST=($(jq -r '.sweep.depth[]' "$SETTINGS"))
ENGINE_LIST=($(jq -r '.execution.engines[]' "$SETTINGS"))
VERBOSE_OUTPUT=$(jq -r '.output.verbose // false' "$SETTINGS")

if [[ "$VERBOSE_OUTPUT" == "false" ]]; then
    export QIBO_LOG_LEVEL=3
fi

N_Q=${#N_QUBITS_LIST[@]}
N_D=${#DEPTH_LIST[@]}
N_E=${#ENGINE_LIST[@]}
TOTAL=$(( N_Q * N_D * N_E ))

if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: sweep produces zero tasks. Check sweep.n_qubits, sweep.depth, and execution.engines in $SETTINGS." >&2
    exit 1
fi

# 3. If NOT inside a SLURM array job, submit the array
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p logs
    echo "Sweep dimensions: n_qubits=${N_Q} × depth=${N_D} × engines=${N_E} = ${TOTAL} tasks"
    sbatch --array="0-$(( TOTAL - 1 ))" "$0" "$SETTINGS" "${EXTRA_ARGS[@]}"
    exit 0
fi

# 4. Inside SLURM array task: compute specific parameters
TASK_ID="${SLURM_ARRAY_TASK_ID}"

# Index mapping matching the nested loop order of run_simple.sh
E_IDX=$(( TASK_ID / (N_Q * N_D) ))
REMAINDER=$(( TASK_ID % (N_Q * N_D) ))
D_IDX=$(( REMAINDER / N_Q ))
Q_IDX=$(( REMAINDER % N_Q ))

N_QUBITS="${N_QUBITS_LIST[$Q_IDX]}"
DEPTH="${DEPTH_LIST[$D_IDX]}"
ENGINE="${ENGINE_LIST[$E_IDX]}"

echo "Task ${TASK_ID}: n_qubits=${N_QUBITS}, depth=${DEPTH}, engine=${ENGINE}"

# 5. Environment Setup (Python & Environment)
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${VENV_PATH:-./.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
fi

# 6. Output Management
OUTPUT_DIR="$(jq -r '.output.filename | split("/")[:-1] | join("/")' "$SETTINGS")"
OUTPUT_FMT="$(jq -r '.output.format // "jsonl"' "$SETTINGS")"
OUTPUT_BASE="$(jq -r '.output.filename | split("/")[-1] | split(".")[0]' "$SETTINGS")"
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_BASE}.${OUTPUT_FMT}"
mkdir -p "${OUTPUT_DIR}"

# 7. Execution
python main.py \
    --settings  "${SETTINGS}"   \
    --n_qubits  "${N_QUBITS}"   \
    --depth     "${DEPTH}"      \
    --engine    "${ENGINE}"     \
    --run_id    "${TASK_ID}"    \
    --output    "${OUTPUT_FILE}" \

echo "Task ${TASK_ID} finished successfully."