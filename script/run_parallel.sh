#!/usr/bin/env bash

# Usage:
#   sbatch run_parallel.sh <settings.json>
#
# The script reads the "sweep" block of the settings JSON and creates one
# SLURM array task per combination of (n_qubits × depth × engine × max_bond_dimension
# × max_terms), or per qubit-specific magnetization observable when
# model.observable == "magnetization".

#SBATCH --job-name=mssim
#SBATCH --output=logs/mssim_%A_%a.out     # %A = job id, %a = array task id
#SBATCH --error=logs/mssim_%A_%a.err
#SBATCH --time=00:30:00                   # wall-clock limit per task
#SBATCH --mem=200M                        # memory per task
#SBATCH --cpus-per-task=1                 # CPUs per task

set -euo pipefail

# 1. Load jq early to parse JSON parameters
module load jq/1.6-GCCcore-12.2.0

# Parse arguments
SETTINGS="${1:?Usage: sbatch run_parallel.sh <settings.json>}"
EXTRA_ARGS=("${@:2}")

if [[ ! -f "$SETTINGS" ]]; then
    echo "ERROR: settings file not found: $SETTINGS" >&2
    exit 1
fi

# 2. Read sweep parameters using the new array syntax
mapfile -t N_QUBITS_LIST < <(jq -r '.sweep.n_qubits[]' "$SETTINGS")
mapfile -t DEPTH_LIST < <(jq -r '.sweep.depth[]' "$SETTINGS")
mapfile -t ENGINE_LIST < <(jq -r '.execution.engines[]' "$SETTINGS")
mapfile -t MAX_BOND_LIST < <(jq -r '(.sweep.max_bond_dimension // [.execution.max_bond_dimension] // [null])[]' "$SETTINGS")
mapfile -t MAX_TERMS_LIST < <(jq -r '(.sweep.max_terms // [.execution.max_terms] // [null])[]' "$SETTINGS")
OBSERVABLE_MODE=$(jq -r '.model.observable // ""' "$SETTINGS")
VERBOSE_OUTPUT=$(jq -r '.output.verbose // false' "$SETTINGS")

if [[ "$VERBOSE_OUTPUT" == "false" ]]; then
    export QIBO_LOG_LEVEL=3
fi

N_Q=${#N_QUBITS_LIST[@]}
N_D=${#DEPTH_LIST[@]}
N_E=${#ENGINE_LIST[@]}
N_B=${#MAX_BOND_LIST[@]}
N_T=${#MAX_TERMS_LIST[@]}

TASKS=()
for N_QUBITS in "${N_QUBITS_LIST[@]}"; do
    for DEPTH in "${DEPTH_LIST[@]}"; do
        for ENGINE in "${ENGINE_LIST[@]}"; do
            for MAX_BOND in "${MAX_BOND_LIST[@]}"; do
                for MAX_TERMS in "${MAX_TERMS_LIST[@]}"; do
                    if [[ "$OBSERVABLE_MODE" == "magnetization" ]]; then
                        # Build one task per qubit: observable_i = I...IZI...I
                        for (( O_IDX=0; O_IDX < N_QUBITS; O_IDX++ )); do
                            OBSERVABLE=$(printf '%*s' "$N_QUBITS" '' | tr ' ' 'I')
                            OBSERVABLE="${OBSERVABLE:0:O_IDX}Z${OBSERVABLE:O_IDX+1}"
                            TASKS+=("${N_QUBITS}|${DEPTH}|${ENGINE}|${MAX_BOND}|${MAX_TERMS}|${OBSERVABLE}")
                        done
                    else
                        TASKS+=("${N_QUBITS}|${DEPTH}|${ENGINE}|${MAX_BOND}|${MAX_TERMS}|${OBSERVABLE_MODE}")
                    fi
                done
            done
        done
    done
done

TOTAL=${#TASKS[@]}

if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: sweep produces zero tasks. Check sweep.n_qubits, sweep.depth, sweep.max_bond_dimension, sweep.max_terms, execution.engines, and model.observable in $SETTINGS." >&2
    exit 1
fi

# 3. If NOT inside a SLURM array job, submit the array
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p logs
    echo "Sweep dimensions: ${TOTAL} tasks"
    sbatch --array="0-$(( TOTAL - 1 ))" "$0" "$SETTINGS" "${EXTRA_ARGS[@]}"
    exit 0
fi

# 4. Inside SLURM array task: compute specific parameters
TASK_ID="${SLURM_ARRAY_TASK_ID}"

IFS='|' read -r N_QUBITS DEPTH ENGINE MAX_BOND MAX_TERMS OBSERVABLE <<< "${TASKS[$TASK_ID]}"

echo "Task ${TASK_ID}: n_qubits=${N_QUBITS}, depth=${DEPTH}, engine=${ENGINE}, max_bond=${MAX_BOND}, max_terms=${MAX_TERMS}, observable=${OBSERVABLE}"

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
MAIN_ARGS=(
    --settings "${SETTINGS}"
    --n_qubits "${N_QUBITS}"
    --depth "${DEPTH}"
    --engine "${ENGINE}"
    --run_id "${TASK_ID}"
    --output "${OUTPUT_FILE}"
)

if [[ -n "${MAX_BOND}" && "${MAX_BOND}" != "null" ]]; then
    MAIN_ARGS+=(--max_bond "${MAX_BOND}")
fi

if [[ -n "${MAX_TERMS}" && "${MAX_TERMS}" != "null" ]]; then
    MAIN_ARGS+=(--max_terms "${MAX_TERMS}")
fi

if [[ -n "${OBSERVABLE}" ]]; then
    MAIN_ARGS+=(--observable "${OBSERVABLE}")
fi

python main.py "${MAIN_ARGS[@]}"

echo "Task ${TASK_ID} finished successfully."