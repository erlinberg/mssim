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

read_json_list() {
    local expr="$1"
    jq -r "(${expr}) | if type == \"array\" then .[] else . end" "$SETTINGS"
}

# 1. Load jq early to parse JSON parameters
module load jq/1.6-GCCcore-12.2.0

# Parse arguments
SETTINGS="${1:?Usage: sbatch run_parallel.sh <settings.json>}"
EXTRA_ARGS=("${@:2}")

if [[ ! -f "$SETTINGS" ]]; then
    echo "ERROR: settings file not found: $SETTINGS" >&2
    exit 1
fi

# 2. Read run parameters. Each entry may come from a sweep array or a scalar
# top-level setting, which makes single-run launches work without a sweep block.
mapfile -t N_QUBITS_LIST < <(read_json_list '(.sweep.n_qubits // .model.n_qubits)')
mapfile -t DEPTH_LIST < <(read_json_list '(.sweep.depth // .model.depth)')
mapfile -t ENGINE_LIST < <(read_json_list '.execution.engines')
mapfile -t MAX_BOND_LIST < <(read_json_list '(.sweep.max_bond_dimension // .execution.max_bond_dimension // null)')
mapfile -t MAX_TERMS_LIST < <(read_json_list '(.sweep.max_terms // .execution.max_terms // null)')
SWEEP_KWARG_COUNT=$(jq -r '.sweep.kwargs | if type == "array" then length else 0 end' "$SETTINGS")
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
            MAX_BOND_VALUES=(null)
            MAX_TERMS_VALUES=(null)

            case "$ENGINE" in
                *quimb*|*mpstab*)
                    MAX_BOND_VALUES=("${MAX_BOND_LIST[@]}")
                    ;;
                *qiskit_paulipropagation*)
                    MAX_TERMS_VALUES=("${MAX_TERMS_LIST[@]}")
                    ;;
                *)
                    ;;
            esac

            for MAX_BOND in "${MAX_BOND_VALUES[@]}"; do
                for MAX_TERMS in "${MAX_TERMS_VALUES[@]}"; do
                    MAX_KWARG_COUNT=1

                    if [[ $SWEEP_KWARG_COUNT != 0 ]]; then
                        MAX_KWARG_COUNT=$SWEEP_KWARG_COUNT
                    fi

                    for ((KWARG_ID=0;KWARG_ID<$MAX_KWARG_COUNT;KWARG_ID++)); do
                        TASKS+=("${N_QUBITS}|${DEPTH}|${ENGINE}|${MAX_BOND}|${MAX_TERMS}|${OBSERVABLE_MODE}|${KWARG_ID}")
                    done
                done
            done
        done
    done
done

TOTAL=${#TASKS[@]}

if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: run produces zero tasks. Check sweep.n_qubits, sweep.depth, sweep.max_bond_dimension, sweep.max_terms, sweep.kwargs, execution.engines, model.n_qubits, model.depth, and model.observable in $SETTINGS." >&2
    exit 1
fi

# 3. If NOT inside a SLURM array job, submit the array
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p logs
    if [[ "$TOTAL" -gt 1 ]]; then
        echo "Run dimensions: ${TOTAL} tasks"
        sbatch --array="0-$(( TOTAL - 1 ))" "$0" "$SETTINGS" "${EXTRA_ARGS[@]}"
        exit 0
    fi

    TASK_ID=0
else
    TASK_ID="${SLURM_ARRAY_TASK_ID}"
fi

# 4. Compute specific parameters for the selected task

IFS='|' read -r N_QUBITS DEPTH ENGINE MAX_BOND MAX_TERMS OBSERVABLE KWARG_ID <<< "${TASKS[$TASK_ID]}"

echo "Task ${TASK_ID}: n_qubits=${N_QUBITS}, depth=${DEPTH}, engine=${ENGINE}, max_bond=${MAX_BOND}, max_terms=${MAX_TERMS}, observable=${OBSERVABLE}, kwarg_id=${KWARG_ID}"

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

if [[ $SWEEP_KWARG_COUNT -gt 0 ]]; then
    MAIN_ARGS+=(--kwarg_id "${KWARG_ID}")
fi

python main.py "${MAIN_ARGS[@]}"

echo "Task ${TASK_ID} finished successfully."