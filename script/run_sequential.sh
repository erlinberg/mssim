#!/usr/bin/env bash

# Usage:
#   ./run_sequential.sh <settings.json>

set -euo pipefail

read_json_list() {
    local expr="$1"
    jq -r "(${expr}) | if type == \"array\" then .[] else . end" "$SETTINGS"
}

# Parse arguments
SETTINGS="${1:?Usage: ./run_sequential.sh <settings.json>}"
EXTRA_ARGS=("${@:2}")

if [[ ! -f "$SETTINGS" ]]; then
    echo "ERROR: settings file not found: $SETTINGS" >&2
    exit 1
fi

# 1. Read run parameters. Each entry may come from a sweep array or a scalar
# top-level setting, which makes single-run launches work without a sweep block.
N_QUBITS_LIST=($(read_json_list '(.sweep.n_qubits // .model.n_qubits)'))
DEPTH_LIST=($(read_json_list '(.sweep.depth // .model.depth)'))
ENGINE_LIST=($(read_json_list '.execution.engines'))
MAX_BOND_LIST=($(read_json_list '(.sweep.max_bond_dimension // .execution.max_bond_dimension // null)'))
MAX_TERMS_LIST=($(read_json_list '(.sweep.max_terms // .execution.max_terms // null)'))
OBSERVABLE_MODE=$(jq -r '.model.observable // ""' "$SETTINGS")
VERBOSE_OUTPUT=$(jq -r '.output.verbose // false' "$SETTINGS")

if [[ "$VERBOSE_OUTPUT" == "false" ]]; then
    export QIBO_LOG_LEVEL=3
fi

# Inizializza array vuoto (sintassi universale)
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
                    TASKS+=("${N_QUBITS}|${DEPTH}|${ENGINE}|${MAX_BOND}|${MAX_TERMS}|${OBSERVABLE_MODE}")
                done
            done
        done
    done
done

TOTAL=${#TASKS[@]}

if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: run produces zero tasks." >&2
    exit 1
fi

echo "Starting sequential run: ${TOTAL} task(s) total."
mkdir -p logs

# 2. Local Environment Setup
VENV_PATH="${VENV_PATH:-./.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
fi

# 3. Output Management
OUTPUT_DIR="$(jq -r '.output.filename | split("/")[:-1] | join("/")' "$SETTINGS")"
OUTPUT_FMT="$(jq -r '.output.format // "jsonl"' "$SETTINGS")"
OUTPUT_BASE="$(jq -r '.output.filename | split("/")[-1] | split(".")[0]' "$SETTINGS")"
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_BASE}.${OUTPUT_FMT}"
mkdir -p "${OUTPUT_DIR}"

# 4. Loop over all tasks sequentially
for (( TASK_ID=0; TASK_ID < TOTAL; TASK_ID++ )); do
    # Parsing compatibile Mac senza 'read -r' con molteplici variabili
    TASK="${TASKS[$TASK_ID]}"
    N_QUBITS=$(echo "$TASK" | cut -d'|' -f1)
    DEPTH=$(echo "$TASK" | cut -d'|' -f2)
    ENGINE=$(echo "$TASK" | cut -d'|' -f3)
    MAX_BOND=$(echo "$TASK" | cut -d'|' -f4)
    MAX_TERMS=$(echo "$TASK" | cut -d'|' -f5)
    OBSERVABLE=$(echo "$TASK" | cut -d'|' -f6)
    
    echo "[${TASK_ID}/${TOTAL}] n_qubits=${N_QUBITS}, depth=${DEPTH}, engine=${ENGINE}, max_bond=${MAX_BOND}, max_terms=${MAX_TERMS}, observable=${OBSERVABLE}"
    
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

    python main.py "${MAIN_ARGS[@]}" # > "logs/mssim_${TASK_ID}.out" 2> "logs/mssim_${TASK_ID}.err"
done

echo "All tasks finished successfully."