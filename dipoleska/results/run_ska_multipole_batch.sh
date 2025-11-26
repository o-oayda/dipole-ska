#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-python}
MPIEXEC_BIN=${MPIEXEC:-mpiexec}
MPI_NP=1
FIT_ARGS=""

usage() {
    cat <<'EOF'
Usage: run_ska_multipole_batch.sh [--mpi_ranks N] [--fit_args "ARG STRING"]

Options:
  --mpi_ranks N    Number of MPI ranks per ska_fit_multipoles run (default: 1)
  --fit_args STR   Additional arguments passed to ska_fit_multipoles.py
  --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mpi_ranks)
            MPI_NP="$2"
            shift 2
            ;;
        --fit_args)
            FIT_ARGS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIST_SCRIPT="$SCRIPT_DIR/list_map_collection_ids.py"
FIT_SCRIPT="$SCRIPT_DIR/ska_fit_multipoles.py"

if [[ ! -f "$LIST_SCRIPT" ]]; then
    echo "Cannot find $LIST_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "Cannot find $FIT_SCRIPT" >&2
    exit 1
fi

echo "Discovering SKA map collection IDs..."
mapfile -t COLLECTION_IDS < <("$PYTHON_BIN" "$LIST_SCRIPT")

if [[ ${#COLLECTION_IDS[@]} -eq 0 ]]; then
    echo "No map collections were found; aborting." >&2
    exit 1
fi

for collection_id in "${COLLECTION_IDS[@]}"; do
    if [[ -z "$collection_id" ]]; then
        continue
    fi
    echo "Running multipole fit for ${collection_id}..."
    $MPIEXEC_BIN -n "$MPI_NP" "$PYTHON_BIN" "$FIT_SCRIPT" \
        --collection-id "$collection_id" $FIT_ARGS
done
