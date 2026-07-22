#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced so the variables remain in the current shell:" >&2
    echo "  source scripts/run_single_thread.sh" >&2
    exit 1
fi

# Prevent each DataLoader worker from creating its own BLAS/OpenMP thread pool.
# This avoids CPU oversubscription when num_workers is close to the vCPU count.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "CPU thread limits: OMP=1 MKL=1 OPENBLAS=1 NUMEXPR=1"
