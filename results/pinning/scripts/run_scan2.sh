#!/bin/bash
export OMP_NUM_THREADS=1 JAX_ENABLE_X64=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd results/pinning/scripts
python3 pinrun.py 1000 256 2.0 0.05 s2 1e-9
for b in "$@"; do python3 pinrun.py 1000 256 2.0 0.05 s2 $b; done
