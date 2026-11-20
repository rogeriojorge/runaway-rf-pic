#!/bin/bash
# lam scan at fixed b; 1 thread per job
export OMP_NUM_THREADS=1 JAX_ENABLE_X64=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd results/pinning/scripts
B=$1
for lam in 3000 1000 300; do python3 pinrun.py $lam 256 2.0 0.05 s1 $B; done
