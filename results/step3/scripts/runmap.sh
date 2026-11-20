#!/bin/bash
# map: 6 sigma x 7 Gamma, NP=64, seeds 1..3, T=1000 beat periods, dtfac 0.1; 4 parallel workers
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1
export NOSTOP=1
for seed in 1 2 3; do for s in 0.5 1.0 1.5 2.0 3.0 4.0; do echo "$seed $s"; done; done | xargs -P4 -L1 bash -c 'python3 batch.py map 1000 0.1 64 $0 $1 0.0 0.01 0.03 0.1 0.3 1.0 3.0 >> ../data/map.log 2>&1'
