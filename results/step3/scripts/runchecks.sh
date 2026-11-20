#!/bin/bash
# convergence checks: (a) window length T=3000 (sigma 1.5, 1.0; Gamma 0, 0.01, 0.03, 0.1), (b) dt: dtfac 0.05 at T=300 vs the production 0.1 at T=300, NP=64, seed 1
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 NOSTOP=1
{
 for s in 1.5 1.0; do echo "long $s 3000 0.1 0.0 0.01 0.03 0.1"; done
 for s in 1.5 3.0; do echo "dt05 $s 300 0.05 0.0 0.03 0.3"; echo "dt10 $s 300 0.1 0.0 0.03 0.3"; done
} | xargs -P4 -L1 bash -c 'tag=$0; s=$1; T=$2; dtf=$3; shift 3; python3 batch.py $tag $T $dtf 64 1 $s "$@" >> ../data/checks.log 2>&1'
