#!/bin/sh
cd "$(dirname "$0")"
while [ ! -f ../data/queue3.done ]; do sleep 30; done
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
for f in 0 3 10 30 100; do NUF=$f NPART=160 python3 expB_sweep.py 1 collL1_f$f 10 > ../data/expB_collL1_f$f.log 2>&1; done
NUF=1 NPART=160 python3 expB_sweep.py 1 collL1_R 1 3 30 > ../data/expB_collL1_R.log 2>&1
echo done > ../data/queue4.done
