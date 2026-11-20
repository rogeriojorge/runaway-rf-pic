#!/bin/sh
cd "$(dirname "$0")"
while [ ! -f ../data/queue2.done ]; do sleep 30; done
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
THR=0.02,0.08 NPART=192 python3 expB_sweep.py 100 lowth100 0.3 1 3 10 30 100 300 > ../data/expB_lowth100.log 2>&1
echo done > ../data/queue3.done
