#!/bin/sh
cd "$(dirname "$0")"
while [ ! -f ../data/queue.done ]; do sleep 30; done
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
NPART=256 python3 expB_sweep.py 100 lam100 0.3 1 3 10 30 100 300 > ../data/expB_lam100.log 2>&1
NPART=192 python3 expB_sweep.py 10 lam10 0.3 1 3 10 30 100 > ../data/expB_lam10.log 2>&1
NPART=192 python3 expB_sweep.py 1 lam1 0.3 1 3 10 30 > ../data/expB_lam1.log 2>&1
echo done > ../data/queue2.done
