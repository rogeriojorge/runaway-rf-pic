#!/bin/sh
# sequential queue (<= 4 threads): waits for the lambda=30 job, then Exp A, D, low-theta B, collisions
cd "$(dirname "$0")"
while pgrep -f "expB_sweep.py 30" > /dev/null; do sleep 20; done
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
python3 expA_collisionless.py > ../data/expA.log 2>&1
python3 expD_twoline.py > ../data/expD.log 2>&1
THR=0.02,0.08 NPART=256 python3 expB_sweep.py 1000 lowth 0.03 0.1 0.3 1 3 10 30 100 300 1000 3000 > ../data/expB_lowth.log 2>&1
NUF=1 NPART=192 python3 expB_sweep.py 1000 coll1 0.3 1 3 10 30 100 300 > ../data/expB_coll1.log 2>&1
for f in 0.01 0.1 10 100 1000; do
  NUF=$f NPART=192 python3 expB_sweep.py 1000 collR64_f$f 64 > ../data/expB_collR64_f$f.log 2>&1
done
echo done > ../data/queue.done
