export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_ENABLE_X64=1
cd results/step1/scripts
out=../data/item4_plateau.txt; : > $out
for cfg in "100 10 0.19 0.21" "100 10 0.1 0.3" "100 20 0.1 0.3" "100 40 0.1 0.3" "300 10 0.1 0.3" "1000 10 0.1 0.3"; do set -- $cfg; python3 plateau2.py $1 $2 $3 $4 64 >> $out 2>&1; done
for v in "0 0" "1 0" "0 1"; do set -- $v; echo "drive terms: fric=$1 rr=$2 (E always on), lam=100 Xs=40" >> $out; FRIC=$1 RR=$2 python3 plateau2.py 100 40 0.1 0.3 64 >> $out 2>&1; done
