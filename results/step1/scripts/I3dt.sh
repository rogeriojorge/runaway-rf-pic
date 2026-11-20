export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_ENABLE_X64=1 OUT=../data/s1
cd results/step1/scripts
for dtf in 0.2 0.05; do python3 st_batch.py eqN10 1000 $dtf 64 4 1.4 2.2; done > ../data/I3dt_eqN10.log 2>&1
