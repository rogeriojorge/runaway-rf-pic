export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_ENABLE_X64=1 OUT=../data/s1
cd results/step1/scripts
python3 st_batch.py eqN10 10000 0.1 32 5 1.4 1.6 2.0 2.2 2.6 3.0 > ../data/I3long_eqN10.log 2>&1 &
python3 st_batch.py gauss45 10000 0.1 32 5 1.4 2.2 > ../data/I3long_gauss45.log 2>&1 &
