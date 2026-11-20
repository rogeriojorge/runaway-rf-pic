export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_ENABLE_X64=1 OUT=../data/s1
cd results/step1/scripts
SIGS="1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.6 3.0 4.0"
for p in 16 64 256; do python3 st_batch.py eqN10 $p 0.1 128 3 $SIGS; done > ../data/I3short_eqN10.log 2>&1 &
for p in 16 64 256; do python3 st_batch.py gauss45 $p 0.1 64 3 $SIGS; done > ../data/I3short_gauss45.log 2>&1 &
