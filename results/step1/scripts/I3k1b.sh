export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_ENABLE_X64=1 OUT=../data/s1 KAPPA=1
cd results/step1/scripts
SIGS="1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.6 3.0 4.0"
python3 st_batch.py eqN5 1000 0.1 128 34 $SIGS > ../data/I3k1_eqN5_p1e3.log 2>&1
for p in 16 64 256; do python3 st_batch.py gauss45 $p 0.1 64 35 $SIGS; done > ../data/I3k1_gauss45_short.log 2>&1
python3 st_batch.py gauss45 1000 0.1 64 36 $SIGS > ../data/I3k1_gauss45_p1e3.log 2>&1
python3 st_batch.py eqN5 10000 0.1 32 37 1.0 1.4 2.2 > ../data/I3k1_eqN5_p1e4.log 2>&1
python3 st_batch.py gauss45 10000 0.1 32 38 1.0 1.4 2.2 > ../data/I3k1_gauss45_p1e4.log 2>&1
