export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_ENABLE_X64=1 OUT=../data/s1
cd results/step1/scripts
while pgrep -f "st_batch.py eqN10 10000" > /dev/null; do sleep 10; done
SIGS="1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.6 3.0 4.0"
python3 st_batch.py eqN10 1000 0.1 128 11 $SIGS > ../data/I3lap_eqN10_s11.log 2>&1 &
python3 st_batch.py eqN10 1000 0.1 128 12 $SIGS > ../data/I3lap_eqN10_s12.log 2>&1
wait
python3 st_batch.py gauss45 1000 0.1 64 13 $SIGS > ../data/I3lap_gauss45_s13.log 2>&1 &
python3 st_batch.py eqN5 10000 0.1 32 14 1.0 1.4 2.2 3.0 > ../data/I3lap_eqN5_p1e4.log 2>&1
wait
