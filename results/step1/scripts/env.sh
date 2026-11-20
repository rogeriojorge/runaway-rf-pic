source ~/jic_stack_gate/dv/bin/activate
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1
cd ~/jic_step1/S1
