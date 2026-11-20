source ~/jic_step1/S1/env.sh
SIGS="1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.6 3.0 4.0"
for nm in gauss45 eqN10; do for sd in 1 2; do nohup python st_batch.py $nm 1000 0.1 128 $sd $SIGS > ../data/s1/I3_${nm}_p1e3_s$sd.log 2>&1 & done; done
for nm in gauss45 eqN10; do nohup python st_batch.py $nm 10000 0.1 64 1 1.2 1.6 2.0 2.2 2.6 3.0 4.0 > ../data/s1/I3_${nm}_p1e4_s1.log 2>&1 & done
nohup python st_batch.py eqN5 1000 0.1 128 1 $SIGS > ../data/s1/I3_eqN5_p1e3_s1.log 2>&1 &
