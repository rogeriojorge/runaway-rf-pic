source ~/jic_step1/S1/env.sh
for s in 0.7 1.4; do nohup python borisrun.py eqN2 1000 $s 1.0 128 1 0 > ../data/s1/B2_N2_s$s.log 2>&1 & done
for s in 1.2 1.6 2.2; do nohup python borisrun.py eqN5 1000 $s 1.0 64 1 0 > ../data/s1/B2_N5_s$s.log 2>&1 & done
