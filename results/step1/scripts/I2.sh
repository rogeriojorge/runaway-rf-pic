source ~/jic_step1/S1/env.sh
for s in 1.6 2.2; do nohup python borisrun.py eqN5 100 $s 1.0 16 1 0 > ../data/s1/B5_lam100_s$s.log 2>&1 & done
