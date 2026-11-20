source ~/jic_step1/S1/env.sh
for s in 0.7 1.4; do for d in 0.15 0.075 0.0375 0.01875; do nohup python redrun.py eqN2 1 $s $d 128 0 > ../data/s1/L1_N2_s${s}_d${d}.log 2>&1 & done; done
