source ~/jic_step1/S1/env.sh
for fr in 0 1; do for rr in 0 1; do FRIC=$fr RR=$rr nohup python borisrun.py eqN1 1000 0.7 1.0 256 5 20 > ../data/s1/V1_f${fr}r${rr}.log 2>&1 & FRIC=$fr RR=$rr nohup python redrun.py eqN1 1000 0.7 0.05 2048 20 > ../data/s1/V1red_f${fr}r${rr}.log 2>&1 & done; done
