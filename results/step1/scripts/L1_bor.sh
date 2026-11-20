source ~/jic_step1/S1/env.sh
for d in 0.5 0.25 0.125 0.0625; do for sd in 1 2 3 4; do nohup python borisrun.py eqN1 1000 0.7 $d 128 $sd 20 > ../data/s1/B1_d${d}_sd${sd}.log 2>&1 & done; done
