source ~/jic_step1/S1/env.sh
nohup python borisrun.py eqN5 30 2.2 1.0 8 1 0 > ../data/s1/B5_lam30_s2.2.log 2>&1 &
nohup python borisrun.py eqN5 10 2.2 1.0 4 1 0 > ../data/s1/B5_lam10_s2.2.log 2>&1 &
