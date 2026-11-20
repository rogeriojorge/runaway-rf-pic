source ~/jic_step1/S1/env.sh
nohup python st_batch.py gauss45 1000 0.1 64 1 1.0 1.4 1.8 2.2 3.0 > ../data/s1/I3b_gauss45_p1e3.log 2>&1 &
nohup python st_batch.py eqN10 10000 0.1 32 1 1.6 2.2 3.0 > ../data/s1/I3b_eqN10_p1e4.log 2>&1 &
nohup python st_batch.py gauss45 10000 0.1 32 1 2.2 > ../data/s1/I3b_gauss45_p1e4.log 2>&1 &
