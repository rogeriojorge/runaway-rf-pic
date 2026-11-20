source ~/jic_step1/S1/env.sh
KAPS=1 TAG=_k1fine OUT=../data/s1 nohup python item2.py 30 32 0.05,0.025 > ../data/s1/item2_lam30_k1fine.log 2>&1 &
KAPS=1 TAG=_k1 OUT=../data/s1 nohup python item2.py 10 32 0.1,0.05,0.025 > ../data/s1/item2_lam10_k1.log 2>&1 &
