source ~/jic_step1/S1/env.sh
mkdir -p ../data/s1
for s in 0.7 1.4; do for d in 0.2 0.1 0.05; do KAPPA=1 nohup python redrun.py eqN2 1 $s $d 64 0 > ../data/s1/L1s_N2_s${s}_d${d}.log 2>&1 & done; done
for lam in 100 30 10; do KAPS=1 TAG=_strang OUT=../data/s1 nohup python item2.py $lam 32 0.2,0.1,0.05 > ../data/s1/item2s_lam$lam.log 2>&1 & done
