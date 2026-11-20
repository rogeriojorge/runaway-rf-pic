source ~/jic_step1/S1/env.sh
rm -f ../data/s1/*
for nm in eqN1 eqN2; do for d in 2 1 0.5 0.25; do OUT=../data/s1 DTFAC=$d nohup python diag1.py $nm 1000 64 0.7 1.4 > ../data/s1/d1_${nm}_lam1000_dt$d.log 2>&1 & done; done
