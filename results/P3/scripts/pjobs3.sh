#!/bin/sh
cd "$(dirname "$0")"
export DTF=0.15
while pgrep -f pjobs2.sh > /dev/null; do sleep 20; done
cp ../data/Pr_eqN2_lam1_f0.pkl ../data/Pr_eqN2_lam1_a.pkl
NPART=64 SEEDS=1 python3 runPr.py eqN2 1 0 0.7 2.0 > ../data/Pr_eqN2_lam1b.log 2>&1
cp ../data/Pr_eqN2_lam1_f0.pkl ../data/Pr_eqN2_lam1_b.pkl
NPART=128 SEEDS=1 python3 runPr.py eqN5 1000 0 0.7 1.2 1.6 2.2 > ../data/Pr_eqN5_lam1000.log 2>&1
NPART=128 SEEDS=1 python3 runPr.py eqN5 100 0 0.7 1.2 1.6 2.2 > ../data/Pr_eqN5_lam100.log 2>&1
NPART=96 SEEDS=1 python3 runPr.py eqN5 10 0 1.2 2.2 > ../data/Pr_eqN5_lam10.log 2>&1
