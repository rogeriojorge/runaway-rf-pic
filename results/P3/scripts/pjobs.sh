#!/bin/sh
cd "$(dirname "$0")"
export NPART=192 SEEDS=2 DTF=0.15
for lam in 1000 100 10 1; do
  python3 runPr.py eqN1 $lam 0 0.5 1.0 1.5 2.0 > ../data/Pr_eqN1_lam$lam.log 2>&1
  python3 runPr.py eqN2 $lam 0 0.7 1.0 1.4 2.0 > ../data/Pr_eqN2_lam$lam.log 2>&1
done
for lam in 1000 100 10; do
  python3 runPr.py eqN5 $lam 0 0.7 1.2 1.6 2.2 > ../data/Pr_eqN5_lam$lam.log 2>&1
done
python3 runPr.py eqN5 1 0 1.2 > ../data/Pr_eqN5_lam1.log 2>&1
