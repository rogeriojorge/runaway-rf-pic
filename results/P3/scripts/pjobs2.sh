#!/bin/sh
cd "$(dirname "$0")"
export DTF=0.15
NPART=96 SEEDS=1 python3 runPr.py eqN2 10 0 0.7 1.4 > ../data/Pr_eqN2_lam10.log 2>&1
NPART=64 SEEDS=1 python3 runPr.py eqN2 1 0 1.4 > ../data/Pr_eqN2_lam1.log 2>&1
