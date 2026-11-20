"""Raw QL-limit check at sigma = 0.5 (band-filling negligible: fill <= 0.4): late-window slope of <dg^2> over [T/2, T] / D_QL, per seed, and the coherent value."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load
M = load("map")
for sg in (0.5,):
    for G in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        r = []
        for z in M[(sg, G)]:
            t = z["ts"]; m = z["m2"]; k = len(t)//2; p = np.polyfit(t[k:], m[k:], 1); r.append(p[0]/float(z["Dql"]))
        r = np.array(r); print(f"sigma {sg} Gamma/Delta {G:5.2f}: slope/D_QL = {r.mean():.3f} +- {r.std(ddof=1)/np.sqrt(len(r)):.3f}  (seeds {np.round(r,3)})")
