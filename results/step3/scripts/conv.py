import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import dd
def late(z, f0=0.5):
    t = z["ts"]; m = z["m2"]; k = int(f0*len(t)); return np.polyfit(t[k:], m[k:], 1)[0]/float(z["Dql"]), float(np.mean(m[-20:])/z["Lw"]**2*6)
print("window length (seed 1, NP=64): late-half slope/D_QL and fill, T=1000 (map, seed1) vs T=3000")
for sg in (1.5, 1.0):
    for G in (0.0, 0.01, 0.03, 0.1):
        a = np.load(f"{dd}/map_s{sg}_g{G}_seed1.npz"); b = np.load(f"{dd}/long_s{sg}_g{G}_seed1.npz")
        la, fa = late(a); lb, fb = late(b)
        print(f"sigma {sg} G {G}: T=1000 slope {la:.3f} fill {fa:.2f} | T=3000 slope {lb:.3f} fill {fb:.2f}")
print("dt (T=300, NP=64, seed 1): fill = m2(T)/(Lw^2/6); dtfac 0.1 vs 0.05 (noise streams differ)")
for sg in (1.5, 3.0):
    for G in (0.0, 0.03, 0.3):
        a = np.load(f"{dd}/dt10_s{sg}_g{G}_seed1.npz"); b = np.load(f"{dd}/dt05_s{sg}_g{G}_seed1.npz")
        print(f"sigma {sg} G {G}: fill {late(a)[1]:.2f} vs {late(b)[1]:.2f}; late slope {late(a)[0]:.2f} vs {late(b)[0]:.2f}")
