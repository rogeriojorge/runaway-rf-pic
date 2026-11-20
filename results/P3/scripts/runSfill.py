"""Band-fill fraction S = <dgamma^2>(T)/(L_c^2/6) at T = 300/omega_b (P2 convention), N equal lines, reduced model."""
import pickle
from val_red import *
from runS_scan import REG, SIGS, b_unit
out = {}
for name in ("eqN2", "eqN3", "eqN5", "eqN10"):
    kind, Delta, kw = REG[name]; ks, a = spectrum(kind, Delta, **kw); bu = b_unit(Delta); res = []
    for s in SIGS:
        b = bu*s**2
        wb = local_numbers(ks, a, b)["wb"][len(ks)//2 - (1 if len(ks) % 2 == 0 else 0)]
        rs = [run_S_red(ks, a, b, Delta, Np=384, seed=sd, Tfix=300/wb, nrec=48) for sd in (1, 2, 3, 4)]
        S = np.array([r["S_end"] for r in rs]); res.append((rs[0]["sigma"], S.mean(), S.std(ddof=1)/2))
        print(name, f"sig={res[-1][0]:.2f} S300={res[-1][1]:.3f}+-{res[-1][2]:.3f}", flush=True)
    out[name] = res
pickle.dump(out, open("../data/Sfill.pkl", "wb"))
