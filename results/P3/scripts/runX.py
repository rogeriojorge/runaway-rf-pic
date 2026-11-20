"""Collisions (nuf = multiplier of DIII-D nu_D) and phase-noise (linewidth) scans, onset-window metric, reduced model.
usage: runX.py coll|noise NAME   -> ../data/X_{kind}_{NAME}.pkl"""
import sys, pickle
from val_red import *
from runS_scan import REG, b_unit
kind_x, name = sys.argv[1], sys.argv[2]
kind, Delta, kw = REG[name]; ks, a = spectrum(kind, Delta, **kw); bu = b_unit(Delta); wl = 5 if kind == "gauss" else None
SIGS = [0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
if kind_x == "coll":
    LEV = [0.0, 1.0, 1e2, 1e3, 1e4, 1e5, 3e5, 1e6, 3e6, 1e7]
else:
    LEV = [0.0, 1e6, 3e6, 1e7, 3e7, 1e8]        # phase-diffusion rate Gamma_phi (1/s)
out = []
for s in SIGS:
    b = bu*s**2
    for lv in LEV:
        kwx = dict(nuf=lv) if kind_x == "coll" else (dict(gcom=lv) if kind_x == "noise" else dict(gind=lv))
        rs = [run_S_red(ks, a, b, Delta, win_lines=wl, Np=384, seed=sd, **kwx) for sd in (1, 2, 3, 4)]
        rho = np.array([r["ratio"] for r in rs])
        d = dict(sigma_nom=s, b=b, level=lv, rho=float(rho.mean()), rho_err=float(rho.std(ddof=1)/np.sqrt(4)), sigma=rs[0]["sigma"], T=rs[0]["T"], S_end=float(np.mean([r["S_end"] for r in rs])))
        out.append(d); print(f"{kind_x} {name} sig={s} lvl={lv:g} rho={d['rho']:.3f}+-{d['rho_err']:.3f} S_end={d['S_end']:.2f} T={d['T']:.2e}", flush=True)
    pickle.dump(out, open(f"../data/X_{kind_x}_{name}.pkl", "wb"))
