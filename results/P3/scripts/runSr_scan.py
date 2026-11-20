"""Reduced-model stationary scans (validated against Boris, val_red.py).  usage: runSr_scan.py NAME [nuf]   -> ../data/Sr_NAME_f{nuf}.pkl"""
import sys, pickle
from val_red import *
from runS_scan import REG, SIGS, b_unit
def scan(name, nuf=0.0, gcom=0.0, gind=0.0, sigs=SIGS, Np=384, seeds=(1, 2, 3, 4), long_beats=(100., 400.), Delta_scale=1.0, dtl=0.0125, **kw_over):
    kind, Delta, kw = REG[name]; Delta = Delta*Delta_scale
    ks, a = spectrum(kind, Delta, **kw); bu = b_unit(Delta)
    wl = 5 if kind == "gauss" else None
    res = []
    for s in sigs:
        b = bu*s**2
        d = dict(sigma_nom=float(s), b=b)
        rs = [run_S_red(ks, a, b, Delta, win_lines=wl, Np=Np, seed=sd, nuf=nuf, gcom=gcom, gind=gind, **{**dict(dtfrac=0.04), **kw_over}) for sd in seeds]
        d["sigma"] = rs[0]["sigma"]; rho = np.array([r["ratio"] for r in rs]); d["rho"] = float(rho.mean()); d["rho_err"] = float(rho.std(ddof=1)/np.sqrt(len(rho)))
        d["S_end"] = float(np.mean([r["S_end"] for r in rs])); d["T"] = rs[0]["T"]; d["Dql"] = rs[0]["Dql"]
        Dd = np.mean(local_numbers(ks, a, b)["Dd"])
        for lb in long_beats:
            rl = [run_S_red(ks, a, b, Delta, win_lines=wl, Np=Np, seed=sd, nuf=nuf, gcom=gcom, gind=gind, Tfix=lb/Dd, dtfrac=dtl, **kw_over) for sd in seeds[:2]]
            d[f"rho_{int(lb)}"] = float(np.mean([r["ratio"] for r in rl])); d[f"m2end_{int(lb)}"] = float(np.mean([r["m2"][-1] for r in rl]))
            d[f"curve_{int(lb)}"] = (rl[0]["ts"], np.mean([r["m2"] for r in rl], 0))
        res.append(d)
        print(f"{name} f={nuf:g} sig={d['sigma']:.2f} b={b:.2e} rho={d['rho']:.3f}+-{d['rho_err']:.3f} S_end={d['S_end']:.2f} " + " ".join(f"rho{int(lb)}={d[f'rho_{int(lb)}']:.2f}" for lb in long_beats), flush=True)
    return dict(name=name, Delta=Delta, nuf=nuf, res=res, bunit=bu)
if __name__ == "__main__":
    name = sys.argv[1]; nuf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    out = scan(name, nuf, long_beats=((100., 400.) if name in ('eqN10', 'gauss45') else ()))
    pickle.dump(out, open(f"../data/Sr2_{name}_f{nuf:g}.pkl", "wb"))
