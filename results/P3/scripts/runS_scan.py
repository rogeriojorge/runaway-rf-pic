"""Stationary N-line scan. usage: runS_scan.py NAME [nuf]   (spectrum registry below); results to ../data/S_NAME.pkl"""
import sys, pickle
from p3lib import *
REG = {
 "eqN1": ("eq", 0.57, dict(N=1)), "eqN2":  ("eq", 0.57, dict(N=2)), "eqN3": ("eq", 0.57, dict(N=3)), "eqN5": ("eq", 0.57, dict(N=5)), "eqN10": ("eq", 0.57, dict(N=10)),
 "gauss34": ("gauss", 0.57, dict(sig_j=3.4)), "gauss45": ("gauss", 0.57, dict(sig_j=4.5)), "gauss70": ("gauss", 0.57, dict(sig_j=7.0)),
 "eqN10_D0285": ("eq", 0.285, dict(N=10)), "eqN10_D114": ("eq", 1.14, dict(N=10)), "eqN10_D228": ("eq", 2.28, dict(N=10)),
}
SIGS = np.geomspace(0.45, 3.0, 13)
def b_unit(Delta):
    ks, a = spectrum("eq", Delta, 5)
    s = local_numbers(ks, a, 1e-4)["sigma"][1]
    return 1e-4/s**2
if __name__ == "__main__":
    name = sys.argv[1]; nuf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    kind, Delta, kw = REG[name]
    ks, a = spectrum(kind, Delta, **kw)
    bu = b_unit(Delta)
    res = []
    for s in SIGS:
        b = bu*s**2
        rs = [run_S(ks, a, b, Delta, win_lines=(5 if kind == "gauss" else None), Np=192, seed=sd, nuf=nuf) for sd in (1, 2, 3)]
        rho = np.array([r["ratio"] for r in rs])
        d = dict(rs[0]); d["rho_seeds"] = rho; d["rho"] = float(rho.mean()); d["rho_err"] = float(rho.std(ddof=1)/np.sqrt(3))
        d["S_end_mean"] = float(np.mean([r["S_end"] for r in rs])); d["m2_mean"] = np.mean([r["m2"] for r in rs], 0)
        res.append(d)
        print(f"{name} nuf={nuf:g} sig={d['sigma']:.2f} b={b:.2e} rho={d['rho']:.3f}+-{d['rho_err']:.3f} S_end={d['S_end_mean']:.2f} beats={d['beats']:.0f} bounces={d['bounces']:.1f} th_rms={d['th_rms']:.3f} t={sum(r['time'] for r in rs):.0f}s", flush=True)
        pickle.dump(dict(name=name, Delta=Delta, nuf=nuf, res=res, bunit=bu), open(f"../data/S_{name}_f{nuf:g}.pkl", "wb"))
