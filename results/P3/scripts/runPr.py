"""Passage protocol with the reduced model (same ICs/metric as runP.py).  usage: runPr.py NAME LAMBDA [nuf] sig1 sig2 ...  (env NPART, JSW, DTF, SEEDS)"""
import sys, pickle
from red import *
import drift
from runS_scan import REG, b_unit
def P_red(name, lam, sigs, nuf=0.0, NP=256, JSW=4, dtf=0.1, seeds=(11,), verbose=True, Delta_scale=1.0, kw_over=None):
    kind, Delta, kw = REG[name]; Delta = Delta*Delta_scale
    ks, a = spectrum(kind, Delta, **kw)
    ls, g, F, kin = line_data(ks); L = len(ks); ic = L//2
    G = np.array([k["G_per_F"] for k in kin]); ddl = np.array([abs(k["ddelta"]) for k in kin])
    A0, _, _ = drift.sweep_rate(case, ls[ic]["kz"], g[ic], TH0, scale=lam); A = A0*Om**2
    Dd = abs(ls[1]["kz"]-ls[0]["kz"])*np.sqrt(g[ic]**2-1)*np.cos(TH0)/g[ic]*Om if L > 1 else 1e9
    bu = b_unit(Delta); out = []
    for s in sigs:
        b = bu*s**2
        wbj = np.sqrt(G*F*b*a*ddl)*Om; wb = wbj.max()
        half = (L-1)/2 if kind == "eq" else JSW
        X = (half+0.5)*(Dd if L > 1 else 0.0) + max(10*np.sqrt(A), 6*wbj[ic]); Xn = X/Om; T_n = 2.1*Xn/A0
        dt = dtf/wb; Tsec = T_n/Om; nst = int(Tsec/dt); nrec = max(nst//64, 1); nst = nrec*64
        R = wbj[ic]**2/A
        rhos = []; m2s = []; dgs = []
        t0 = time.time()
        for sd in seeds:
            r = np.random.default_rng(sd)
            theta = r.uniform(TH0-0.01, TH0+0.01, NP)
            gam = solve_gamma(case, ls[ic]["kz"], np.full(NP, -Xn), theta); p = np.sqrt(gam**2-1)
            phi = r.uniform(0, 2*np.pi, NP); z0 = r.uniform(0, 2*np.pi/Delta, NP)
            phis = r.uniform(0, 2*np.pi, L)
            red = Red(ks, b*a)
            uz0 = -p*np.cos(theta)
            fn = jax.jit(lambda: red.run(gam, uz0, phi, z0, phis-phis[ic], dt, nst, nrec, lam=lam, nuf=nuf, drive=True, key=sd))
            (gf, uzf, _, _, _, _), _ = fn(); gf = np.asarray(gf); uzf = np.asarray(uzf)
            red0 = Red(ks, 0.0*a)
            fn0 = jax.jit(lambda: red0.run(gam, uz0, phi, z0, phis-phis[ic], dt, nst, nrec, lam=lam, nuf=nuf, drive=True, key=sd))
            (gc, uzc, _, _, _, _), _ = fn0(); gc = np.asarray(gc); uzc = np.asarray(uzc)
            uc = np.stack([np.zeros(NP), np.zeros(NP), uzc], 1)
            dg = gf-gc; dgs.append(dg)
            g_s = gam; g_e = gc
            d0 = np.array([np.mean(delta_of(case, l["kz"], g_s, theta)) for l in ls]); d1 = np.array([np.mean(case.wn - l["kz"]*uc[:, 2]/g_e - 1.0/g_e) for l in ls])
            crossed = (d0 < 0) & (d1 > 0)
            QL = np.sum((np.pi*(G*F*b*a*Om)**2/A)[crossed])
            m2s.append(np.mean(dg**2)); rhos.append(m2s[-1]/QL)
        d = dict(sigma=s, b=b, wb=float(wbj[ic]), R=float(R), Q=float(Dd**2/A), A=A, lam=lam, ncross=int(crossed.sum()), QL=float(QL), rho=float(np.mean(rhos)), rho_err=float(np.std(rhos, ddof=1)/np.sqrt(len(rhos))) if len(rhos) > 1 else float(np.std(dg**2)/np.sqrt(NP)/QL),
                 mean_dg=float(np.mean(dgs[0])), m2=float(np.mean(m2s)), Lc=float(L*Dd/(Om*ddl[ic])) if L>1 else 0.0, rms_dg=float(np.sqrt(np.mean(m2s))), nst=nst, time=time.time()-t0, name=name, nuf=nuf, Delta=Delta)
        out.append(d)
        if verbose: print(f"{name} lam={lam:g} nuf={nuf:g} sig={s:.2f} b={b:.2e} R={R:.3g} Q={d['Q']:.3g} ncross={d['ncross']} rho_P={d["rho"]:.4f}+-{d["rho_err"]:.4f} rms={np.sqrt(d["m2"]):.4f} mean dg={d['mean_dg']:+.2e} steps={nst:.2e} ({d['time']:.0f}s)", flush=True)
    return out
if __name__ == "__main__":
    name = sys.argv[1]; lam = float(sys.argv[2]); nuf = float(sys.argv[3]); sigs = [float(x) for x in sys.argv[4:]]
    out = P_red(name, lam, sigs, nuf=nuf, NP=int(os.environ.get("NPART", 256)), JSW=int(os.environ.get("JSW", 4)), dtf=float(os.environ.get("DTF", 0.1)), seeds=tuple(range(11, 11+int(os.environ.get("SEEDS", 1)))))
    pickle.dump(out, open(f"../data/Pr_{name}_lam{lam:g}_f{nuf:g}{os.environ.get('TAG','')}.pkl", "wb"))
