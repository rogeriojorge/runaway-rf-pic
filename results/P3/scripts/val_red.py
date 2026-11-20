"""Reduced model vs full-orbit Boris: stationary N-line scan (same ICs / metric as run_S)."""
import sys, pickle
from red import *
from runS_scan import REG, b_unit
def run_S_red(ks, amps, b, Delta, win_lines=None, Np=192, seed=1, nuf=0.0, nrec=96, Tfac_cap=300., Tmin_beats=10., fill_frac=0.25, dtfrac=0.1, th0=TH0, Tfix=None, gcom=0.0, gind=0.0):
    ks = np.asarray(ks); L = len(ks); amps = np.asarray(amps)
    nl = local_numbers(ks, amps, b, th0)
    ls, g, F, kin = line_data(ks, th0)
    G = np.array([k["G_per_F"] for k in kin]); ic = L//2
    if win_lines is None: win_lines = L
    gsp = abs(np.mean(np.diff(g))); gc = g[ic] if L % 2 == 1 else 0.5*(g[L//2-1]+g[L//2])
    gwin = (gc-0.5*win_lines*gsp, gc+0.5*win_lines*gsp)
    inside = [j for j in range(L) if gwin[0] <= g[j] <= gwin[1]]
    Dd = np.mean(nl["Dd"]); ddel_I = abs(kin[ic]['ddelta'])*Om
    wbj = np.sqrt(G*F*b*amps*np.array([abs(k["ddelta"]) for k in kin]))*Om; wb_c = wbj[ic]
    rate = np.pi*(G*F*b*amps*Om)**2; Wd = win_lines*Dd
    Dql = sum(rate[j] for j in inside)/Wd; Lw = Wd/ddel_I
    t_fill = fill_frac*(Lw**2/6)/Dql; t_cap = Tfac_cap/wb_c
    T = Tfix if Tfix else max(min(t_fill, t_cap), Tmin_beats/Dd)
    dt = dtfrac/wbj.max()
    nst = int(T/dt); nr = max(nst//nrec, 1); nst = nr*nrec
    r = np.random.default_rng(seed+100); phis = r.uniform(0, 2*np.pi, L)
    rr = np.random.default_rng(seed)
    theta = rr.uniform(th0-0.01, th0+0.01, Np); g0 = rr.uniform(*gwin, Np); p = np.sqrt(g0**2-1)
    phi = rr.uniform(0, 2*np.pi, Np); z0m = rr.uniform(0, 2*np.pi/Delta, Np)
    red = Red(ks, b*amps)
    t0 = time.time()
    fn = jax.jit(lambda: red.run(g0, -p*np.cos(theta), phi, z0m, phis - phis[ic], dt, nst, nr, nuf=nuf, gcom=gcom, gind=gind))
    carry, rec = fn(); rec = np.asarray(rec)
    dg = rec-g0[None, :]; ts = np.arange(1, nrec+1)*nr*dt; m2 = np.mean(dg**2, 1); h = nrec//2
    sl = np.polyfit(ts[h:], m2[h:], 1)[0]
    return dict(b=b, sigma=float(np.mean(nl["sigma"][max(0, ic-2):ic+2])), ratio=float(sl/Dql), S_end=float(m2[-1]/(Lw**2/6)), T=T, nst=nst, time=time.time()-t0, Dql=float(Dql), ts=ts, m2=m2)
if __name__ == "__main__":
    name = sys.argv[1]; kind, Delta, kw = REG[name]; ks, a = spectrum(kind, Delta, **kw); bu = b_unit(Delta)
    for s in [float(x) for x in sys.argv[2:]]:
        b = bu*s**2
        rs = [run_S_red(ks, a, b, Delta, win_lines=(5 if kind == "gauss" else None), seed=sd) for sd in (1, 2, 3)]
        print(f"RED {name} sig={rs[0]['sigma']:.2f} b={b:.2e} rho={np.mean([r['ratio'] for r in rs]):.3f}+-{np.std([r['ratio'] for r in rs], ddof=1)/np.sqrt(3):.3f} S_end={np.mean([r['S_end'] for r in rs]):.2f} nst={rs[0]['nst']:.2e} t={sum(r['time'] for r in rs):.0f}s", flush=True)
