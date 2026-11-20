"""Passage protocol: ensemble swept by E_par + friction + LL radiation reaction (time-compressed by lam_A, amplitude NOT rescaled) through the whole N-line spectrum.
Metric: <(gamma_f(b)-gamma_f(0))^2> against the QL sum  sum_j pi (G F b a_j)^2/|d delta/dt|  over lines whose resonance is crossed.
usage: runP.py NAME LAMBDA sig1 sig2 ... (sigma-nominal values; b = b_unit sigma^2)   NAME as in runS_scan.REG"""
import sys, pickle
from p3lib import *
import drift
from runS_scan import REG, b_unit
name = sys.argv[1]; lam = float(sys.argv[2]); sigs = [float(s) for s in sys.argv[3:]]
NP = int(os.environ.get("NPART", 128)); JSW = int(os.environ.get("JSW", 4))
kind, Delta, kw = REG[name]
ks, a = spectrum(kind, Delta, **kw)
if kind == "gauss":       # sweep only the central 2*JSW+1 lines; the others stay in the field
    pass
ls, g, F, kin = line_data(ks)
L = len(ks); ic = L//2
G = np.array([k["G_per_F"] for k in kin])
A0, _, _ = drift.sweep_rate(case, ls[ic]["kz"], g[ic], TH0, scale=lam); A = A0*Om**2          # rad/s^2
Dd = abs(ls[1]["kz"]-ls[0]["kz"])*np.sqrt(g[ic]**2-1)*np.cos(TH0)/g[ic]*Om if L > 1 else 1e9
bu = b_unit(Delta)
print(f"{name} lam={lam:g} A={A:.3e} rad/s^2 Dd={Dd:.3e} Q=Dd^2/A={Dd**2/A:.3g}", flush=True)
out = []
for s in sigs:
    b = bu*s**2
    wbj = np.sqrt(G*F*b*a*np.array([abs(k["ddelta"]) for k in kin]))*Om
    wbm = wbj.max()
    # swept lines
    half = (L-1)/2 if kind == "eq" else JSW
    X = (half+0.5)*Dd + max(10*np.sqrt(A), 6*wbj[ic])          # rad/s
    Xn = X/Om; T = 2.1*Xn/(A0)                                  # 1/Omega
    nst = int(T/DT); nrec = max(nst//48, 1); nst = nrec*48
    est = nst*NP*L/8e7
    R = wbj[ic]**2/A
    print(f"  sigma={s:.2f} b={b:.2e} wb={wbj[ic]:.2e} R={R:.3g} steps={nst:.2e} est={est:.0f}s", flush=True)
    if os.environ.get("DRY"): continue
    r = np.random.default_rng(11)
    theta = r.uniform(TH0-0.01, TH0+0.01, NP)
    gam = solve_gamma(case, ls[ic]["kz"], np.full(NP, -Xn), theta)
    p = np.sqrt(gam**2-1); phi = r.uniform(0, 2*np.pi, NP)
    zr = 2*np.pi/(Delta*pl.c/Om) if L > 1 else 2*np.pi/ls[0]["kz"]
    z0 = r.uniform(0, zr, NP)
    u = np.stack([p*np.sin(theta)*np.cos(phi), p*np.sin(theta)*np.sin(phi), -p*np.cos(theta)], 1)
    x = np.stack([p*np.sin(theta)*np.sin(phi), np.zeros(NP), z0], 1)
    lines, aa, _ = bw.make_lines(case, list(ks), list(b*a), r.uniform(0, 2*np.pi, L))
    E0z, fric = lam*case.EoverEc*case.Ec_n, lam*case.Ec_n
    t0 = time.time()
    fn = jax.jit(lambda x, u: bw.simulate((x, u, 0.0), nst, nst, DT, wn, lines, aa, E0z=E0z, fric=fric, eps_rr=lam, rec_fn=rec_u))
    st, rec = fn(jnp.asarray(x), jnp.asarray(u)); uf = np.asarray(rec)[-1]
    uc = drift.propagate_control(case, u, nst*DT, scale=lam)
    _, xi_f, th_f, g_f = quantities(uf); _, xi_c, th_c, g_c = quantities(uc)
    dg = g_f-g_c
    # QL sum over crossed lines
    u0 = u; g_s = np.sqrt(1+np.sum(u0**2, 1)); g_e = np.sqrt(1+np.sum(uc**2, 1))
    # detuning of each line at start and end of the control orbit (mean particle)
    d0 = np.array([np.mean(delta_of(case, l["kz"], g_s, theta)) for l in ls])
    thc_e = np.arctan2(np.sqrt(uc[:, 0]**2+uc[:, 1]**2), -uc[:, 2])
    d1 = np.array([np.mean(case.wn - l["kz"]*uc[:, 2]/g_e - 1.0/g_e) for l in ls])
    crossed = (d0 < 0) & (d1 > 0)
    QL = np.sum((np.pi*(G*F*b*a*Om)**2/A)[crossed])
    m2 = float(np.mean(dg**2)); dxi = xi_f-xi_c
    res = dict(sigma=s, b=b, wb=float(wbj[ic]), R=float(R), A=A, Dd=Dd, lam=lam, ncross=int(crossed.sum()), QL=float(QL), m2=m2, rho=m2/QL, rho_err=float(np.std(dg**2)/np.sqrt(NP)/QL),
               mean_dg=float(dg.mean()), rms_dxi=float(np.sqrt(np.mean(dxi**2))), time=time.time()-t0, name=name, dg=dg)
    out.append(res)
    print(f"  -> crossed={res['ncross']} rho_P={res['rho']:.3f}+-{res['rho_err']:.3f} mean dg={res['mean_dg']:+.3e} ({res['time']:.0f}s)", flush=True)
    pickle.dump(out, open(f"../data/P_{name}_lam{lam:g}.pkl", "wb"))
