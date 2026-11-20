"""Unified passage set-up for the Boris (full-orbit) and reduced models: identical window, ICs and amplitude envelope.
Envelope: sin^2 ramp-up over tr = ramp_wb/omega_b at the start and ramp-down over tr at the end (ramp_wb=0: abrupt, as in P3)."""
from pcommon import *
import red_env
from pcommon import simulate_env

def setup(name, lam, sig, NP=64, seed=11, Xw=6.0, ramp_wb=0.0, Tfac=2.1, b_scale=1.0, Delta_scale=1.0, fric=1, rr=1, Xs=10.0, tr_sA=0.0, ic="start"):
    kind, Delta, kw = REG[name]; Delta = Delta*Delta_scale
    ks, a = spectrum(kind, Delta, **kw)
    ls, g, F, kin = line_data(ks); L = len(ks); ic = L//2
    G = np.array([k["G_per_F"] for k in kin]); ddl = np.array([abs(k["ddelta"]) for k in kin])
    A0, _, _ = drift.sweep_rate(case, ls[ic]["kz"], g[ic], TH0, scale=lam, fric_on=bool(fric), rr_on=bool(rr)); A = A0*Om**2
    Dd = abs(ls[1]["kz"]-ls[0]["kz"])*np.sqrt(g[ic]**2-1)*np.cos(TH0)/g[ic]*Om if L > 1 else 0.0
    b = b_unit(REG[name][1])*sig**2*b_scale          # sigma defined at the nominal Delta, as in P3
    wbj = np.sqrt(G*F*b*a*ddl)*Om; wb = float(wbj[ic])
    half = (L-1)/2 if kind == "eq" else 4
    tr = ramp_wb/wb if ramp_wb > 0 else 0.0
    if tr_sA > 0: tr = tr_sA/np.sqrt(A)
    X = (half+0.5)*Dd + max(Xs*np.sqrt(A), Xw*wb) + A*tr           # rad/s  (half-window in detuning)
    Xn = X/Om; Tn = Tfac*Xn/A0                                       # 1/Omega
    r = np.random.default_rng(seed)
    theta = r.uniform(TH0-0.01, TH0+0.01, NP)
    if ic == "start":
        gam = solve_gamma(case, ls[ic_]["kz"], np.full(NP, -Xn), theta) if False else solve_gamma(case, ls[ic]["kz"], np.full(NP, -Xn), theta)
    else:     # state at the resonance crossing has theta in [TH0-.01,TH0+.01]; integrate the control backwards to detuning -X
        gres = np.array([case.gamma_res(ls[ic]["kz"], -1, t_) for t_ in theta]); pr = np.sqrt(gres**2-1)
        ures = np.stack([pr*np.sin(theta), np.zeros(NP), -pr*np.cos(theta)], 1)
        Tb = Xn/A0
        for _ in range(3):
            ub = drift.propagate_control(case, ures, -Tb, scale=lam, nsteps=200)
            gb = np.sqrt(1+np.sum(ub**2, 1)); dl = np.mean(case.wn - ls[ic]["kz"]*ub[:, 2]/gb - 1/gb)
            Tb *= (-Xn)/dl
        gam = gb; theta = np.arctan2(np.hypot(ub[:, 0], ub[:, 1]), -ub[:, 2])
    p = np.sqrt(gam**2-1); phi = r.uniform(0, 2*np.pi, NP)
    zr = 2*np.pi/(Delta*pl.c/Om) if L > 1 else 2*np.pi/ls[0]["kz"]
    z0n = r.uniform(0, zr, NP)
    phis = r.uniform(0, 2*np.pi, L)
    return dict(Xs=Xs, fric=fric, rr=rr, name=name, lam=lam, sig=sig, ks=ks, a=a, ls=ls, L=L, ic=ic, b=b, wb=wb, wbj=wbj, A=A, A0=A0, Dd=Dd, X=X, Xn=Xn, Tn=Tn, tr=tr, NP=NP,
                theta=theta, gam=gam, p=p, phi=phi, z0n=z0n, phis=phis, Delta=Delta, R=wb**2/A, ddl=ddl, G=G)

def boris_run(S, dtfac=1.0, wave=True):
    dt = DT*dtfac; nst = int(S["Tn"]/dt); Tn = nst*dt; trn = S["tr"]*Om
    def env(t):
        if trn == 0.0: return 1.0
        return jnp.sin(0.5*jnp.pi*jnp.clip(t/trn, 0, 1))**2*jnp.sin(0.5*jnp.pi*jnp.clip((Tn-t)/trn, 0, 1))**2
    th, p, phi = S["theta"], S["p"], S["phi"]
    u = np.stack([p*np.sin(th)*np.cos(phi), p*np.sin(th)*np.sin(phi), -p*np.cos(th)], 1)
    x = np.stack([p*np.sin(th)*np.sin(phi), np.zeros(S["NP"]), S["z0n"]], 1)
    r = np.random.default_rng(S.get("seed_lines", 12345))
    lines, aa, _ = bw.make_lines(case, list(S["ks"]), list(S["b"]*S["a"]), (S["phis"]))
    lam = S["lam"]; E0z, fric = lam*case.EoverEc*case.Ec_n, lam*case.Ec_n*S["fric"]
    am = aa if wave else 0.0*aa
    fn = jax.jit(lambda x, u: simulate_env((x, u, 0.0), nst, dt, wn, lines, am, env, E0z=E0z, fric=fric, eps_rr=lam*S["rr"]))
    (xf, uf, _), _ = fn(jnp.asarray(x), jnp.asarray(u))
    return quantities(np.asarray(uf))[3], nst

def boris_dg(S, dtfac=1.0):
    t0 = time.time(); gw, nst = boris_run(S, dtfac, True); g0, _ = boris_run(S, dtfac, False)
    return dict(dg=gw-g0, g_wave=gw, g_b0=g0, nst=nst, dtfac=dtfac, time=time.time()-t0)

def red_dg(S, dtf=0.1, nuf=0.0, key=11, kappa=0.0, kD=1.0):
    dt = dtf/S["wb"]; Tsec = S["Tn"]/Om; nst = int(Tsec/dt); nrec = max(nst//64, 1); nst = nrec*64; tr = S["tr"]
    def envf(t):
        if tr == 0.0: return 1.0
        return jnp.sin(0.5*jnp.pi*jnp.clip(t/tr, 0, 1))**2*jnp.sin(0.5*jnp.pi*jnp.clip((nst*dt-t)/tr, 0, 1))**2
    ic = S["ic"]; phis = S["phis"]; uz0 = -S["p"]*np.cos(S["theta"]); z0 = S["z0n"]*pl.c/Om
    t0 = time.time()
    # the reduced model takes z in metres; Boris z in c/Omega
    red = red_env.Red(S["ks"], S["b"]*S["a"]); red0 = red_env.Red(S["ks"], 0.0*S["a"])
    for rd in (red, red0): rd.use_fric = float(S["fric"]); rd.use_rr = float(S["rr"]); rd.kappa = kappa; rd.kD = kD; rd.exactpar = bool(int(os.environ.get('EXACTPAR', 0)))
    f1 = jax.jit(lambda: red.run(S["gam"], uz0, S["phi"], z0, phis-phis[ic], dt, nst, nrec, lam=S["lam"], nuf=nuf, drive=True, key=key, envf=envf))
    f0 = jax.jit(lambda: red0.run(S["gam"], uz0, S["phi"], z0, phis-phis[ic], dt, nst, nrec, lam=S["lam"], nuf=nuf, drive=True, key=key, envf=envf))
    gf = np.asarray(f1()[0][0]); gc = np.asarray(f0()[0][0])
    return dict(dg=gf-gc, g_wave=gf, g_ctl=gc, nst=nst, dtf=dtf, time=time.time()-t0)

def stats(d):
    dg = d["dg"]; return f"rms={np.sqrt(np.mean(dg**2)):.5f} mean={dg.mean():+.5f} std={dg.std():.5f}"

def ql_value(S):
    """Stationary-phase quasilinear <dgamma^2> per passage summed over the lines crossed: sum_j pi (G F b a_j Om)^2 / A  (F at theta0=0.2, as in P3)."""
    ls, g, F, kin = line_data(S["ks"])
    G = np.array([k["G_per_F"] for k in kin])
    return float(np.sum(np.pi*(G*F*S["b"]*S["a"]*Om)**2/S["A"]))
