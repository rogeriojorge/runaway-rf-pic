"""Step-1 common: locate P2/P3 script dirs (laptop or office copy) and import p3lib."""
import os, sys
for d in ("results/P3/scripts", "~/jic_step1/P3scripts"):
    d = os.path.expanduser(d)
    if os.path.isdir(d): sys.path.insert(0, d); break
os.environ.setdefault("JAX_ENABLE_X64", "1")
from p3lib import *            # noqa
import drift
from runS_scan import REG, b_unit
import pickle, time, json

def boris_passage(name, lam, sig, dtfac=1.0, NP=64, JSW=4, seed=11, b_scale=1.0, controls=True, window_extra=1.0, Tfac=2.1):
    """One full-orbit passage (P3/runP.py protocol) at overlap sigma.  Returns per-particle final gamma for: wave run, same-integrator zero-wave run, RK4 control.
    dtfac multiplies the production Boris step DT = T_gyro/32.  Same ICs for all runs."""
    kind, Delta, kw = REG[name]
    ks, a = spectrum(kind, Delta, **kw)
    ls, g, F, kin = line_data(ks); L = len(ks); ic = L//2
    G = np.array([k["G_per_F"] for k in kin])
    A0, _, _ = drift.sweep_rate(case, ls[ic]["kz"], g[ic], TH0, scale=lam); A = A0*Om**2
    Dd = abs(ls[1]["kz"]-ls[0]["kz"])*np.sqrt(g[ic]**2-1)*np.cos(TH0)/g[ic]*Om if L > 1 else 1e9
    b = b_unit(Delta)*sig**2*b_scale
    wbj = np.sqrt(G*F*b*a*np.array([abs(k["ddelta"]) for k in kin]))*Om
    half = (L-1)/2 if kind == "eq" else JSW
    X = (half+0.5)*Dd + max(10*np.sqrt(A), 6*wbj[ic]); Xn = X/Om
    T = Tfac*Xn/A0
    dt = DT*dtfac
    nst = int(T/dt); nst = max(nst, 1)
    r = np.random.default_rng(seed)
    theta = r.uniform(TH0-0.01, TH0+0.01, NP)
    gam = solve_gamma(case, ls[ic]["kz"], np.full(NP, -Xn), theta)
    p = np.sqrt(gam**2-1); phi = r.uniform(0, 2*np.pi, NP)
    zr = 2*np.pi/(Delta*pl.c/Om) if L > 1 else 2*np.pi/ls[0]["kz"]
    z0 = r.uniform(0, zr, NP)
    u = np.stack([p*np.sin(theta)*np.cos(phi), p*np.sin(theta)*np.sin(phi), -p*np.cos(theta)], 1)
    x = np.stack([p*np.sin(theta)*np.sin(phi), np.zeros(NP), z0], 1)
    lines, aa, _ = bw.make_lines(case, list(ks), list(b*a), r.uniform(0, 2*np.pi, L))
    E0z, fric = lam*case.EoverEc*case.Ec_n, lam*case.Ec_n
    def run(amps):
        fn = jax.jit(lambda x, u: bw.simulate((x, u, 0.0), nst, nst, dt, wn, lines, amps, E0z=E0z, fric=fric, eps_rr=lam, rec_fn=rec_u))
        st, rec = fn(jnp.asarray(x), jnp.asarray(u)); return np.asarray(rec)[-1]
    t0 = time.time()
    uf = run(aa); t1 = time.time()
    out = dict(name=name, lam=lam, sigma=sig, b=b, wb=float(wbj[ic]), R=float(wbj[ic]**2/A), A=A, Dd=Dd, dt=dt, dtfac=dtfac, nst=nst, NP=NP, T=T*1.0, time_wave=t1-t0,
               g_wave=quantities(uf)[3], xi_wave=quantities(uf)[1], th_wave=quantities(uf)[2])
    if controls:
        u0f = run(0.0*aa)
        out["g_b0"] = quantities(u0f)[3]; out["xi_b0"] = quantities(u0f)[1]
        uc = drift.propagate_control(case, u, nst*dt, scale=lam)
        out["g_rk4"] = quantities(uc)[3]; out["xi_rk4"] = quantities(uc)[1]
    out["g_start"] = np.sqrt(1+np.sum(u**2, 1))
    return out

def summarize(o):
    s = dict(name=o["name"], lam=o["lam"], sigma=o["sigma"], R=o["R"], dtfac=o["dtfac"], NP=o["NP"])
    dgA = o["g_wave"]-o["g_rk4"]; dgB = o["g_wave"]-o["g_b0"]; dg0 = o["g_b0"]-o["g_rk4"]
    for k, v in (("old(wave-RK4)", dgA), ("new(wave-b0)", dgB), ("bias(b0-RK4)", dg0)):
        s[k] = dict(rms=float(np.sqrt(np.mean(v**2))), mean=float(v.mean()), std=float(v.std()))
    return s

def red_passage(name, lam, sig, dtf=0.1, NP=256, seed=11, JSW=4, b_scale=1.0, Delta_scale=1.0, nuf=0.0, Tfac=2.1, retstate=False):
    """Reduced-model passage, same protocol/ICs as boris_passage.  dtf = omega_b*dt.  Returns per-particle dg = g(wave)-g(b=0 reduced) plus diagnostics."""
    from red import Red
    kind, Delta, kw = REG[name]; Delta = Delta*Delta_scale
    ks, a = spectrum(kind, Delta, **kw)
    ls, g, F, kin = line_data(ks); L = len(ks); ic = L//2
    G = np.array([k["G_per_F"] for k in kin]); ddl = np.array([abs(k["ddelta"]) for k in kin])
    A0, _, _ = drift.sweep_rate(case, ls[ic]["kz"], g[ic], TH0, scale=lam); A = A0*Om**2
    Dd = abs(ls[1]["kz"]-ls[0]["kz"])*np.sqrt(g[ic]**2-1)*np.cos(TH0)/g[ic]*Om if L > 1 else 1e9
    b = b_unit(Delta*1.0 if Delta_scale == 1.0 else Delta/Delta_scale)*sig**2*b_scale
    wbj = np.sqrt(G*F*b*a*ddl)*Om; wb = wbj.max()
    half = (L-1)/2 if kind == "eq" else JSW
    X = (half+0.5)*(Dd if L > 1 else 0.0) + max(10*np.sqrt(A), 6*wbj[ic]); Xn = X/Om; T_n = 2.1*Xn/A0 if Tfac == 2.1 else Tfac*Xn/A0
    dt = dtf/wb; Tsec = T_n/Om; nst = int(Tsec/dt); nrec = max(nst//64, 1); nst = nrec*64
    r = np.random.default_rng(seed)
    theta = r.uniform(TH0-0.01, TH0+0.01, NP)
    gam = solve_gamma(case, ls[ic]["kz"], np.full(NP, -Xn), theta); p = np.sqrt(gam**2-1)
    phi = r.uniform(0, 2*np.pi, NP); z0 = r.uniform(0, 2*np.pi/Delta, NP); phis = r.uniform(0, 2*np.pi, L)
    uz0 = -p*np.cos(theta)
    t0 = time.time()
    red = Red(ks, b*a); (gf, uzf, *_), _ = jax.jit(lambda: red.run(gam, uz0, phi, z0, phis-phis[ic], dt, nst, nrec, lam=lam, nuf=nuf, drive=True, key=seed))()
    red0 = Red(ks, 0.0*a); (gc, uzc, *_), _ = jax.jit(lambda: red0.run(gam, uz0, phi, z0, phis-phis[ic], dt, nst, nrec, lam=lam, nuf=nuf, drive=True, key=seed))()
    gf, gc, uzf, uzc = map(np.asarray, (gf, gc, uzf, uzc))
    return dict(name=name, lam=lam, sigma=sig, b=b, wb=float(wbj[ic]), R=float(wbj[ic]**2/A), A=A, dtf=dtf, nst=nst, NP=NP, g_wave=gf, g_ctl=gc, dg=gf-gc, time=time.time()-t0, g_start=gam)

# ---------------------------------------------------------------- Boris with an amplitude envelope (ramp up and down)
from jax import lax
def simulate_env(state, nsteps, dt, omega, lines, amps, env, E0z=0.0, fric=0.0, eps_rr=0.0, nrec=None, rec_fn=None):
    """Same push as bw.simulate (Boris tan-exact + explicit E_par/friction/LL), wave amplitudes multiplied by env(t) (t in 1/Omega)."""
    nrec = nrec or nsteps
    def step(carry, _):
        x, u, t = carry
        fields = bw.make_fields(lines, amps*env(t), omega)
        E, B = fields(x, t)
        E = E.at[:, 2].add(E0z); B = B.at[:, 2].add(1.0)
        un = bw.boris_push(x, u, t, dt, E, B)
        if eps_rr != 0.0:
            du_si, _p = bw.landau_lifshitz(un*bw.C_LIGHT, E*bw.B0_REF*bw.C_LIGHT, B*bw.B0_REF, -bw.E_CHARGE, bw.M_E)
            un = un + dt*eps_rr*du_si/(bw.C_LIGHT*bw.OM_REF)
        if fric != 0.0:
            un2 = jnp.sum(un*un, axis=-1); gh2 = 1.0 + un2
            un = un - dt*fric*(gh2/un2)[:, None]*un/jnp.sqrt(un2)[:, None]
        xn = x + dt*un/bw.lorentz_gamma(un)[:, None]
        return (xn, un, t+dt), None
    def block(carry, _):
        carry, _ = lax.scan(step, carry, None, length=nrec)
        return carry, (rec_fn(*carry[:2], carry[2]) if rec_fn else None)
    carry, recs = lax.scan(block, (state[0], state[1], state[2]), None, length=nsteps//nrec)
    return carry, recs

def boris_passage_env(name, lam, sig, dtfac=1.0, NP=64, JSW=4, seed=11, ramp_wb=30.0, Tfac=2.1, nuf=0.0, ret_u=False):
    """As boris_passage but with smooth amplitude envelope: sin^2 ramp-up over t_r and ramp-down over t_r at the end; t_r = ramp_wb/omega_b in time.  Control = same integrator b=0."""
    kind, Delta, kw = REG[name]
    ks, a = spectrum(kind, Delta, **kw)
    ls, g, F, kin = line_data(ks); L = len(ks); ic = L//2
    G = np.array([k["G_per_F"] for k in kin])
    A0, _, _ = drift.sweep_rate(case, ls[ic]["kz"], g[ic], TH0, scale=lam); A = A0*Om**2
    Dd = abs(ls[1]["kz"]-ls[0]["kz"])*np.sqrt(g[ic]**2-1)*np.cos(TH0)/g[ic]*Om if L > 1 else 1e9
    b = b_unit(Delta)*sig**2
    wbj = np.sqrt(G*F*b*a*np.array([abs(k["ddelta"]) for k in kin]))*Om; wb = wbj[ic]
    half = (L-1)/2 if kind == "eq" else JSW
    X = (half+0.5)*Dd + max(10*np.sqrt(A), 6*wb)
    # ramp length in detuning units: must cover ramp_wb * wb of detuning change; start further out so the wave is fully on within X
    tr = ramp_wb/wb                               # s
    Xr = A*tr                                     # detuning swept during the ramp
    X = X + Xr; Xn = X/Om; T = Tfac*Xn/A0         # Tfac*X/A0 in 1/Omega
    dt = DT*dtfac; nst = int(T/dt)
    Tn = nst*dt; trn = tr*Om
    def env(t):
        up = jnp.sin(0.5*jnp.pi*jnp.clip(t/trn, 0, 1))**2; dn = jnp.sin(0.5*jnp.pi*jnp.clip((Tn-t)/trn, 0, 1))**2
        return up*dn
    r = np.random.default_rng(seed)
    theta = r.uniform(TH0-0.01, TH0+0.01, NP)
    gam = solve_gamma(case, ls[ic]["kz"], np.full(NP, -Xn), theta)
    p = np.sqrt(gam**2-1); phi = r.uniform(0, 2*np.pi, NP)
    zr = 2*np.pi/(Delta*pl.c/Om) if L > 1 else 2*np.pi/ls[0]["kz"]
    z0 = r.uniform(0, zr, NP)
    u = np.stack([p*np.sin(theta)*np.cos(phi), p*np.sin(theta)*np.sin(phi), -p*np.cos(theta)], 1)
    x = np.stack([p*np.sin(theta)*np.sin(phi), np.zeros(NP), z0], 1)
    lines, aa, _ = bw.make_lines(case, list(ks), list(b*a), r.uniform(0, 2*np.pi, L))
    E0z, fric = lam*case.EoverEc*case.Ec_n, lam*case.Ec_n
    def run(amps):
        fn = jax.jit(lambda x, u: simulate_env((x, u, 0.0), nst, dt, wn, lines, amps, env, E0z=E0z, fric=fric, eps_rr=lam))
        (xf, uf, _), _ = fn(jnp.asarray(x), jnp.asarray(u)); return np.asarray(uf)
    t0 = time.time(); uf = run(aa); u0f = run(0.0*aa)
    gw, g0 = quantities(uf)[3], quantities(u0f)[3]
    return dict(name=name, lam=lam, sigma=sig, b=b, wb=float(wb), R=float(wb**2/A), A=A, dt=dt, dtfac=dtfac, nst=nst, NP=NP, ramp_wb=ramp_wb, g_wave=gw, g_b0=g0, dg=gw-g0, time=time.time()-t0)
