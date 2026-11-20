"""Step 3: stationary reduced N-line model (step1 red_st.py physics, kappa on) with independent random-walk phase noise per line.
Line phase phi_j(t) = phi_j0 + W_j(t), <dW^2> = 2 Gamma t (Lorentzian line of HWHM Gamma, field coherence exp(-Gamma |tau|)).
Every particle carries its own noise realisation (non-interacting particles, so the ensemble is also an ensemble over realisations).
Noise phase is advanced once per RK4 step and held over the step (error O(Gamma dt), checked by dtfac convergence)."""
import os, sys
sys.path.insert(0, os.path.expanduser("results/step1/scripts"))
from pcommon import *
import red_env
C = pl.c

def make_rhs(red, kappa=1.0):
    kz, r, bamps, dk = red.kz, red.r, red.bamps, red.dk_m
    ic = red.ic; kzc = kz[ic]; a_j = wn/kz
    def rhs(y, phis):
        g, uz, psi, z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        p2 = g*g-1.0; up = jnp.sqrt(jnp.maximum(p2-uz*uz, 1e-12)); thp = jnp.arctan2(up, -uz)
        Fc = red.Fc(g, thp)
        Gj = up[:, None]/(g[:, None]-uz[:, None]/a_j[None, :])
        cj = Gj*(Fc[:, None]*r[None, :])*bamps[None, :]*Om
        ph = psi[:, None] + dk[None, :]*z[:, None] + phis
        dg_ = jnp.sum(cj*jnp.cos(ph), 1)
        duz = jnp.sum((kz[None, :]/wn)*cj*jnp.cos(ph), 1)
        dpsi = -(wn - kzc*uz/g - 1.0/g)*Om
        kj = (Fc*Om/up)[:, None]*r[None, :]*bamps[None, :]
        dpsi = dpsi - kappa*jnp.sum(kj*jnp.sin(ph), 1)
        dz = C*uz/g
        return jnp.stack([dg_, duz, dpsi, dz], 1)
    return rhs

def setup(name, sigma, b_scale=1.0):
    kind, Delta, kw = REG[name]
    ks, a = spectrum(kind, Delta, **kw); L = len(ks); ic = L//2
    bu = b_unit(Delta); b = bu*sigma**2*b_scale
    nl = local_numbers(ks, a, b, TH0)
    ls, g, F, kin = line_data(ks)
    G = np.array([k["G_per_F"] for k in kin]); ddl = np.array([abs(k["ddelta"]) for k in kin])
    win_lines = L
    gsp = abs(np.mean(np.diff(g))); gc = g[ic] if L % 2 == 1 else 0.5*(g[L//2-1]+g[L//2])
    gwin = (gc-0.5*win_lines*gsp, gc+0.5*win_lines*gsp)
    Dd = float(np.mean(nl["Dd"])); wbj = np.sqrt(G*F*b*a*ddl)*Om; wb = float(wbj[ic])
    rate = np.pi*(G*F*b*a*Om)**2
    Dql = float(sum(rate)/(win_lines*Dd))
    Lw = win_lines*Dd/(ddl[ic]*Om)
    sig = float(np.mean(nl["sigma"][max(0, ic-2):ic+2]))
    return dict(ks=ks, a=a, L=L, ic=ic, b=b, g=g, ddl=ddl, rate=rate, Dd=Dd, wb=wb, Dql=Dql, Lw=float(Lw), sig=sig, gwin=gwin, gsp=gsp, win_lines=win_lines)

def run(name, sigma, Gphi_over_D, nbeats, dtfac=0.1, NP=128, seed=1, nrec=2000, kappa=1.0, log_rec=True, stop_m2=None):
    s = setup(name, sigma); L, ic, Dd, wb = s["L"], s["ic"], s["Dd"], s["wb"]
    Gam = Gphi_over_D*Dd
    Tbeat = 2*np.pi/Dd; T = nbeats*Tbeat
    fastest = ((L+s["win_lines"])/2*Dd + 4*wb)
    dt = dtfac/fastest
    # record times: log-spaced in steps (blocks of unequal length handled by python loop over blocks)
    nst = int(T/dt)
    bs = max(nst//nrec, 1); nblk = nst//bs; nst = nblk*bs; lens = np.full(nblk, bs)
    r = np.random.default_rng(seed)
    gwin = s["gwin"]
    theta = r.uniform(TH0-0.01, TH0+0.01, NP); gam = r.uniform(*gwin, NP); p = np.sqrt(gam**2-1)
    phi = r.uniform(0, 2*np.pi, NP); z0 = r.uniform(0, 2*np.pi/ (REG[name][1]) , NP)
    phis0 = r.uniform(0, 2*np.pi, L); phis0 = phis0-phis0[ic]
    red = red_env.Red(s["ks"], s["b"]*s["a"])
    rhs = make_rhs(red, kappa)
    sq = np.sqrt(2*Gam*dt)
    def step(c, _):
        y, W, key = c
        key, k2 = jax.random.split(key)
        W = W + sq*jax.random.normal(k2, W.shape)
        ph = W + jnp.asarray(phis0)[None, :]
        k1 = rhs(y, ph); k2_ = rhs(y+0.5*dt*k1, ph); k3 = rhs(y+0.5*dt*k2_, ph); k4 = rhs(y+dt*k3, ph)
        return (y+dt/6*(k1+2*k2_+2*k3+k4), W, key), None
    y = jnp.stack([jnp.asarray(gam), jnp.asarray(-p*np.cos(theta)), jnp.asarray(phi), jnp.asarray(z0)], 1)
    W = jnp.zeros((NP, L)); key = jax.random.PRNGKey(seed*7919+int(1000*Gphi_over_D))
    carry = (y, W, key); jitted = {}
    ts = []; m2 = []; mg = []; t0 = time.time(); nd = 0
    for ln in lens:
        ln = int(ln)
        if ln not in jitted: jitted[ln] = jax.jit(lambda c, n=ln: jax.lax.scan(step, c, None, length=n)[0])
        carry = jitted[ln](carry); nd += ln
        gg = np.asarray(carry[0][:, 0]); ts.append(nd*dt); m2.append(float(np.mean((gg-gam)**2))); mg.append(float(np.mean(gg-gam)))
        if stop_m2 is not None and m2[-1] >= stop_m2*s['Lw']**2: break
    return dict(name=name, sigma=s["sig"], sigma_nom=sigma, b=s["b"], wb=wb, Dd=Dd, Dql=s["Dql"], Lw=s["Lw"], Gphi_over_D=Gphi_over_D, nbeats=nbeats, dtfac=dtfac, NP=NP, seed=seed,
                nst=nst, dt=dt, ts=np.array(ts), beats=np.array(ts)/Tbeat, m2=np.array(m2), mean=np.array(mg), time=time.time()-t0, Tbeat=Tbeat, g_final=gg, g_init=gam, gwin=gwin)

# ---------- reference: quasilinear (Langevin) diffusion in the same finite band ----------
def ql_reference(Lw=None, N=400, nt=400, taumax=12.0):
    """m2_ref(tau)/Lw^2 for QL diffusion (variance slope Dql, i.e. x-variance = tau, tau = Dql t/Lw^2) of particles uniform on [0,1], absorbed (frozen) at the band edges.
    Deterministic: eigen-decomposition of the finite-difference generator (N interior nodes); uniform D (dense-comb QL limit).  Returns (tau, m)."""
    h = 1.0/(N+1); x = np.arange(1, N+1)*h
    A = (np.diag(-2*np.ones(N))+np.diag(np.ones(N-1), 1)+np.diag(np.ones(N-1), -1))/(2*h*h)
    lam, V = np.linalg.eigh(A)               # symmetric
    taus = np.geomspace(1e-4, taumax, nt); out = []
    for t in taus:
        E = np.exp(lam*t); P = (V*E)@V.T      # P[x0 (col index source), x] : P[i,j] = prob from node j to node i ... symmetric
        # P[x, x0]; interior part
        d2 = (x[:, None]-x[None, :])**2
        inter = np.sum(P*d2)/N
        # absorbed mass at each edge from source x0: a0(x0,t) = (1/(2h^2)) * int_0^t P[1st node, x0] ds = (1/(2h^2)) sum_k V[0,k] V[j,k] (e^{lam t}-1)/lam
        I = (V*((E-1)/lam))@V.T
        a0 = I[0, :]/(2*h*h); a1 = I[-1, :]/(2*h*h)
        out.append(inter + np.sum(a0*x**2)/N + np.sum(a1*(1-x)**2)/N)
    return taus, np.array(out)

def s_star(d, ref, frac=0.5):
    """Mixing-equivalent D_eff/D_QL.  m2_ref plateau m_sat; t_c = first time m2_sim >= frac*m_sat, else T.  s = tau_ref(m2_sim(t_c)) /(Dql t_c / Lw^2)."""
    tau, m = ref; msat = m[-1]
    m2 = d["m2"]/d["Lw"]**2; ts = d["ts"]
    idx = np.where(m2 >= frac*msat)[0]
    i = idx[0] if len(idx) else len(ts)-1
    tauq = np.interp(m2[i], m, tau)       # m monotone increasing
    return float(tauq/(d["Dql"]*ts[i]/d["Lw"]**2)), float(ts[i]/d["Tbeat"]), bool(len(idx) > 0)
