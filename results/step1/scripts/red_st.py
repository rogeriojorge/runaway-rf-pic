"""Reduced stationary N-line model with a high-order RK4 integrator (for long windows).  State (gamma, u_z, psi_c, z); no drive, collisions or noise.
Same right-hand side as P3 red.py (gyro-averaged n=1 force, c_j = G_j F_j b_j Om, du_z = (k_j/omega) dgamma, psi_j = psi_c + dk_j z + phi_j),
but integrated by classical RK4 at a step set by the fastest line phase rate.  Window/IC/QL conventions as in p3lib.run_S."""
from pcommon import *
import red_env
C = pl.c

def make_rhs(red, phis, kappa=0.0):
    kz, r, bamps, dk = red.kz, red.r, red.bamps, red.dk_m
    ic = red.ic; kzc = kz[ic]; a_j = wn/kz
    def rhs(y):
        g, uz, psi, z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        p2 = g*g-1.0; up = jnp.sqrt(jnp.maximum(p2-uz*uz, 1e-12)); thp = jnp.arctan2(up, -uz)
        Fc = red.Fc(g, thp)
        Gj = up[:, None]/(g[:, None]-uz[:, None]/a_j[None, :])
        cj = Gj*(Fc[:, None]*r[None, :])*bamps[None, :]*Om
        ph = psi[:, None] + dk[None, :]*z[:, None] + phis[None, :]
        dg_ = jnp.sum(cj*jnp.cos(ph), 1)
        duz = jnp.sum((kz[None, :]/wn)*cj*jnp.cos(ph), 1)
        dpsi = -(wn - kzc*uz/g - 1.0/g)*Om
        if kappa != 0.0:
            kj = (Fc*Om/up)[:, None]*r[None, :]*bamps[None, :]
            dpsi = dpsi - kappa*jnp.sum(kj*jnp.sin(ph), 1)
        dz = C*uz/g
        return jnp.stack([dg_, duz, dpsi, dz], 1)
    return rhs

def run_st(name, sigma, nbeats, dtfac=0.1, NP=256, seed=1, nrec=64, win_lines=None, TAILS=False, b_scale=1.0, kappa=0.0):
    kind, Delta, kw = REG[name]
    ks, a = spectrum(kind, Delta, **kw); L = len(ks); ic = L//2
    bu = b_unit(Delta); b = bu*sigma**2*b_scale
    nl = local_numbers(ks, a, b, TH0)
    ls, g, F, kin = line_data(ks)
    G = np.array([k["G_per_F"] for k in kin]); ddl = np.array([abs(k["ddelta"]) for k in kin])
    if win_lines is None: win_lines = 5 if kind == "gauss" else L
    gsp = abs(np.mean(np.diff(g)))
    gc = g[ic] if L % 2 == 1 else 0.5*(g[L//2-1]+g[L//2])
    gwin = (gc-0.5*win_lines*gsp, gc+0.5*win_lines*gsp)
    inside = [j for j in range(L) if gwin[0] <= g[j] <= gwin[1]]
    Dd = float(np.mean(nl["Dd"])); wbj = np.sqrt(G*F*b*a*ddl)*Om; wb = float(wbj[ic])
    rate = np.pi*(G*F*b*a*Om)**2
    Dql = sum(rate[j] for j in inside)/(win_lines*Dd)
    Lw = win_lines*Dd/(ddl[ic]*Om)
    sig = float(np.mean(nl["sigma"][max(0, ic-2):ic+2]))
    Tbeat = 2*np.pi/Dd; T = nbeats*Tbeat      # nbeats counts beat PERIODS 2 pi/Dd
    fastest = ((L+win_lines)/2*Dd + 4*wb)
    dt = dtfac/fastest; nst = int(T/dt); nr = max(nst//nrec, 1); nst = nr*nrec
    r = np.random.default_rng(seed)
    theta = r.uniform(TH0-0.01, TH0+0.01, NP); gam = r.uniform(*gwin, NP); p = np.sqrt(gam**2-1)
    phi = r.uniform(0, 2*np.pi, NP); z0 = r.uniform(0, 2*np.pi/Delta, NP); phis = r.uniform(0, 2*np.pi, L)
    red = red_env.Red(ks, b*a)
    rhs = make_rhs(red, jnp.asarray(phis-phis[ic]), kappa)
    def rk4(y, _):
        k1 = rhs(y); k2 = rhs(y+0.5*dt*k1); k3 = rhs(y+0.5*dt*k2); k4 = rhs(y+dt*k3)
        return y+dt/6*(k1+2*k2+2*k3+k4), None
    def block(y, _):
        y, _ = jax.lax.scan(rk4, y, None, length=nr); return y, y[:, 0]
    y0 = jnp.stack([jnp.asarray(gam), jnp.asarray(-p*np.cos(theta)), jnp.asarray(phi), jnp.asarray(z0)], 1)
    t0 = time.time()
    yf, recs = jax.jit(lambda y: jax.lax.scan(block, y, None, length=nrec))(y0)
    recs = np.asarray(recs); dg = recs-gam[None, :]
    ts = np.arange(1, nrec+1)*nr*dt; m2 = np.mean(dg**2, 1)
    return dict(kappa=kappa, name=name, sigma=sig, b=b, wb=wb, Dd=Dd, Dql=float(Dql), Lw=float(Lw), nbeats=nbeats, dtfac=dtfac, NP=NP, seed=seed, nst=nst, dt=dt, ts=ts, m2=m2, beats=ts/Tbeat,
                time=time.time()-t0, mean_g=np.mean(recs, 1), R_fill=float(m2[-1]/(Lw**2/6)), Tbeat=Tbeat, g_final=recs[-1])

def slope_ratio(d, f0, f1):
    """D_eff/D_QL from the slope of <dg^2> over the fraction [f0,f1] of the window."""
    ts, m2 = d["ts"], d["m2"]; i0, i1 = int(f0*len(ts)), int(f1*len(ts))
    return float(np.polyfit(ts[i0:i1], m2[i0:i1], 1)[0]/d["Dql"])
