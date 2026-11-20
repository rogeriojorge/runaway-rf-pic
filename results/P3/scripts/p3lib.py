"""P3 helpers: spectra, QL rates, overlap parameter, stationary/sweep Boris runs on the P2 integrator (imported unchanged)."""
import sys, os
sys.path.insert(0, os.path.expanduser("results/P2/scripts"))
os.environ.setdefault("JAX_ENABLE_X64", "1")
from exp import *      # P2: bw, th, pl, delta_of, quantities, rec_u ...
case = bw.Case(); wn = case.wn; Om = case.Om
TH0 = 0.2
K0 = 41.0           # central core k_par (1/m)
DT = 2*np.pi*16.4/32

def spectrum(kind, Delta, N=None, sig_j=None):
    """lines (k_par in 1/m) and relative amplitudes. 'eq' = N equal lines; 'gauss' = Gaussian power spectrum, sigma in line index."""
    if kind == "eq":
        j = np.arange(N) - (N-1)/2.0
        return K0 + Delta*j, np.ones(N)
    jm = int(np.ceil(2.5*sig_j)); j = np.arange(-jm, jm+1).astype(float)
    return K0 + Delta*j, np.exp(-j**2/(4*sig_j**2))       # amplitude = sqrt(power)

def line_data(ks, th0=TH0):
    ls = [case.line(k) for k in ks]
    g = np.array([case.gamma_res(l["kz"], -1, th0) for l in ls])
    F = np.array([th.frad(l, wn, gj, th0)[0] for l, gj in zip(ls, g)])
    kin = [th.kinematics(wn, l["kz"], gj, th0) for l, gj in zip(ls, g)]
    return ls, g, F, kin

def local_numbers(ks, amps, b, th0=TH0):
    """omega_b per line (1/s), detuning spacing (1/s), QL rate pi (G F b)^2/Delta_delta in gamma^2/s, overlap sigma between neighbours."""
    ls, g, F, kin = line_data(ks, th0)
    G = np.array([k["G_per_F"] for k in kin]); dd = np.array([abs(k["ddelta"]) for k in kin])
    wb = np.sqrt(G*F*b*amps*dd)*Om                              # 1/s
    Ddel = np.abs(np.diff(ks))*(case.wn*0+1)                     # placeholder
    # detuning separation of resonances at the resonance of line i+1: |delta_i - delta_{i+1}| = dk_n |u_par|/gamma
    dkn = np.diff(np.array([l["kz"] for l in ls]))
    upar = np.sqrt(g**2-1)*np.cos(th0)
    Dd = np.abs(dkn)*upar[:-1]/g[:-1]*Om                         # 1/s
    sig = 2*(wb[:-1]+wb[1:])/Dd if len(ks) > 1 else np.array([0.0])
    # QL rate in gamma^2 per second, smoothed over line spacing
    gam_rate = np.pi*(G*F*b*amps*Om)**2                          # (dgamma/dt amplitude)^2 * pi, units gamma^2/s^2 ; divide by detuning spacing (1/s)
    Dql = gam_rate[:-1]/Dd if len(ks) > 1 else np.array([0.0])
    return dict(g=g, wb=wb, Dd=Dd, sigma=sig, Dql=Dql, ls=ls, F=F, G=G)

if __name__ == "__main__":
    for Delta in (0.57,):
        ks, a = spectrum("eq", Delta, 5)
        for b in (1e-5, 1e-4, 1e-3):
            d = local_numbers(ks, a, b)
            print(f"Delta={Delta} b={b:g} wb={d['wb'][2]:.3e}/s Dd={d['Dd'][0]:.3e}/s sigma={d['sigma'][0]:.3f} Dql={d['Dql'][0]:.3e} g^2/s  gres={d['g'][0]:.3f}..{d['g'][-1]:.3f}  ")

# ----------------------------------------------------------------------------- stationary (no-drive) protocol
def init_ensemble(ks, gwin, Np, seed, th0=TH0, dth=0.01, Delta=None):
    r = np.random.default_rng(seed)
    theta = r.uniform(th0-dth, th0+dth, Np)
    g = r.uniform(gwin[0], gwin[1], Np)
    p = np.sqrt(g**2-1); phi = r.uniform(0, 2*np.pi, Np)
    zr = 2*np.pi/(Delta*pl.c/Om) if Delta else 2*np.pi/(0.1*pl.c/Om)
    z0 = r.uniform(0, zr, Np)
    u = np.stack([p*np.sin(theta)*np.cos(phi), p*np.sin(theta)*np.sin(phi), -p*np.cos(theta)], 1)
    x = np.stack([p*np.sin(theta)*np.sin(phi), np.zeros(Np), z0], 1)
    return x, u

def rec_gt(x, u, t):
    g = bw.lorentz_gamma(u)
    return jnp.stack([g, jnp.arctan2(jnp.sqrt(u[:, 0]**2+u[:, 1]**2), -u[:, 2])], -1)

def run_S(ks, amps, b, Delta, win_lines=None, Np=192, seed=1, nuf=0.0, nrec=96, Tfac_cap=300., Tmin_beats=10., fill_frac=0.25, Tfix=None, th0=TH0, rr=False):
    """Stationary ensemble: gamma uniform in a window of `win_lines` line-spacings centred on the central line resonance, theta0=0.2.
    Returns dict with <dgamma^2>(t) and QL reference.  nuf = multiplier of the DIII-D relativistic pitch-scattering rate nu_D=(Z+1)gamma/(p^3 tau_c)."""
    ks = np.asarray(ks); L = len(ks); amps = np.asarray(amps)
    nl = local_numbers(ks, amps, b, th0) if L > 1 else None
    ls, g, F, kin = line_data(ks, th0)
    G = np.array([k["G_per_F"] for k in kin]); dd = abs(kin[0]["ddelta"])
    ic = L//2
    # window: win_lines spacings around the central line (equal lines: all N)
    if win_lines is None: win_lines = L
    gsp = abs(np.mean(np.diff(g))) if L > 1 else 0.2
    gc = g[ic] if L % 2 == 1 else 0.5*(g[L//2-1]+g[L//2])
    gwin = (gc-0.5*win_lines*gsp, gc+0.5*win_lines*gsp)
    # lines resonant inside window and the QL rate averaged over the window
    inside = [j for j in range(L) if gwin[0] <= g[j] <= gwin[1]]
    Dd = np.mean(nl["Dd"]) if L > 1 else 1.0
    wbj = np.sqrt(G*F*b*amps*abs(np.array([k["ddelta"] for k in kin])))*Om
    wb_c = wbj[ic]
    rate = np.pi*(G*F*b*amps*Om)**2                             # gamma^2/s^2 per line
    Wd = win_lines*Dd                                            # window width in detuning (1/s)
    Dql = sum(rate[j] for j in inside)/Wd
    ddel_I = abs(kin[ic]['ddelta'])*Om
    Lw = Wd/ddel_I            # band extent along the single-line invariant curve (gamma units) = N Delta_delta / |d delta/d gamma|_I
    t_fill = fill_frac*(Lw**2/6)/Dql
    t_cap = Tfac_cap/wb_c
    T = Tfix if Tfix else max(min(t_fill, t_cap), Tmin_beats/Dd)
    nst = int(T/(DT/Om)); nr = max(nst//nrec, 1); nst = nr*nrec
    r = np.random.default_rng(seed+100)
    lines, aa, _ = bw.make_lines(case, list(ks), list(b*amps), r.uniform(0, 2*np.pi, L))
    x, u = init_ensemble(ks, gwin, Np, seed, th0, Delta=Delta)
    nu_scale = (case.Zeff+1)*case.nu0*nuf
    t0 = time.time()
    fn = jax.jit(lambda x, u: bw.simulate((jnp.asarray(x), jnp.asarray(u), 0.0), nst, nr, DT, wn, lines, aa, rec_fn=rec_gt, nu_scale=nu_scale, eps_rr=(1.0 if rr else 0.0), key=jax.random.PRNGKey(seed+7)))
    st, rec = fn(x, u)
    rec = np.asarray(rec)                                        # (nrec, Np, 2)
    g0 = np.sqrt(1+np.sum(u**2, 1)); 
    dg = rec[:, :, 0]-g0[None, :]
    ts = np.arange(1, nrec+1)*nr*DT/Om
    m2 = np.mean(dg**2, 1)
    th_rms = np.std(rec[-1, :, 1])
    # late slope over the second half
    h = nrec//2
    sl = np.polyfit(ts[h:], m2[h:], 1)[0]
    sl_a = np.polyfit(ts[h//2:h], m2[h//2:h], 1)[0]
    # bootstrap-free error via halves of the ensemble
    def slope(idx):
        mm = np.mean(dg[:, idx]**2, 1); return np.polyfit(ts[h:], mm[h:], 1)[0]
    s1, s2 = slope(np.arange(Np//2)), slope(np.arange(Np//2, Np))
    return dict(b=b, L=L, Delta=Delta, T=T, nst=nst, wb=float(wb_c), Dd=float(Dd), sigma=float(np.mean(nl["sigma"][max(0, ic-2):ic+2])) if L > 1 else 0.0,
                Dql=float(Dql), t_fill=float(t_fill), t_cap=float(t_cap), ts=ts, m2=m2, slope=float(sl), ratio=float(sl/Dql), ratio_err=float(abs(s1-s2)/2/Dql),
                S_end=float(m2[-1]/(Lw**2/6)), Lw=float(Lw), th_rms=float(th_rms), nuf=nuf, time=time.time()-t0, beats=float(T*Dd), bounces=float(T*wb_c/(2*np.pi)))

if __name__ == "__main__":
    import json, sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        for b in (3e-5, 1e-4, 3e-4):
            ks, a = spectrum("eq", 0.57, 5)
            r = run_S(ks, a, b, 0.57)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("ts", "m2")}, flush=True)
