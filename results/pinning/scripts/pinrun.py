"""Driven, replenished steady state of resonant runaways in the Gaussian helicon comb (reduced model with energy accounting).
Protocol: Np test electrons start from the P1 kinetic distribution restricted to the comb zone; E-field, collisional friction and LL synchrotron
(time-compressed by lam), Langevin pitch scattering (also x lam), the Gaussian comb (N_eff ~ 11, spacing 0.57/m).  Particles leaving the zone (gamma or
pitch) are re-injected from the unperturbed (P1) zone distribution (E-field/collisional refill).  Reported: wave work rate per particle P = <d(gamma)_wave>/dt
(m c^2 / s of simulated time) in the stationary window.  lam compresses ONLY the slow drives; see NOTES for the conversion to physical time."""
import os, sys, time, json, pickle
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from redpin import RedE, F_table, spectrum, line_data, local_numbers, case, Om, wn, TH0, th, pl
from runS_scan import b_unit
Delta = 0.57; SIGJ = 4.5
GZ = (12.5, 22.5); TMAX = 0.75         # zone limits for recycling
def p1_sample():
    d = np.load(os.path.expanduser("results/P1/data/full_fp.npz")); t = d["t"]; sel = t >= 60
    U = d["eta"][sel].reshape(-1, 3).astype(float); p = np.linalg.norm(U, axis=1); g = np.sqrt(1+p**2); thv = np.arccos(np.clip(U[:, 0]/p, -1, 1))
    z = (g > GZ[0]) & (g < GZ[1]) & (thv < TMAX)
    return g[z], thv[z], float(z.mean())
def make_model(bpk):
    ks, a = spectrum("gauss", Delta, sig_j=SIGJ)
    red = RedE(ks, bpk*a)
    gg, tt, F = F_table(red.ls[red.ic], ng=72, nt=60, grange=(11.5, 23.5), trange=(0.0, 0.8))
    red.gg, red.tt, red.F = jnp.asarray(gg), jnp.asarray(tt), jnp.asarray(F)
    return red, ks, a
def run_pin(bpk, lam, Np=512, T_rel=3.0, nblk_total=60, dtf=0.05, seed=1, nuf=None, verbose=True, drive=True, collisions=True):
    rng = np.random.default_rng(seed)
    gS, thS, fz = p1_sample()
    red, ks, a = make_model(bpk); L = len(ks)
    ls, g_, Fr, kin = line_data(ks)
    G = np.array([k["G_per_F"] for k in kin]); dd = np.array([abs(k["ddelta"]) for k in kin])
    wb = (np.sqrt(G*Fr*bpk*a*dd)*Om)                # per-line bounce freq (1/s)
    wbmax = max(float(wb.max()), 1e6)
    dt = dtf/wbmax
    aB = float(__import__("red").aB)
    # slow time scale: RR damping of p_perp, 1/(aB gamma lam)  (gamma ~17)
    t_rr = 1.0/(aB*17.0*lam)
    T = T_rel*t_rr*2.0                               # total simulated time
    nst_blk = max(int(T/nblk_total/dt), 20); nst_blk = (nst_blk//20)*20 or 20
    nuf = lam if nuf is None else nuf
    if not collisions: nuf = 0.0
    def draw(n):
        i = rng.integers(0, len(gS), n); g = gS[i]; thv = thS[i]; p = np.sqrt(g**2-1)
        return g, -p*np.cos(thv), thv
    g0, uz0, th0 = draw(Np)
    psi0 = rng.uniform(0, 2*np.pi, Np); z0 = rng.uniform(0, 2*np.pi/Delta, Np); phis = rng.uniform(0, 2*np.pi, L)
    ph = jnp.asarray(phis-phis[red.ic])
    key0 = jax.random.PRNGKey(seed)
    fn = jax.jit(lambda g, uz, psi, z, key: red.run(g, uz, psi, z, ph, dt, nst_blk, nst_blk, lam=lam, nuf=nuf, drive=drive, key=key))
    g, uz, psi, z = [jnp.asarray(x) for x in (g0, uz0, psi0, z0)]
    key = key0
    rec = []; t0 = time.time(); Ew_prev = np.zeros(Np); Ed_prev = np.zeros(Np); n_recycled = 0
    for ib in range(nblk_total):
        carry, r = fn(g, uz, psi, z, key)
        g, uz, psi, z, key, phn, Ew, Ed = carry
        gn, uzn, Ewn, Edn = [np.asarray(x) for x in (g, uz, Ew, Ed)]
        # block increments
        dEw = Ewn.copy(); dEd = Edn.copy()          # (accumulators reset every call since carry0 zeros Ew,Ed)
        p = np.sqrt(np.maximum(gn**2-1, 1e-12)); thv = np.arccos(np.clip(-uzn/p, -1, 1))
        out = (gn < GZ[0]) | (gn > GZ[1]) | (thv > TMAX)
        gp = np.asarray(g); uzp = np.asarray(uz)
        nz = (~out) & (gn > 14.0) & (gn < 20.5) & (thv < 0.6)
        rec.append(dict(t=(ib+1)*nst_blk*dt, Ew=float(dEw.mean()), Ed=float(dEd.mean()), n_out=int(out.sum()), th2=float(np.mean(np.sin(thv)**2*p**2)), gmean=float(gn.mean()), frac_core=float(nz.mean())))
        if out.any():
            m = int(out.sum()); gg, uu, tt_ = draw(m); gp = gp.copy(); uzp = uzp.copy(); gp[out] = gg; uzp[out] = uu
            g = jnp.asarray(gp); uz = jnp.asarray(uzp); n_recycled += m
    T_sim = nst_blk*dt*nblk_total
    # stationary window: second half
    h = nblk_total//2
    Pw = np.mean([r["Ew"] for r in rec[h:]])/(nst_blk*dt)
    Pd = np.mean([r["Ed"] for r in rec[h:]])/(nst_blk*dt)
    P1_ = np.mean([r["Ew"] for r in rec[:h]])/(nst_blk*dt)
    out = dict(bpk=bpk, lam=lam, Np=Np, dt=dt, T_sim=T_sim, nsteps=nst_blk*nblk_total, wbmax=wbmax, P_wave=Pw, P_wave_first_half=P1_, P_drive=Pd,
               recycled_per_particle_per_sim_s=n_recycled/Np/T_sim, frac_zone_P1=fz, rec=rec, time=time.time()-t0, nuf=nuf)
    if verbose: print(f"b={bpk:.2e} lam={lam:g}: P_wave={Pw:.3e} (first half {P1_:.3e}) P_drive={Pd:.3e} mc2/s per particle; steps={nst_blk*nblk_total:.2e} recycled/part/s={out['recycled_per_particle_per_sim_s']:.3g} t={out['time']:.0f}s", flush=True)
    return out
if __name__ == "__main__":
    lam = float(sys.argv[1]); Np = int(sys.argv[2]); T_rel = float(sys.argv[3]); dtf = float(sys.argv[4]); tag = sys.argv[5]; bs = [float(x) for x in sys.argv[6:]]
    for b in bs:
        o = run_pin(b, lam, Np=Np, T_rel=T_rel, dtf=dtf)
        json.dump(o, open(f"../data/pin_{tag}_b{b:.0e}_lam{lam:g}.json", "w"))
