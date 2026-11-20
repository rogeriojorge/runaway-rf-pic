"""Full-orbit (relativistic Boris, P2 integrator pieces) vs reduced model: wave work per particle with E-field, friction, LL (all x lam), same Gaussian comb.
Energy accounting in the full-orbit run: Ew = sum(-E_wave . v) dt (work of the wave field), Ed = sum of E0z work + friction + LL changes; check
gamma_final - gamma_0 = Ew + Ed (+ pitch-scattering, which conserves gamma, switched off here)."""
import os, sys, time, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from redpin import RedE, spectrum, case, Om, wn, bw, pl
from pinrun import p1_sample, make_model, Delta, SIGJ
from jaxincell._core import landau_lifshitz
C, E_CH, M_E = pl.c, pl.e, pl.me
def run_full(bpk, lam, Np, T_sim, seed=3, steps_per_gyro=32, nrec_blocks=20, drive=True):
    rng = np.random.default_rng(seed)
    gS, thS, fz = p1_sample()
    ks, a = spectrum("gauss", Delta, sig_j=SIGJ)
    lines, aa, ls = bw.make_lines(case, list(ks), list(bpk*a), rng.uniform(0, 2*np.pi, len(ks)))
    i = rng.integers(0, len(gS), Np); g0 = gS[i]; th0 = thS[i]; p = np.sqrt(g0**2-1)
    phi = rng.uniform(0, 2*np.pi, Np); z0 = rng.uniform(0, 2*np.pi/Delta, Np)*(C/Om)
    # positions in units c/Omega; z0 in 1/m -> c/Omega: z period 2 pi/Delta[1/m] * Omega/c
    z0 = rng.uniform(0, 2*np.pi/Delta, Np)*(Om/C)*0 + rng.uniform(0, 2*np.pi/(Delta*C/Om), Np)
    u = np.stack([p*np.sin(th0)*np.cos(phi), p*np.sin(th0)*np.sin(phi), -p*np.cos(th0)], 1)
    x = np.stack([p*np.sin(th0)/g0*np.sin(phi), np.zeros(Np), z0], 1)
    dt = 2*np.pi*16.4/steps_per_gyro               # same dt as P3 (gyro period at gamma 16.4)
    Ecn = case.Ec_n; E0 = case.EoverEc*Ecn
    nst = int(T_sim*Om/dt); nblk = nrec_blocks; nr = nst//nblk; nst = nr*nblk
    fields = bw.make_fields(lines, aa, wn, 0.0)
    def step(carry, _):
        x, u, t, Ew, Ed = carry
        Ew_, Bw_ = fields(x, t)
        E = Ew_.at[:, 2].add(lam*E0 if drive else 0.0); B = Bw_.at[:, 2].add(1.0)
        un = bw.boris_push(x, u, t, dt, E, B)
        gm = bw.lorentz_gamma(un); go = bw.lorentz_gamma(u)
        vbar = 0.5*(un/gm[:, None] + u/go[:, None])
        dEw = -jnp.sum(Ew_*vbar, 1)*dt                      # wave work (gamma units; E in B0 c, t in 1/Omega)
        g_after_boris = bw.lorentz_gamma(un)
        if drive:
            du_si, _p = landau_lifshitz(un*C, E*bw.B0_REF*C, B*bw.B0_REF, -E_CH, M_E)
            un2 = un + dt*lam*du_si/(C*bw.OM_REF)
            n2 = jnp.sum(un2*un2, -1); gh2 = 1.0+n2
            un3 = un2 - dt*lam*Ecn*(gh2/n2)[:, None]*un2/jnp.sqrt(n2)[:, None]
        else:
            un3 = un
        dEd_extra = bw.lorentz_gamma(un3) - g_after_boris
        # E0z work: dgamma = -(E0z z-hat).v dt
        dEd = -(lam*E0 if drive else 0.0)*vbar[:, 2]*dt + dEd_extra
        xn = x + dt*un3/bw.lorentz_gamma(un3)[:, None]
        return (xn, un3, t+dt, Ew+dEw, Ed+dEd), None
    def block(carry, _):
        carry, _ = lax.scan(step, carry, None, length=nr)
        return carry, jnp.stack([bw.lorentz_gamma(carry[1]), carry[3], carry[4]], 0)
    carry0 = (jnp.asarray(x), jnp.asarray(u), 0.0, jnp.zeros(Np), jnp.zeros(Np))
    t0 = time.time()
    carry, recs = jax.jit(lambda: lax.scan(block, carry0, None, length=nblk))()
    recs = np.asarray(recs)          # (nblk, 3, Np)
    g = recs[:, 0]; Ew = recs[:, 1]; Ed = recs[:, 2]
    acc = (g[-1]-g0) - (Ew[-1]+Ed[-1])
    return dict(bpk=bpk, lam=lam, Np=Np, T_sim=nst*dt/Om, dt_s=dt/Om, nst=nst, Ew_mean=float(Ew[-1].mean()), Ew_sem=float(Ew[-1].std()/np.sqrt(Np)), Ed_mean=float(Ed[-1].mean()),
                dg_mean=float((g[-1]-g0).mean()), acct_resid_rms=float(np.sqrt(np.mean(acc**2))), acct_resid_mean=float(acc.mean()), dg_rms=float(np.std(g[-1]-g0)), time=time.time()-t0,
                Ew_t=Ew.mean(1).tolist())
def run_reduced_same(bpk, lam, Np, T_sim, seed=3, dtf=0.05, nblk=20):
    rng = np.random.default_rng(seed)
    gS, thS, fz = p1_sample(); red, ks, a = make_model(bpk); L = len(ks)
    i = rng.integers(0, len(gS), Np); g0 = gS[i]; th0 = thS[i]; p = np.sqrt(g0**2-1)
    psi0 = rng.uniform(0, 2*np.pi, Np); z0 = rng.uniform(0, 2*np.pi/Delta, Np); phis = rng.uniform(0, 2*np.pi, L)
    from p3lib import line_data
    ls, g_, Fr, kin = line_data(ks); G = np.array([k["G_per_F"] for k in kin]); dd = np.array([abs(k["ddelta"]) for k in kin])
    wbmax = float(np.sqrt(G*Fr*bpk*a*dd).max()*Om); dt = dtf/wbmax
    nst = int(T_sim/dt); nr = nst//nblk; nst = nr*nblk
    ph = jnp.asarray(phis-phis[red.ic])
    (carry, rec) = jax.jit(lambda: red.run(g0, -p*np.cos(th0), psi0, z0, ph, dt, nst, nr, lam=lam, nuf=0.0, drive=True, key=seed))()
    rec = np.asarray(rec)
    return dict(Ew_mean=float(rec[-1, 2].mean()), Ew_sem=float(rec[-1, 2].std()/np.sqrt(Np)), Ed_mean=float(rec[-1, 3].mean()), dg_rms=float(np.std(rec[-1, 0]-g0)), nst=nst, T=nst*dt)
if __name__ == "__main__":
    bpk = float(sys.argv[1]); lam = float(sys.argv[2]); Np = int(sys.argv[3]); T = float(sys.argv[4]); tag = sys.argv[5]
    r = run_reduced_same(bpk, lam, Np, T); print("REDUCED", r, flush=True)
    f = run_full(bpk, lam, Np, T); print("FULL", {k: v for k, v in f.items() if k != "Ew_t"}, flush=True)
    json.dump(dict(reduced=r, full=f), open(f"../data/fullcheck_{tag}.json", "w"))
