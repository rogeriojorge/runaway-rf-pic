"""Direct check of the quasilinear absorbed power against test particles: reduced model, no drive, no collisions, P1-sampled electrons in the zone, Gaussian comb at
b_pk (sigma>1, stochastic).  Wave work Ew(t) per particle; slope compared with the QL energy-balance prediction (qlib) for the same distribution."""
import os, sys, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from pinrun import p1_sample, make_model, Delta
from p3lib import line_data
from redpin import Om
import pin_model as pm
def run(bpk, Np=4096, T=3e-6, nrec=60, dtf=0.05, seed=5):
    rng = np.random.default_rng(seed); gS, thS, fz = p1_sample(); red, ks, a = make_model(bpk); L = len(ks)
    i = rng.integers(0, len(gS), Np); g0 = gS[i]; th0 = thS[i]; p = np.sqrt(g0**2-1)
    psi0 = rng.uniform(0, 2*np.pi, Np); z0 = rng.uniform(0, 2*np.pi/Delta, Np); phis = rng.uniform(0, 2*np.pi, L)
    ls, g_, Fr, kin = line_data(ks); G = np.array([k["G_per_F"] for k in kin]); dd = np.array([abs(k["ddelta"]) for k in kin])
    wb = float(np.sqrt(G*Fr*bpk*a*dd).max()*Om); dt = dtf/wb; nst = int(T/dt); nr = nst//nrec; nst = nr*nrec
    ph = jnp.asarray(phis-phis[red.ic])
    carry, rec = jax.jit(lambda: red.run(g0, -p*np.cos(th0), psi0, z0, ph, dt, nst, nr, lam=0.0, nuf=0.0, drive=False, key=seed))()
    rec = np.asarray(rec); t = np.arange(1, nrec+1)*nr*dt; Ew = rec[:, 2].mean(1)
    return t, Ew, dt, wb
if __name__ == "__main__":
    out = {}
    for b in (1e-4, 3e-4):
        t, Ew, dt, wb = run(b)
        # slope between 25% and 100% of T
        m = t > 0.25*t[-1]; sl = np.polyfit(t[m], Ew[m], 1)[0]
        mq = pm.P_QL_zone(b)/pm.MC2
        print(f"b={b:.0e}: wb={wb:.2e}  wave work slope = {sl:.3e} mc2/s per zone particle (early: first quarter {np.polyfit(t[~m], Ew[~m],1)[0]:.3e});  QL x rho(sigma) prediction = {mq*pm.rho(b):.3e} (rho={pm.rho(b):.2f}), pure QL {mq:.3e}", flush=True)
        out[str(b)] = dict(t=t.tolist(), Ew=Ew.tolist(), slope=float(sl), QL=float(mq), rho=float(pm.rho(b)))
    json.dump(out, open("../data/ql_vs_particles.json", "w"))
