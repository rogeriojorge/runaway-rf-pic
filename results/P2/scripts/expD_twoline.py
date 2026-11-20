"""Exp D: two coherent lines, spacing dk (1/m): overlap parameter sigma = 2(w_b1+w_b2)/Delta, Delta = separation of the two resonances in detuning.
Ensemble: gamma uniform between the two resonances at theta=0.2 (+-0.01).  Collisionless, no E.
S = <(gamma_t-gamma_0)^2>/(L^2/6) with L the gamma distance of the two resonances; global chaos fills the region (S -> 1)."""
import sys, pickle
from exp import *
case = bw.Case(); wn = case.wn
N = int(os.environ.get("NPART", 256)); dt = 2*np.pi*16.4/32
def one(b, dk, tfac=300., seed=5, rec_n=60):
    lns = [case.line(41.0), case.line(41.0+dk)]
    th0 = 0.2
    g1 = case.gamma_res(lns[0]["kz"], -1, th0); g2 = case.gamma_res(lns[1]["kz"], -1, th0)
    wb = [th.omega_b(l, wn, g, th0, b) for l, g in zip(lns, (g1, g2))]            # Omega units
    d_sep = abs(delta_of(case, lns[0]["kz"], g2, th0) - delta_of(case, lns[1]["kz"], g2, th0))   # separation at gamma_res2
    sigma = 2*(wb[0]+wb[1])/d_sep
    r = np.random.default_rng(seed)
    theta = r.uniform(th0-0.01, th0+0.01, N)
    g = r.uniform(min(g1, g2), max(g1, g2), N)
    p = np.sqrt(g**2-1); phi = r.uniform(0, 2*np.pi, N)
    zr = 2*np.pi/(abs(lns[1]["kz"]-lns[0]["kz"]))
    z0 = r.uniform(0, zr, N)
    u = np.stack([p*np.sin(theta)*np.cos(phi), p*np.sin(theta)*np.sin(phi), -p*np.cos(theta)], 1)
    x = np.stack([p*np.sin(theta)*np.sin(phi), np.zeros(N), z0], 1)
    lines, amps, _ = bw.make_lines(case, [41.0, 41.0+dk], [b, b], [0.0, r.uniform(0, 2*np.pi)])
    tmax = tfac/wb[0]
    nst = int(tmax/dt); nrec = max(nst//rec_n, 1); nst = nrec*rec_n
    st, rec = jax.jit(lambda x, u: bw.simulate((jnp.asarray(x), jnp.asarray(u), 0.0), nst, nrec, dt, wn, lines, amps, rec_fn=rec_u))(x, u)
    rec = np.asarray(rec)
    _, xi_t, th_t, g_t = quantities(rec)
    dg = g_t - g[None, :]
    L = abs(g2-g1)
    S = np.mean(dg**2, 1)/(L**2/6)
    # single-line island size in gamma for reference: halfwidth = 2 w_b/|d delta/d gamma|
    ddel = abs(th.kinematics(wn, lns[0]["kz"], g1, th0)["ddelta"])
    return dict(b=b, dk=dk, sigma=sigma, wb=[w*case.Om for w in wb], d_sep=d_sep*case.Om, L=L, island_hw=2*wb[0]/ddel,
                ts=np.arange(1, rec_n+1)*nrec*dt/case.Om, S=S, S_end=float(np.mean(S[-10:])), nst=nst)

if __name__ == "__main__":
    out = {"scan_dk": [], "scan_b": []}
    for dk in np.geomspace(0.05, 4.0, 14):
        t0 = time.time(); r = one(1e-4, dk); out["scan_dk"].append(r)
        print(f"b=1e-4 dk={dk:.3f} sigma={r['sigma']:.3f} L/island_hw={r['L']/r['island_hw']:.2f} S_end={r['S_end']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    for b in np.geomspace(1e-6, 1e-3, 13):
        t0 = time.time(); r = one(b, 0.44); out["scan_b"].append(r)
        print(f"dk=0.44 b={b:.2e} sigma={r['sigma']:.3f} S_end={r['S_end']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(out, open(os.path.join(OUT, "expD.pkl"), "wb"))
