"""Trace outliers: record (gamma,u_z) of wave and b=0 Boris runs; report detuning history of bulk vs outlier particles."""
from pc2 import *
name = sys.argv[1]; lam = float(sys.argv[2]); sig = float(sys.argv[3]); NP = int(sys.argv[4]); ramp = float(sys.argv[5]); nrec = 200
S = setup(name, lam, sig, NP=NP, ramp_wb=ramp)
dt = DT; nst = int(S["Tn"]/dt); nst = (nst//nrec)*nrec; Tn = nst*dt; trn = S["tr"]*Om
def env(t):
    if trn == 0.0: return 1.0
    return jnp.sin(0.5*jnp.pi*jnp.clip(t/trn, 0, 1))**2*jnp.sin(0.5*jnp.pi*jnp.clip((Tn-t)/trn, 0, 1))**2
th, p, phi = S["theta"], S["p"], S["phi"]
u = np.stack([p*np.sin(th)*np.cos(phi), p*np.sin(th)*np.sin(phi), -p*np.cos(th)], 1); x = np.stack([p*np.sin(th)*np.sin(phi), np.zeros(NP), S["z0n"]], 1)
lines, aa, _ = bw.make_lines(case, list(S["ks"]), list(S["b"]*S["a"]), S["phis"])
lam_ = S["lam"]; E0z, fric = lam_*case.EoverEc*case.Ec_n, lam_*case.Ec_n
def rec(x, u, t): return jnp.stack([bw.lorentz_gamma(u), u[:, 2]], -1)
def run(am):
    fn = jax.jit(lambda x, u: simulate_env((x, u, 0.0), nst, dt, wn, lines, am, env, E0z=E0z, fric=fric, eps_rr=lam_, nrec=nst//nrec, rec_fn=rec))
    return np.asarray(fn(jnp.asarray(x), jnp.asarray(u))[1])
rw, r0 = run(aa), run(0*aa); t = np.arange(1, nrec+1)*(nst//nrec)*dt/Om
kz = S["ls"][S["ic"]]["kz"]
dlw = (case.wn - kz*rw[..., 1]/rw[..., 0] - 1/rw[..., 0])*Om; dl0 = (case.wn - kz*r0[..., 1]/r0[..., 0] - 1/r0[..., 0])*Om
dg = rw[..., 0]-r0[..., 0]
np.savez(f"../data/trace2_{name}_lam{lam:g}_s{sig}_r{ramp:g}.npz", t=t, dg=dg, dlw=dlw, dl0=dl0, wb=S["wb"], A=S["A"])
out = dg[-1] > np.median(dg[-1]) + 0.01
print("outliers", out.sum(), "of", NP, "wb", S["wb"], "A", S["A"])
idx = np.r_[np.where(~out)[0][:2], np.where(out)[0][:4]]
for k in idx:
    print("particle", k, "outlier" if out[k] else "bulk", " detuning/wb (wave) at t-fraction .25 .5 .6 .7 .8 .9 1:", np.round(dlw[[int(f*nrec)-1 for f in (.25, .5, .6, .7, .8, .9, 1.)], k]/S["wb"], 2))
    print("      control detuning/wb :", np.round(dl0[[int(f*nrec)-1 for f in (.25, .5, .6, .7, .8, .9, 1.)], k]/S["wb"], 2), " dg:", np.round(dg[[int(f*nrec)-1 for f in (.25, .5, .6, .7, .8, .9, 1.)], k], 4))
