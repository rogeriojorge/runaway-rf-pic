"""V1: integrator checks. (a) no wave: gamma and gyro-phase; (b) pure wave: exact first integral
I = gamma - (w/kz)(u_z - A_z); (c) agreement with the JAX-in-Cell Boris (small dt) and its gyro-phase error.
"""
import os, sys, json
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
import bw, plasma as pl
sys.path.insert(0, os.path.expanduser("~/local/jic-work/P2/JAX-in-Cell"))

case = bw.Case()
res = {}
g0, th0 = 16.0, 0.2
u0m = np.sqrt(g0**2-1)
def init(n, seed=0, g=g0, th=th0):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2*np.pi, n)
    u = np.stack([u0m*np.sin(th)*np.cos(ph), u0m*np.sin(th)*np.sin(ph), -u0m*np.cos(th)*np.ones(n)], 1)
    x = np.stack([rng.uniform(0, 50, n), np.zeros(n), rng.uniform(0, 50, n)], 1)
    return jnp.asarray(x), jnp.asarray(u), ph

# ---- (a) no wave, B0 only: exact rotation angle per step; gamma conserved
lines, amps, ls = bw.make_lines(case, [41.0], [0.0], [0.0])
for dtfrac in (8, 16, 32):
    dt = 2*np.pi*g0/dtfrac
    n = 8; x, u, ph = init(n)
    nst = 4000
    st, rec = bw.simulate((x, u, 0.0), nst, nst, dt, case.wn, lines, amps, rec_fn=bw.std_rec)
    uf = np.asarray(st[1]); gf = np.sqrt(1+np.sum(uf**2, 1))
    # exact: angle advanced = nst*dt/gamma ; u^{n-1/2}: phase of u at t=(nst-1/2)dt
    phi_exact = ph + (nst - 0.5)*dt/g0 + 0 # initial u taken at -1/2 step as given: treat as u^{-1/2}
    phi_exact = ph + nst*dt/g0
    phi_num = np.arctan2(uf[:, 1], uf[:, 0])
    dphi = np.angle(np.exp(1j*(phi_num-phi_exact))).mean()
    res[f"a_dt{dtfrac}"] = dict(gamma_drift=float(np.abs(gf/g0-1).max()), phase_err_rad=float(dphi),
                                  steps=nst, phase_per_step=float(dphi/nst))
    print("(a)", dtfrac, res[f"a_dt{dtfrac}"])

# ---- (b) pure wave: exact first integral
b = 3e-4
kz_list = [41.0]
lines, amps, ls = bw.make_lines(case, kz_list, [b], [0.7])
Ehat = ls[0]["E"]; kx, kz, w = ls[0]["kx"], ls[0]["kz"], case.wn
def invariant(x, u, t):
    arg = kx*x[:, 0] + kz*x[:, 2] - w*t + 0.7
    Az = b*np.real(Ehat[2]/(1j*w)*np.exp(1j*arg))
    g = np.sqrt(1+np.sum(u**2, 1))
    return g - (w/kz)*(u[:, 2] - Az)
gres = case.gamma_res(kz, -1, th0)
x, u, ph = init(16, g=gres)
out = {}
for dtfrac in (8, 16, 32):
    dt = 2*np.pi*gres/dtfrac
    nst = 200000
    nrec = nst//50
    st, rec = bw.simulate((x, u, 0.0), nst, nrec, dt, case.wn, lines, amps, rec_fn=lambda x_, u_, t_: jnp.concatenate([x_, u_], 1))
    rec = np.asarray(rec)
    ts = (np.arange(1, 51)*nrec)*dt
    Is = np.array([invariant(r[:, :3], r[:, 3:], t - 0.0*dt) for r, t in zip(rec, ts)])
    I0 = invariant(np.asarray(x), np.asarray(u), 0.0)
    drift = np.abs(Is - I0[None, :]).max()
    # excursion of u_z and gamma to show the invariant is non-trivial
    gz = np.sqrt(1+np.sum(rec[:, :, 3:]**2, 2))
    out[dtfrac] = dict(max_dI=float(drift), gamma_excursion=float(np.ptp(gz, axis=0).max()), t_end=float(ts[-1]))
    print("(b)", dtfrac, out[dtfrac])
res["b_invariant"] = out

# ---- (c) cross-check against JAX-in-Cell Boris (SI) on the same fields
try:
    from jaxincell._core import boris_relativistic
    c, me, e = pl.c, pl.me, pl.e
    B0 = case.B0; Om = case.Om
    x, u, ph = init(4, g=gres)
    dt_n = 2*np.pi*gres/64
    nst = 3000
    st, _ = bw.simulate((x, u, 0.0), nst, nst, dt_n, case.wn, lines, amps)
    un = np.asarray(st[1])
    # JIC push
    fields = bw.make_fields(lines, amps, case.wn)
    xj, uj = x, u*c
    for n in range(nst):
        t = n*dt_n
        E, Bw = fields(xj, t)
        Bn = Bw.at[:, 2].add(1.0)
        uj = boris_relativistic(uj, E*B0*c, Bn*B0, -e/me, dt_n/Om)
        xj = xj + dt_n*uj/c/jnp.sqrt(1+jnp.sum((uj/c)**2, 1))[:, None]
    uj = np.asarray(uj)/c
    rel = np.abs(uj-un).max()/np.abs(un).max()
    res["c_jic_vs_bw_dt_T/64"] = dict(max_rel_diff_u=float(rel), steps=nst,
                                      note="JIC boris_relativistic uses t=qB dt/(2 gamma m) (no tan): gyro-phase error ~ (omega_c dt)^2/12 per step")
    print("(c)", res["c_jic_vs_bw_dt_T/64"])
except Exception as ex:
    print("JIC check failed:", ex)
    res["c_error"] = str(ex)
json.dump(res, open("../data/val1_integrator.json", "w"), indent=1)
