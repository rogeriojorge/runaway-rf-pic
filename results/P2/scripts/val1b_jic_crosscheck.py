"""V1c: same fields, same particles, same physical duration; JAX-in-Cell boris_relativistic versus bw.boris_push (tan-exact).
The two agree as dt -> 0 (difference ~ dt^2, the gyro-phase error of the linearised rotation)."""
import os, sys, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
import bw, plasma as pl
from jaxincell._core import boris_relativistic
case = bw.Case(); c, me, e = pl.c, pl.me, pl.e
b = 3e-4
lines, amps, ls = bw.make_lines(case, [41.0], [b], [0.7])
fields = bw.make_fields(lines, amps, case.wn)
gres = case.gamma_res(ls[0]["kz"], -1, 0.2); p = np.sqrt(gres**2-1)
rng = np.random.default_rng(0); ph = rng.uniform(0, 2*np.pi, 4)
u0 = jnp.asarray(np.stack([p*np.sin(.2)*np.cos(ph), p*np.sin(.2)*np.sin(ph), -p*np.cos(.2)*np.ones(4)], 1))
x0 = jnp.asarray(np.stack([p*np.sin(.2)*np.sin(ph), np.zeros(4), rng.uniform(0, 50, 4)], 1))
Tend = 3000*2*np.pi*gres/64          # fixed duration (1/Omega)
res = {}
ref = None
out = {}
for fr in (32, 64, 128, 256):
    dt = 2*np.pi*gres/fr; n = int(round(Tend/dt))
    st, _ = bw.simulate((x0, u0, 0.0), n, n, dt, case.wn, lines, amps)
    ub = np.asarray(st[1]); out[fr] = ub
    xj, uj = x0, u0*c
    @jax.jit
    def step(carry, n_):
        xj, uj = carry
        t = n_*dt
        E, Bw = fields(xj, t)
        Bn = Bw.at[:, 2].add(1.0)
        uj = boris_relativistic(uj, E*case.B0*c, Bn*case.B0, -e/me, dt/case.Om)
        xj = xj + dt*uj/c/jnp.sqrt(1+jnp.sum((uj/c)**2, 1))[:, None]
        return (xj, uj), None
    (xj, uj), _ = jax.lax.scan(step, (xj, uj), jnp.arange(n))
    out[("jic", fr)] = np.asarray(uj)/c
ref = out[256]
for fr in (32, 64, 128):
    res[f"bw_T/{fr}_vs_bw_T/256"] = float(np.abs(out[fr]-ref).max()/np.abs(ref).max())
    res[f"jic_T/{fr}_vs_bw_T/256"] = float(np.abs(out[("jic", fr)]-ref).max()/np.abs(ref).max())
res["jic_T/256_vs_bw_T/256"] = float(np.abs(out[("jic", 256)]-ref).max()/np.abs(ref).max())
for k, v in res.items(): print(k, "%.3e" % v)
json.dump(res, open("../data/val1b_jic.json", "w"), indent=1)
