"""P1: test-particle tail against a frozen Maxwellian bulk + ions, production JAX-in-Cell kernels.

Tail particles evolve with: E drive (parallel to B), production Landau-Lifshitz (reduced,
B-only, as in examples/3_advanced/runaway_baseline.py) and the production relativistic binary
collision kernel (jaxincell._collisions.collide, Perez rate + Nanbu angle) against frozen
electron and ion marker sets. The background rows of the kernel output are discarded each step,
so each tail particle sees a Maxwell-Juttner bulk of fixed density: exactly the test-particle
(linearized) operator of a Maxwellian bulk, the field-particle part being O(n_tail/n).
Modes: full | drag (scattering switched off: the post-collision momentum is rotated back to its
pre-collision direction, magnitude kept) | noion (electron-electron collisions only).
usage: tail_steady.py out.npz  key=value ...   (nt nb dt T start mode seed snap B E lnL nlnl)
"""
import os, sys, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import jax, jax.numpy as jnp
assert jax.config.jax_enable_x64
from functools import partial
from jaxincell import elementary_charge as e, epsilon_0, mass_electron as me, speed_of_light as c
from jaxincell._collisions import collide
from jaxincell._core import landau_lifshitz

kw = dict(a.split("=") for a in sys.argv[2:])
nt = int(kw.get("nt", 8192)); nb = int(kw.get("nb", nt)); dt = float(kw.get("dt", 1e-3))
T = float(kw.get("T", 120.)); start = kw.get("start", "fp"); mode = kw.get("mode", "full")
seed = int(kw.get("seed", 0)); snap = float(kw.get("snap", 2.))
B0 = float(kw.get("B", 2.)); E_acc = float(kw.get("E", 5e-3))
nlnl = float(kw.get("nlnl", 1.5e19)); ne = float(kw.get("ne", 1e18)); lnL = nlnl / ne
Te = float(kw.get("Te", 100.)); Z = 1
wspread = float(kw.get("wspread", 0.))   # relative spread of tail weights (weighted seed test)
mi = 3.3435837768e-27
tau_c = 4*np.pi*epsilon_0**2*me**2*c**3/(ne*e**4*lnL)
tau_rad = e**2/(6*np.pi*epsilon_0*me*c**3)
a_B = tau_rad*(e*B0/me)**2; a_E = E_acc*e/(me*c)

def thermal(count, temp_ev, mass, sd):
    theta = temp_ev*e/(mass*c**2)
    r = np.linspace(0., max(20*np.sqrt(theta), 40*theta), 20001)
    pdf = r**2*np.exp(-r**2/((np.sqrt(1+r**2)+1)*theta))
    cdf = np.r_[0., np.cumsum(pdf[1:]+pdf[:-1])]; cdf /= cdf[-1]
    rng = np.random.default_rng(sd)
    d = rng.normal(size=(count//2, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    m = np.interp(rng.permutation((np.arange(count//2)+.5)/(count//2)), cdf, r)
    h = c*m[:, None]*d
    return np.concatenate((h, -h))

def start_state(n):
    g0, th0 = {"fp": (13.134, 0.1361), "g40": (40., 0.1), "g5": (5., 0.3)}[start]
    rng = np.random.default_rng(seed+100)
    ph = rng.uniform(0, 2*np.pi, n)
    p = c*np.sqrt(g0**2-1)
    return np.stack((p*np.cos(th0)*np.ones(n), p*np.sin(th0)*np.cos(ph), p*np.sin(th0)*np.sin(ph)), 1)

ub = jnp.asarray(np.concatenate((thermal(nb, Te, me, 21+seed), thermal(nb, Te, mi, 22+seed))))
wb = ne/nb
rng = np.random.default_rng(seed+7)
wt = (ne*1e-8/nt)*(np.exp(wspread*rng.normal(size=nt)) if wspread > 0 else np.ones(nt))
weight = jnp.asarray(np.concatenate((wt, np.full(nb, wb), np.full(nb, wb))))
mass = jnp.asarray(np.concatenate((np.full(nt, me), np.full(nb, me), np.full(nb, mi))))
charge = jnp.asarray(np.concatenate((np.full(nt, -e), np.full(nb, -e), np.full(nb, Z*e))))
blocks = ((0, nt), (nt, nb), (nt+nb, nb))
pairs = ((0, 1),) if mode == "noion" else ((0, 1), (0, 2))
Ex = jnp.array([1., 0., 0.]); Bv = B0*Ex; zero3 = jnp.zeros(3)
tm = jnp.full(nt, me); tq = jnp.full(nt, -e)
acc = (-e/me)*(-E_acc)*Ex   # electron charge -e, field -E_acc along x (force along +x), as the baseline

def drift(u, h):
    f1, _ = landau_lifshitz(u, zero3, Bv, tq, tm)
    mid = u + .5*h*(acc + f1)
    f2, _ = landau_lifshitz(mid, zero3, Bv, tq, tm)
    return u + h*(acc + f2)

@partial(jax.jit, static_argnames=("steps",))
def run(ut, key, steps):
    def step(carry, _):
        ut, key = carry
        ut = drift(ut, .5*dt)
        key, sub = jax.random.split(key)
        full = jnp.concatenate((ut, ub))
        out = collide(sub, jnp.zeros_like(full), full, weight, mass, charge, blocks, pairs, lnL, dt, 1., 1., 1,
                      relativistic=True)[:nt]
        if mode == "drag":
            n_old = jnp.linalg.norm(ut, axis=1, keepdims=True)
            n_new = jnp.linalg.norm(out, axis=1, keepdims=True)
            out = ut/n_old*n_new
        return (drift(out, .5*dt), key), None
    (ut, key), _ = jax.lax.scan(step, (ut, key), None, length=steps)
    return ut, key

ut = jnp.asarray(start_state(nt)); key = jax.random.PRNGKey(seed)
per = int(round(snap/dt)); nsn = int(round(T/snap))
ts, snaps = [0.], [np.asarray(ut)]
t0 = time.time()
for k in range(nsn):
    ut, key = run(ut, key, per); ut.block_until_ready()
    ts.append((k+1)*per*dt); snaps.append(np.asarray(ut))
    if k % 10 == 0:
        s = snaps[-1]; g = np.sqrt(1+(s**2).sum(1)/c**2)
        print(f"t={ts[-1]:.1f} <g>={g.mean():.3f} wall={time.time()-t0:.0f}s", flush=True)
S = np.array(snaps)/c
np.savez_compressed(sys.argv[1], t=np.array(ts), eta=S.astype(np.float32), wt=wt/wt.sum(),
                    params=np.array([nt, nb, dt, T, a_B, a_E, tau_c, lnL]))
