"""Reduced Langevin model of the tail: the L2 test-particle ODE plus pitch noise of strength kappa
(kappa=1 is the Ito-consistent spherical diffusion whose mean drift is the ODE's -(Z+1) xi gamma/(tau_c eta^3)
term; kappa=0 is exactly the L2 ODE). Used only to isolate the cause of the bump; the production-kernel
runs are the measurement.  usage: sde_scan.py out.npz kappa nt dt T [Z] [lnL_scale]"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
from jaxincell import elementary_charge as e, epsilon_0, mass_electron as me, speed_of_light as c
out, kappa, nt, dt, T = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
Z = float(sys.argv[6]) if len(sys.argv) > 6 else 1.
B0, E_acc, ne, lnL = 2., 5e-3, 1e18, 15.
tau_c = 4*np.pi*epsilon_0**2*me**2*c**3/(ne*e**4*lnL)
a_B = e**2/(6*np.pi*epsilon_0*me*c**3)*(e*B0/me)**2; a_E = E_acc*e/(me*c)
g0, th0 = 13.134, 0.1361
def step(carry, _):
    eta, xi, key = carry
    key, k = jax.random.split(key)
    g = jnp.sqrt(1+eta**2)
    deta = a_E*xi - g**2/(tau_c*eta**2) - a_B*g*eta*(1-xi**2)
    nu = (Z+1)*g/(tau_c*eta**3)
    dxi = a_E*(1-xi**2)/eta - nu*xi + a_B*xi*(1-xi**2)/g
    # Heun-free Euler-Maruyama with tiny dt; noise on the sphere
    xi = xi + dxi*dt + jnp.sqrt(kappa*nu*(1-xi**2)*dt)*jax.random.normal(k, xi.shape)
    xi = jnp.clip(xi, -1+1e-12, 1-1e-12)
    eta = jnp.maximum(eta + deta*dt, 1e-3)
    return (eta, xi, key), None
from functools import partial
@partial(jax.jit, static_argnums=3)
def run(eta, xi, key, n):
    (eta, xi, key), _ = jax.lax.scan(step, (eta, xi, key), None, length=n)
    return eta, xi, key
eta = jnp.full(nt, np.sqrt(g0**2-1)); xi = jnp.full(nt, np.cos(th0)); key = jax.random.PRNGKey(1)
snap = 2.; per = int(round(snap/dt)); ts = [0.]; S = [np.stack((np.asarray(eta)*np.asarray(xi), np.asarray(eta)*np.sqrt(1-np.asarray(xi)**2), np.zeros(nt)), 1)]
rng = np.random.default_rng(0)
for k in range(int(round(T/snap))):
    eta, xi, key = run(eta, xi, key, per)
    a, x = np.asarray(eta), np.asarray(xi)
    S.append(np.stack((a*x, a*np.sqrt(1-x**2), np.zeros(nt)), 1)); ts.append((k+1)*snap)
np.savez_compressed(out, t=np.array(ts), eta=np.array(S).astype(np.float32), wt=np.full(nt, 1./nt))
