"""Cross-check of Boris with an independent RK4 integrator of the exact Lorentz-force ODE for a PARALLEL circularly polarised wave (z-dependence only), same ICs.
usage: para_rk4.py sign NP sigma dtfac_rk (RK4 step = DT*dtfac_rk)"""
from pc2 import *
import plasma
sign = float(sys.argv[1]); NP = int(sys.argv[2]); sg = float(sys.argv[3]); dtr = float(sys.argv[4])
def line_par(kpar):
    kz = kpar*pl.c/case.Om; s2 = np.sqrt(0.5)
    return dict(kperp=0.0, kpar=kpar, kx=0.0, kz=kz, E=np.array([1, 1j*sign, 0])*s2*(case.wn/kz), B=-1j*sign*np.array([1, 1j*sign, 0])*s2)
case.line = line_par
S = setup("eqN1", 1000, sg, NP=NP, ramp_wb=0.0, ic="start", rr=0)
lam = S["lam"]; E0z = lam*case.EoverEc*case.Ec_n; fricc = lam*case.Ec_n
l = line_par(S["ks"][0]); kz = l["kz"]; b = S["b"]; wn_ = case.wn; ph0 = S["phis"][0]
Er, Ei, Br, Bi = l["E"].real, l["E"].imag, l["B"].real, l["B"].imag
def acc(u, z, t, b):
    g = jnp.sqrt(1+jnp.sum(u*u, -1)); v = u/g[:, None]
    arg = kz*z - wn_*t + ph0
    # amplitude envelope omitted here (checked separately): abrupt switch-on
    c, s = jnp.cos(arg)[:, None]*b, jnp.sin(arg)[:, None]*b
    E = c*jnp.asarray(Er)[None]-s*jnp.asarray(Ei)[None]; B = c*jnp.asarray(Br)[None]-s*jnp.asarray(Bi)[None]
    B = B.at[:, 2].add(1.0); E = E.at[:, 2].add(E0z)
    du = -(E + jnp.cross(v, B))
    p2 = jnp.sum(u*u, -1); du = du - (fricc*(g*g/p2)/jnp.sqrt(p2))[:, None]*u
    return du, v[:, 2]
def rhs(y, t, b):
    u, z = y[:, :3], y[:, 3]; du, dz = acc(u, z, t, b); return jnp.concatenate([du, dz[:, None]], 1)
dt = DT*dtr; nst = int(S["Tn"]/dt)
th, p, phi = S["theta"], S["p"], S["phi"]
u0 = np.stack([p*np.sin(th)*np.cos(phi), p*np.sin(th)*np.sin(phi), -p*np.cos(th)], 1)
y0 = jnp.asarray(np.concatenate([u0, S["z0n"][:, None]], 1))
def step(c, _):
    y, t, b = c
    k1 = rhs(y, t, b); k2 = rhs(y+0.5*dt*k1, t+0.5*dt, b); k3 = rhs(y+0.5*dt*k2, t+0.5*dt, b); k4 = rhs(y+dt*k3, t+dt, b)
    return (y+dt/6*(k1+2*k2+2*k3+k4), t+dt, b), None
def run(bb):
    (yf, _, _), _ = jax.lax.scan(step, (y0, 0.0, jnp.asarray(bb)), None, length=nst); return np.asarray(yf)
t0 = time.time(); yw = run(S["b"]); y0f = run(0.0)
gw = np.sqrt(1+np.sum(yw[:, :3]**2, 1)); g0 = np.sqrt(1+np.sum(y0f[:, :3]**2, 1)); dg = gw-g0
print("RK4 dt=%g*DT steps=%d  time %.0fs" % (dtr, nst, time.time()-t0)); print(np.round(dg, 4))
bd = boris_dg(S, 1.0); print("Boris(abrupt, dt=DT):"); print(np.round(bd["dg"], 4))
