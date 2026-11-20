"""Multi-line parallel-wave check: Boris vs independent RK4 (Cartesian exact eq) particle by particle.  usage: para_rk4N.py NAME NP sigma dtfac_rk"""
from pc2 import *
sign = 1.0; NAME = sys.argv[1]; NP = int(sys.argv[2]); sg = float(sys.argv[3]); dtr = float(sys.argv[4])
def line_par(kpar):
    kz = kpar*pl.c/case.Om; s2 = np.sqrt(0.5)
    return dict(kperp=0.0, kpar=kpar, kx=0.0, kz=kz, E=np.array([1, 1j*sign, 0])*s2*(case.wn/kz), B=-1j*sign*np.array([1, 1j*sign, 0])*s2)
case.line = line_par
S = setup(NAME, 1000, sg, NP=NP, ramp_wb=0.0, ic="start", rr=0)
lam = S["lam"]; E0z = lam*case.EoverEc*case.Ec_n; fricc = lam*case.Ec_n
L = S["L"]; lines, aa, _ = bw.make_lines(case, list(S["ks"]), list(S["b"]*S["a"]), S["phis"])
wn_ = case.wn
def acc(u, z, t, amps):
    g = jnp.sqrt(1+jnp.sum(u*u, -1)); v = u/g[:, None]
    arg = z[:, None]*lines["kz"][None, :] - wn_*t + lines["phi"][None, :]
    c, s = jnp.cos(arg)*amps[None, :], jnp.sin(arg)*amps[None, :]
    E = c@lines["Er"] - s@lines["Ei"]; B = c@lines["Br"] - s@lines["Bi"]
    B = B.at[:, 2].add(1.0); E = E.at[:, 2].add(E0z)
    du = -(E + jnp.cross(v, B)); p2 = jnp.sum(u*u, -1); du = du - (fricc*(g*g/p2)/jnp.sqrt(p2))[:, None]*u
    return du, v[:, 2]
def rhs(y, t, amps):
    du, dz = acc(y[:, :3], y[:, 3], t, amps); return jnp.concatenate([du, dz[:, None]], 1)
dt = DT*dtr; nst = int(S["Tn"]/dt)
th, p, phi = S["theta"], S["p"], S["phi"]
u0 = np.stack([p*np.sin(th)*np.cos(phi), p*np.sin(th)*np.sin(phi), -p*np.cos(th)], 1); y0 = jnp.asarray(np.concatenate([u0, S["z0n"][:, None]], 1))
def step(c, _):
    y, t, am = c
    k1 = rhs(y, t, am); k2 = rhs(y+0.5*dt*k1, t+0.5*dt, am); k3 = rhs(y+0.5*dt*k2, t+0.5*dt, am); k4 = rhs(y+dt*k3, t+dt, am)
    return (y+dt/6*(k1+2*k2+2*k3+k4), t+dt, am), None
def run(am):
    (yf, _, _), _ = jax.lax.scan(step, (y0, 0.0, am), None, length=nst); return np.asarray(yf)
yw = run(aa); y0f = run(0.0*aa)
dg = np.sqrt(1+np.sum(yw[:, :3]**2, 1)) - np.sqrt(1+np.sum(y0f[:, :3]**2, 1))
bd = boris_dg(S, 1.0)["dg"]
np.set_printoptions(precision=4, suppress=True, linewidth=160)
print("RK4:", dg); print("Boris:", bd)
q = lambda x: np.round(np.percentile(x, [5, 25, 50, 75, 95]), 4)
print("RK4 q", q(dg), "Boris q", q(bd))
