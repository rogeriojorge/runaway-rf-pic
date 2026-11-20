"""Deterministic drift of a resonant runaway (E_par, collisional friction, reduced LL radiation) and the resulting detuning sweep."""
import numpy as np, os, sys
sys.path.insert(0, os.path.expanduser("~/local/jic-work/P2/JAX-in-Cell"))
import plasma as pl

def du_dt(case, u, EoverEc=None, fric_on=True, rr_on=True, E_on=True, scale=1.0):
    """du/dt (units Omega) from E_par, friction and reduced LL, for u = (ux,uy,uz) with B0 z-hat only."""
    EoverEc = case.EoverEc if EoverEc is None else EoverEc
    g = np.sqrt(1+np.dot(u, u)); v = u/g; p = np.sqrt(np.dot(u, u))
    d = np.zeros(3)
    if E_on:
        d += np.array([0, 0, -EoverEc*case.Ec_n])           # electron: force = -E  (E along +z)
    if fric_on:
        d += -case.Ec_n*(g*g/(p*p))*u/p
    if rr_on:
        import jax.numpy as jnp
        from jaxincell._core import landau_lifshitz
        Ez = (EoverEc*case.Ec_n if E_on else 0.0)
        du, _ = landau_lifshitz(jnp.asarray([u*pl.c]), jnp.asarray([[0, 0, Ez*case.B0*pl.c]]), jnp.asarray([[0, 0, case.B0]]), -pl.e, pl.me)
        d += np.asarray(du)[0]/(pl.c*case.Om)
    return scale*d

def sweep_rate(case, kzn, gam, theta, **kw):
    """d delta/dt (Omega^2 units, >0 if resonance is approached from below) and dgamma/dt, dtheta/dt."""
    p = np.sqrt(gam**2-1)
    u = np.array([p*np.sin(theta), 0.0, -p*np.cos(theta)])
    d = du_dt(case, u, **kw)
    dn = np.linalg.norm(d); d = d/dn; h = 1e-6     # step along the drift direction, rate rescaled below
    def dl(uv):
        g = np.sqrt(1+np.dot(uv, uv))
        return case.wn - kzn*uv[2]/g - 1.0/g
    dd = (dl(u+h*d)-dl(u-h*d))/(2*h)*dn    # d delta/dt
    g = np.sqrt(1+np.dot(u, u))
    dgdt = np.dot(u, d)/g*dn
    s = np.hypot(u[0], u[1]); up = u[2]
    th = lambda uv: np.arctan2(np.hypot(uv[0], uv[1]), -uv[2])
    dthdt = (th(u+h*d)-th(u-h*d))/(2*h)*dn
    return dd, dgdt, dthdt

if __name__ == "__main__":
    import bw, theory as th_
    case = bw.Case(); kzn = case.line(41.0)["kz"]
    print("Ec = %.4f V/m, E = %.4f V/m, tau_c = %.2f s, tau_rad = %.2f s, nu0 = %.3e /s" % (case.Ec, case.EoverEc*case.Ec, case.tau_c, case.tau_rad, 1/case.tau_c))
    for th in (0.05, 0.1, 0.2, 0.3):
        g = case.gamma_res(kzn, -1, th)
        for lab, kw in (("E only", dict(fric_on=False, rr_on=False)), ("E+fric", dict(rr_on=False)), ("E+fric+RR", dict()), ("fric+RR (E=0)", dict(E_on=False))):
            dd, dg, dth = sweep_rate(case, kzn, g, th, **kw)
            print(f"theta={th:4.2f} gamma_res={g:6.3f}  {lab:14s}: d(delta)/dt={dd*case.Om**2:+.3e} rad/s^2  dgamma/dt={dg*case.Om:+.3e}/s  dtheta/dt={dth*case.Om:+.3e}/s")
        line = case.line(41.0)
        wb = th_.omega_b(line, case.wn, g, th, 1.0)*case.Om
        print(f"            omega_b/sqrt(b) = {wb:.3e} /s  => omega_b^2/b = {wb**2:.3e} s^-2;  F_rad = {th_.frad(line, case.wn, g, th)[0]:.3f}")

def drift_vec(case, u, scale=1.0, EoverEc=None, E_on=True, fric_on=True, rr_on=True):
    """Vectorised du/dt (units Omega^2 ... i.e. per unit of 1/Omega time) for u of shape (N,3)."""
    import jax.numpy as jnp
    from jaxincell._core import landau_lifshitz
    EoverEc = case.EoverEc if EoverEc is None else EoverEc
    u = np.asarray(u); p2 = np.sum(u*u, 1); g2 = 1+p2; p = np.sqrt(p2)
    d = np.zeros_like(u)
    Ez = EoverEc*case.Ec_n if E_on else 0.0
    d[:, 2] += -Ez
    if fric_on:
        d += -case.Ec_n*(g2/p2)[:, None]*u/p[:, None]
    if rr_on:
        N = len(u)
        E = np.zeros((N, 3)); E[:, 2] = Ez*case.B0*pl.c
        B = np.zeros((N, 3)); B[:, 2] = case.B0
        du, _ = landau_lifshitz(jnp.asarray(u*pl.c), jnp.asarray(E), jnp.asarray(B), -pl.e, pl.me)
        d += np.asarray(du)/(pl.c*case.Om)
    return scale*d

def propagate_control(case, u0, T, scale=1.0, nsteps=400, **kw):
    """RK4 integration of the unperturbed drift (no wave) for time T (1/Omega units). Returns final u."""
    u = np.array(u0, float); h = T/nsteps
    for _ in range(nsteps):
        k1 = drift_vec(case, u, scale, **kw)
        k2 = drift_vec(case, u+0.5*h*k1, scale, **kw)
        k3 = drift_vec(case, u+0.5*h*k2, scale, **kw)
        k4 = drift_vec(case, u+h*k3, scale, **kw)
        u = u + h/6*(k1+2*k2+2*k3+k4)
    return u
