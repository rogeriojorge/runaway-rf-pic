"""Independent test-particle integrator for runaway electrons in B0 z-hat plus plane whistler waves.

Normalised units: time 1/Omega_ce, length c/Omega_ce, momentum u = p/(m c), B in B0, E in B0 c.
Electron: du/dt = -(E + v x B), v = u/gamma.  Relativistic Boris with the exact-rotation (tan) angle
so the gyro-phase of the uniform B0 is exact; positions leapfrogged.
Optional: static E_par, collisional friction, Langevin pitch-angle scattering (Perez/Takizuka-Abe type,
relativistic coefficient nu_D = (Z+1) gamma/(p^3 tau_c)), reduced Landau-Lifshitz radiation reaction.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import plasma as pl
import sys
sys.path.insert(0, os.path.expanduser("~/local/jic-work/P2/JAX-in-Cell"))
from jaxincell._core import landau_lifshitz      # reference reduced Landau-Lifshitz (validated against Smilei in this program)

# ----------------------------------------------------------------------------- DIII-D case
class Case:
    """DIII-D numbers (Choudhury PRL/PPCF): 476 MHz, k_par = 41 1/m, n_e = 0.8e19 (reproduces k_perp = 76)."""
    def __init__(self, B0=1.4, ne=0.8e19, f=476e6, kpar=41.0, Te_eV=2165., Zeff=1.5, lnL=17.0, EoverEc=8.5):
        self.B0, self.ne, self.f, self.kpar0 = B0, ne, f, kpar
        self.Om = pl.e*B0/pl.me
        self.w = 2*np.pi*f
        self.sp = pl.species(B0, ne)
        self.wn = self.w/self.Om                 # omega / Omega
        self.Zeff, self.lnL = Zeff, lnL
        # collisional time tau_c = 4 pi eps0^2 m^2 c^3/(n e^4 lnL) and critical field
        self.tau_c = 4*np.pi*pl.eps0**2*pl.me**2*pl.c**3/(ne*pl.e**4*lnL)
        self.Ec = ne*pl.e**3*lnL/(4*np.pi*pl.eps0**2*pl.me*pl.c**2)      # V/m
        self.tau_rad = 6*np.pi*pl.eps0*pl.me**3*pl.c**3/(pl.e**4*B0**2)  # s
        self.EoverEc = EoverEc
        # normalised drive coefficients
        self.Ec_n = self.Ec/(pl.c*B0)            # E_c in B0 c units
        self.eps_rr = 1.0/(self.tau_rad*self.Om)
        self.nu0 = 1.0/(self.tau_c*self.Om)      # 1/tau_c in units of Omega

    def line(self, kpar):
        """k_perp, E-hat (unit B) for a line at kpar (1/m): returns normalised wavevector, fields per unit b."""
        kp, _ = pl.solve_kperp(self.w, kpar, self.sp)
        E, B, _ = pl.polarization(self.w, kp, kpar, self.sp)
        s = 1.0/np.linalg.norm(B)               # |B_vec|=1 (units B0 per unit b)
        return dict(kperp=kp, kpar=kpar, kx=kp*pl.c/self.Om, kz=kpar*pl.c/self.Om,
                    E=E*s, B=B*s)               # E in B0 c, B in B0 per unit b

    def gamma_res(self, kz, uz_sign=-1, theta=0.1, n=1):
        """gamma satisfying gamma*w - kz*u_par = n (units Omega), u_par = uz_sign*gamma*beta*cos(theta)."""
        from scipy.optimize import brentq
        f = lambda g: g*self.wn - kz*uz_sign*np.sqrt(g*g-1)*np.cos(theta) - n
        return brentq(f, 1.5, 200.)

# ----------------------------------------------------------------------------- integrator
def make_fields(lines, amps, omega, ramp=0.0):
    """Return a function (x, t) -> (E, B) summing the wave lines; lines is dict of jnp arrays (L,...)."""
    kx, kz, ph = lines["kx"], lines["kz"], lines["phi"]
    Er, Ei, Br, Bi = lines["Er"], lines["Ei"], lines["Br"], lines["Bi"]      # (L,3)
    def fields(x, t):
        arg = x[:, 0:1]*kx[None, :] + x[:, 2:3]*kz[None, :] - omega*t + ph[None, :]    # (N,L)
        a_t = amps if ramp == 0.0 else amps*jnp.minimum(t/ramp, 1.0)          # linear turn-on of the amplitude over `ramp`
        co, si = jnp.cos(arg)*a_t[None, :], jnp.sin(arg)*a_t[None, :]
        E = co @ Er - si @ Ei
        B = co @ Br - si @ Bi
        return E, B
    return fields

def make_lines(case, kpars, amps, phases):
    ls = [case.line(k) for k in kpars]
    d = dict(kx=np.array([l["kx"] for l in ls]), kz=np.array([l["kz"] for l in ls]),
             Er=np.array([l["E"].real for l in ls]), Ei=np.array([l["E"].imag for l in ls]),
             Br=np.array([l["B"].real for l in ls]), Bi=np.array([l["B"].imag for l in ls]),
             phi=np.asarray(phases, float))
    return {k: jnp.asarray(v) for k, v in d.items()}, jnp.asarray(amps, float), ls

C_LIGHT, E_CHARGE, M_E = pl.c, pl.e, pl.me
B0_REF, OM_REF = 1.4, pl.e*1.4/pl.me      # RR is evaluated for the DIII-D B0; eps_rr acts only as an on/off switch (1.0) or scale

def lorentz_gamma(u):
    return jnp.sqrt(1.0 + jnp.sum(u*u, axis=-1))

def boris_push(x, u, t, dt, Efield, Bfield, a=-1.0):
    """One relativistic Boris step (x^n,u^{n-1/2}) -> (x^{n+1},u^{n+1/2}); tan-exact rotation."""
    E, B = Efield, Bfield
    um = u + 0.5*a*dt*E
    gm = lorentz_gamma(um)
    Bn = jnp.sqrt(jnp.sum(B*B, axis=-1)) + 1e-300
    tv = (a*B/Bn[:, None])*jnp.tan(jnp.abs(a)*Bn*dt/(2*gm))[:, None]
    s = 2*tv/(1.0 + jnp.sum(tv*tv, axis=-1, keepdims=True))
    upr = um + jnp.cross(um, tv)
    up = um + jnp.cross(upr, s)
    un = up + 0.5*a*dt*E
    return un

def simulate(state, nsteps, nrec, dt, omega, lines, amps, E0z=0.0, fric=0.0, eps_rr=0.0, nu_scale=0.0,
             key=None, rec_fn=None, ramp=0.0):
    """Advance for nsteps (multiple of nrec); returns final state and records (every nrec steps).

    state = (x, u, t).  Forces: static E0z along +z (accelerates electrons toward -z), friction
    fric*(gamma^2/u^2) u_hat, reduced LL radiation reaction (eps_rr), pitch Langevin (nu_scale = nu_D*u^3/gamma in 1/Omega).
    """
    fields = make_fields(lines, amps, omega, ramp)
    if key is None:
        key = jax.random.PRNGKey(0)

    def step(carry, _):
        x, u, t, key = carry
        E, B = fields(x, t)
        E = E.at[:, 2].add(E0z)
        B = B.at[:, 2].add(1.0)
        un = boris_push(x, u, t, dt, E, B)
        gh = lorentz_gamma(un)
        v = un/gh[:, None]
        # radiation reaction: field-quadratic Landau-Lifshitz from JAX-in-Cell (SI call, normalised back); explicit, tiny
        if eps_rr != 0.0:
            du_si, _p = landau_lifshitz(un*C_LIGHT, E*B0_REF*C_LIGHT, B*B0_REF, -E_CHARGE, M_E)
            un = un + dt*eps_rr*du_si/(C_LIGHT*OM_REF)
        if fric != 0.0:
            un2 = jnp.sum(un*un, axis=-1)
            gh2 = 1.0 + un2
            un = un - dt*fric*(gh2/un2)[:, None]*un/jnp.sqrt(un2)[:, None]
        if nu_scale != 0.0:
            key, k1 = jax.random.split(key)
            gm = lorentz_gamma(un); p = jnp.sqrt(jnp.sum(un*un, axis=-1))
            nuD = nu_scale*gm/p**3
            uh = un/p[:, None]
            # orthonormal tangent basis
            ref = jnp.where(jnp.abs(uh[:, 2:3]) < 0.9, jnp.array([[0., 0., 1.]]), jnp.array([[1., 0., 0.]]))
            e1 = jnp.cross(uh, ref); e1 = e1/jnp.linalg.norm(e1, axis=-1, keepdims=True)
            e2 = jnp.cross(uh, e1)
            g = jax.random.normal(k1, (x.shape[0], 2))
            sig = jnp.sqrt(nuD*dt)[:, None]
            d = uh + sig*(g[:, 0:1]*e1 + g[:, 1:2]*e2)
            un = p[:, None]*d/jnp.linalg.norm(d, axis=-1, keepdims=True)
        xn = x + dt*un/lorentz_gamma(un)[:, None]
        return (xn, un, t + dt, key), None

    def block(carry, _):
        carry, _ = lax.scan(step, carry, None, length=nrec)
        x, u, t, key = carry
        r = rec_fn(x, u, t) if rec_fn is not None else None
        return carry, r

    carry, recs = lax.scan(block, (state[0], state[1], state[2], key), None, length=nsteps//nrec)
    return (carry[0], carry[1], carry[2]), recs

def std_rec(x, u, t):
    g = lorentz_gamma(u)
    s = u[:, 0]**2 + u[:, 1]**2
    return jnp.stack([g, u[:, 2], s, jnp.arctan2(jnp.sqrt(s), -u[:, 2]), jnp.arctan2(u[:, 1], u[:, 0]), x[:, 2]], axis=-1)
