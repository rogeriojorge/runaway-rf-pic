"""Shared helpers for the P2 experiments."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import json, time
import numpy as np, jax, jax.numpy as jnp
import bw, plasma as pl, theory as th

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def delta_of(case, kzn, gam, theta):
    upar = -np.sqrt(gam**2-1)*np.cos(theta)
    return case.wn - kzn*upar/gam - 1.0/gam

def solve_gamma(case, kzn, delta0, theta, g0=16.4):
    """gamma(s) giving detuning delta0 (Omega units) at pitch theta; Newton, vectorised."""
    g = np.full_like(np.asarray(theta, float), g0) + 0.0*np.asarray(delta0)
    for _ in range(60):
        f = delta_of(case, kzn, g, theta) - delta0
        h = 1e-6
        d = (delta_of(case, kzn, g+h, theta) - delta_of(case, kzn, g-h, theta))/(2*h)
        g = g - f/d
    return g

def frad_vec(line, wn, gam, theta, nphi=48, full=False):
    """Radial resonant force amplitude F_rad per unit b for particles (gam, theta arrays), one line."""
    gam = np.atleast_1d(gam); theta = np.atleast_1d(theta)
    p = np.sqrt(gam**2-1); uperp = p*np.sin(theta); upar = -p*np.cos(theta)
    ph = np.linspace(0, 2*np.pi, nphi, endpoint=False)[None, :]
    zeta = line["kx"]*uperp[:, None]
    v = np.stack([uperp[:, None]*np.cos(ph), uperp[:, None]*np.sin(ph), upar[:, None]*np.ones_like(ph)], -1)/gam[:, None, None]
    E, B = line["E"], line["B"]
    def f(Ev, Bv):
        return -(Ev[None, None, :] + np.cross(v, Bv[None, None, :]))
    fp, fq = f(E, B), f(E.conj(), B.conj())
    P = 0.5*(fp[..., 0]+1j*fp[..., 1])*np.exp(1j*zeta*np.sin(ph))*np.exp(-1j*ph)
    Q = 0.5*(fq[..., 0]+1j*fq[..., 1])*np.exp(-1j*zeta*np.sin(ph))*np.exp(-1j*ph)
    P1 = np.mean(P*np.exp(-1j*ph), 1); Q1 = np.mean(Q*np.exp(1j*ph), 1)
    if full:
        return np.abs(P1+np.conj(Q1)), np.sqrt(np.abs(P1)**2+np.abs(Q1)**2)      # radial amplitude, vector amplitude
    return np.abs(P1+np.conj(Q1))

def dxi_duperp(case, kzn, gam, theta):
    """d xi / d u_perp along the single-line invariant curve, xi = u_par/|u|, for each particle."""
    a = case.wn/kzn
    p = np.sqrt(gam**2-1); upar = -p*np.cos(theta); uperp = p*np.sin(theta)
    I = gam - a*upar
    def xi_of(g):
        up = (g-I)/a; s = np.maximum(g*g-1-up**2, 0.0)
        return up/np.sqrt(g*g-1), np.sqrt(s)
    h = 1e-6*gam
    x1, s1 = xi_of(gam+h); x0, s0 = xi_of(gam-h)
    return (x1-x0)/(s1-s0)

def build(case, line, N, theta_rng, delta_arr, seed, zrange=None):
    """Initial state with x_gc random along the wave phase, gyrophase random: psi0 = Phi_gc + phi uniform."""
    r = np.random.default_rng(seed)
    theta = r.uniform(*theta_rng, N)
    g = solve_gamma(case, line["kz"], delta_arr, theta)
    p = np.sqrt(g**2-1)
    phi = r.uniform(0, 2*np.pi, N)
    z0 = r.uniform(0, zrange if zrange else 2*np.pi/line["kz"], N)   # z0 spread >> 1/dk so lines get independent phases
    u = np.stack([p*np.sin(theta)*np.cos(phi), p*np.sin(theta)*np.sin(phi), -p*np.cos(theta)], 1)
    x = np.stack([p*np.sin(theta)*np.sin(phi), np.zeros(N), z0], 1)
    return jnp.asarray(x), jnp.asarray(u), theta, g

def quantities(u):
    u = np.asarray(u)
    s = np.sqrt(u[..., 0]**2+u[..., 1]**2); p = np.sqrt(np.sum(u**2, -1))
    return s, u[..., 2]/p, np.arctan2(s, -u[..., 2]), np.sqrt(1+p**2)

def rec_u(x, u, t):
    return u
