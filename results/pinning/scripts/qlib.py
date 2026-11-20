"""Quasilinear (and linear) absorption of a cold-plasma wave by a relativistic electron distribution.
Independent of the P2/P3 test-particle coupling except for the cold-plasma polarisation helper (P2 plasma.py).

Derivation (see NOTES.md sec. 1):  dgamma/dt of a resonant electron in a mode of peak field E0 is  c_n cos(Psi),
c_n = (e E0/(m c)) |A_n|,  A_n = <(E_hat.beta) exp(i zeta sin phi - i n phi)>_phi   (gyro average, zeta = k_perp p_perp c/omega_ce).
Random phase:  <dgamma^2>/t = pi c_n^2 delta(omega - k_par v_par - n Omega/gamma).  The QL operator diffuses along the resonance characteristic
d p_par/d gamma = k_par c/omega,  d p_perp/d gamma = n omega_ce/(omega p_perp)  (Kennel-Engelmann), whence the power given to the particles
P = -(m c^2/2) Int d^3p D_gg d_gamma f|_char , and the amplitude growth rate gamma_k = -P/(2 W),  W = time-averaged cold wave energy density."""
import os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.expanduser("results/P2/scripts"))
import plasma as pl
e, me, c, eps0, mu0 = pl.e, pl.me, pl.c, pl.eps0, pl.mu0

def Kmat(w, sp):
    S, D, P = pl.stix(w, sp)
    return np.array([[S, -1j*D, 0], [1j*D, S, 0], [0, 0, P]])

def wave_W(w, sp, kperp, kpar, E):
    h = 1e-5*w
    dwK = ((w+h)*Kmat(w+h, sp)-(w-h)*Kmat(w-h, sp))/(2*h)
    Wc = 0.25*eps0*np.real(E.conj()@dwK@E)
    Bv = np.cross(np.array([kperp, 0, kpar]), E)/w
    return Wc + np.real(Bv.conj()@Bv)/(4*mu0), Wc

def omega_of_k(kperp, kpar, sp, wlo, whi, nscan=4000):
    """cold-plasma root w of det M(w; k) in [wlo, whi] (lowest-frequency root with a sign change)"""
    def d(w):
        S, D, P = pl.stix(w, sp); n2 = (kperp**2+kpar**2)*c**2/w**2
        M = pl.wavematrix(kperp*c/w, kpar*c/w, S, D, P)
        return np.linalg.det(M).real
    ws = np.geomspace(wlo, whi, nscan); v = np.array([d(w) for w in ws])
    idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    if len(idx) == 0: return None
    i = idx[0]; return brentq(d, ws[i], ws[i+1], xtol=1e-12*ws[i])

def pol(w, kperp, kpar, sp):
    E, B, res = pl.polarization(w, kperp, kpar, sp)
    return E

def A_n(E, kperp, Om_e, gam, upar_over_p_sign, pitch, n, nphi=256):
    """gyro-averaged resonant energy-gain amplitude per unit (e E0/(m c)); E unit complex vector in (x,y,z). pitch theta from B; u_par = sign*p cos(theta)."""
    p = np.sqrt(gam**2-1); up = p*np.sin(pitch); upar = upar_over_p_sign*p*np.cos(pitch)
    ph = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    zeta = kperp*up*c/Om_e
    beta = np.stack([up*np.cos(ph), up*np.sin(ph), np.full_like(ph, upar)], 1)/gam
    Ev = beta@E
    return abs(np.mean(Ev*np.exp(1j*zeta*np.sin(ph))*np.exp(-1j*n*ph)))

def gain_power(w, kperp, kpar, sp, Oce, n, fgrad, pperp_grid, ppar_range, E=None, E0=1.0, detail=False):
    """P (W/m^3) given to electrons by one mode of peak field E0 (V/m).  fgrad(ppar, pperp) -> (f, df/dppar, df/dpperp), f in m^-3 (mc)^-3.
    returns P, W_wave, growth rate gamma_k = -P/(2W) (amplitude, 1/s)."""
    if E is None: E = pol(w, kperp, kpar, sp)
    W1, _ = wave_W(w, sp, kperp, kpar, E); Wt = W1*E0**2          # E is unit-norm
    integ = np.zeros(len(pperp_grid)); info = []
    for j, pp in enumerate(pperp_grid):
        def g(pz):
            gam = np.sqrt(1+pz*pz+pp*pp); return w - kpar*c*pz/gam - n*Oce/gam
        pzs = np.linspace(*ppar_range, 3001); gv = g(pzs)
        for i in np.where(np.sign(gv[:-1]) != np.sign(gv[1:]))[0]:
            pz = brentq(g, pzs[i], pzs[i+1]); gam = np.sqrt(1+pz*pz+pp*pp)
            dg = abs((g(pz+1e-6)-g(pz-1e-6))/2e-6)
            f, fz, fp = fgrad(pz, pp)
            dfdg = (kpar*c*fz + n*Oce/pp*fp)/w
            A = A_n(E, kperp, Oce, gam, np.sign(pz) if pz != 0 else 1, np.arctan2(pp, abs(pz)), n)
            Dgg = np.pi*(e*E0/(me*c))**2*A**2/dg
            integ[j] += -(me*c**2/2)*Dgg*dfdg*2*np.pi*pp
            info.append((pp, pz, gam, A, dfdg))
    P = np.trapezoid(integ, pperp_grid)
    return (P, Wt, -P/(2*Wt), info) if detail else (P, Wt, -P/(2*Wt))
