"""Analytic / semi-analytic resonance theory for one plane wave line (numpy)."""
import numpy as np

def coupling(line, wn, gam, theta, nphi=512, uz_sign=-1):
    """Gyro-averaged resonant force on W=(u_x+i u_y)e^{-i phi} per unit b (normalised units).
    returns P1, Q (complex): dW/dt|res = P1 e^{i psi} + Q e^{-i psi}.  Radial force amplitude F_rad=|P1+conj(Q)|."""
    p = np.sqrt(gam**2-1); uperp, upar = p*np.sin(theta), uz_sign*p*np.cos(theta)
    ph = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    zeta = line["kx"]*uperp
    v = np.stack([uperp*np.cos(ph), uperp*np.sin(ph), np.full_like(ph, upar)], 1)/gam
    E, B = line["E"], line["B"]
    def f(Ev, Bv):  # -(E + v x B)
        return -(Ev[None, :] + np.cross(v, Bv[None, :]))
    fp = f(E, B); fq = f(E.conj(), B.conj())
    P = 0.5*(fp[:, 0]+1j*fp[:, 1])*np.exp(1j*zeta*np.sin(ph))*np.exp(-1j*ph)
    Q = 0.5*(fq[:, 0]+1j*fq[:, 1])*np.exp(-1j*zeta*np.sin(ph))*np.exp(-1j*ph)
    P1 = np.mean(P*np.exp(-1j*ph))
    Q1 = np.mean(Q*np.exp(1j*ph))
    return P1, Q1

def frad(line, wn, gam, theta):
    P1, Q1 = coupling(line, wn, gam, theta)
    return abs(P1+np.conj(Q1)), P1, Q1

def delta(wn, kz, gam, upar):
    return wn - kz*upar/gam - 1.0/gam

def kinematics(wn, kz, gam, theta, uz_sign=-1):
    """Quantities along the wave-frame invariant curve gamma - a u_par = I (a = w/kz)."""
    a = wn/kz
    p = np.sqrt(gam**2-1); upar = uz_sign*p*np.cos(theta); uperp = p*np.sin(theta)
    I = gam - a*upar
    ddelta = -(kz**2/wn - wn)/gam                      # d delta/d gamma at resonance, fixed I
    dgam_per_F = uperp/(gam - upar/a)                  # d gamma/dt = dgam_per_F * F_rad * cos(.)
    return dict(a=a, I=I, uperp=uperp, upar=upar, ddelta=ddelta, G_per_F=dgam_per_F)

def omega_b(line, wn, gam, theta, b):
    k = kinematics(wn, line["kz"], gam, theta)
    F = frad(line, wn, gam, theta)[0]*b
    return np.sqrt(k["G_per_F"]*F*abs(k["ddelta"]))

def dtheta_dgamma(wn, kz, gam, I, uz_sign=-1):
    """d theta / d gamma along the invariant curve (theta = atan2(u_perp, -u_par))."""
    a = wn/kz
    def th(g):
        upar = (g-I)/a; s = g*g-1-upar**2
        return np.arctan2(np.sqrt(max(s, 0)), -upar)
    h = 1e-5*gam
    return (th(gam+h)-th(gam-h))/(2*h)
