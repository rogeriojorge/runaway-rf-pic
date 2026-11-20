"""Collisional (e-i, Krook omega->omega+i nu on electrons) damping of the cold magnetosonic/whistler/fast wave: amplitude rate Im(omega)/nu_ei, and comparison
with Fulop 2006 eq. 25 (gamma_d = 1.5/tau_ei).  Complex root of det M at fixed real k."""
import numpy as np, json
from scipy.optimize import brentq
e, me, c, eps0 = 1.602176634e-19, 9.1093837015e-31, 2.99792458e8, 8.8541878128e-12
mD = 2*1.66053906660e-27
def tau_ei(ne, TeV, Z, lnL):                         # Fulop eq. 25 definition
    vT = np.sqrt(2*TeV*e/me)
    return 3*np.pi**1.5*me**2*vT**3*eps0**2/(ne*Z**2*e**4*lnL)
def nu_ei_spitzer(ne, TeV, Z, lnL):                  # PPCF B9 (used in step 2)
    return 4*np.sqrt(2*np.pi)*ne*Z*e**4*lnL/(3*(4*np.pi*eps0)**2*np.sqrt(me)*(TeV*e)**1.5)
def Mmat(w, B, ne, kperp, kpar, nu):
    wpe2 = ne*e**2/(eps0*me); wpi2 = ne*e**2/(eps0*mD); Oe = -e*B/me; Oi = e*B/mD
    we = w+1j*nu
    R = 1 - wpe2/(w*(we+Oe)) - wpi2/(w*(w+Oi)); L = 1 - wpe2/(w*(we-Oe)) - wpi2/(w*(w-Oi)); P = 1 - wpe2/(w*we) - wpi2/w**2
    S = (R+L)/2; D = (R-L)/2; npe, npa = kperp*c/w, kpar*c/w; n2 = npe**2+npa**2
    return np.array([[S-npa**2, -1j*D, npe*npa], [1j*D, S-n2, 0], [npe*npa, 0, P-npe**2]])
def damping(B, ne, TeV, Z, lnL, kperp, kpar, w_guess):
    nu = nu_ei_spitzer(ne, TeV, Z, lnL)
    f = lambda w: np.linalg.det(Mmat(w, B, ne, kperp, kpar, nu))
    wc = w_guess+0j
    for _ in range(80):
        h = 1e-6*abs(wc); df = (f(wc+h)-f(wc-h))/(2*h); dw = -f(wc)/df; wc += dw
        if abs(dw) < 1e-10*abs(wc): break
    return wc, nu
def real_root(B, ne, kperp, kpar, wlo, whi):
    f = lambda w: np.linalg.det(Mmat(w, B, ne, kperp, kpar, 0.0)).real
    ws = np.geomspace(wlo, whi, 6000); v = np.array([f(w) for w in ws]); i = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return brentq(f, ws[i[0]], ws[i[0]+1])
if __name__ == "__main__":
    out = []
    wci = lambda B: e*B/mD
    cases = [("Fulop Fig3: ne=5e19 B=2 T=10eV k=450 th=85", 2.0, 5e19, 10., 1.0, 15., 450., 85.),
             ("Fulop Fig3 k=150 th=85", 2.0, 5e19, 10., 1.0, 15., 150., 85.),
             ("Fulop-like parallel whistler k=450 th=20", 2.0, 5e19, 10., 1.0, 15., 450., 20.),
             ("DIII-D helicon: ne=0.8e19 B=1.4 T=2165, k_par=41, fast kperp (476 MHz)", 1.4, 0.8e19, 2165., 1.5, 17., None, None)]
    for name, B, ne, TeV, Z, lnL, k, thd in cases:
        if k is None:
            w0 = 2*np.pi*476e6; kpar = 41.0
            # fast root in kperp at real w: scan kperp
            def g(kp): return np.linalg.det(Mmat(w0, B, ne, kp, kpar, 0.0)).real
            kps = np.linspace(5, 400, 4000); v = np.array([g(x) for x in kps]); i = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
            kperp = brentq(g, kps[i[0]], kps[i[0]+1])
            wc, nu = damping(B, ne, TeV, Z, lnL, kperp, kpar, w0); w = w0
        else:
            kperp, kpar = k*np.sin(np.deg2rad(thd)), k*np.cos(np.deg2rad(thd))
            w = real_root(B, ne, kperp, kpar, 1.5*wci(B), 0.3*e*B/me)
            wc, nu = damping(B, ne, TeV, Z, lnL, kperp, kpar, w)
        tau = tau_ei(ne, TeV, Z, lnL); gam = -wc.imag
        r = dict(case=name, w_over_wci=w/wci(B), kperp=kperp, kpar=kpar, nu_ei=nu, tau_ei_inv=1/tau, nu_tau=nu*tau,
                 gamma_amp=gam, gamma_over_nu=gam/nu, gamma_over_tauinv=gam*tau, fulop_gamma_d_over_nu=1.5/(tau*nu))
        out.append(r); print(json.dumps(r))
    json.dump(out, open("../data/collisional_damping.json", "w"), indent=1)
