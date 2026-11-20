"""Reproduce Fulop et al. 2006 (PoP 13 062506) Fig. 3 / eqs. 21-24 growth rates with the energy-balance QL code (qlib.py).
Case (Fig. 3 caption): n_e = 5e19, B = 2 T, T = 10 eV, n_r/n_e = 5e-3, deuterium, theta_k = 85 deg."""
import sys, json
import numpy as np
from qlib import *
B = 2.0; ne = 5e19; nr = 5e-3*ne; Z = 1.0
sp = pl.species(B, ne)               # electrons + deuterium
Oce = e*B/me; wci = e*B/(2*1.66053906660e-27)
wpe = np.sqrt(ne*e**2/(eps0*me)); wpi = np.sqrt(ne*e**2/(eps0*2*1.66053906660e-27)); wpr = np.sqrt(nr*e**2/(eps0*me))
vA = c*wci/wpi
def analytic(k, theta, lnL, E_over=30.0):
    cZ = np.sqrt(3*(Z+5)/np.pi)*lnL
    kpar = k*np.cos(theta); kperp = k*np.sin(theta)
    w = omega_of_k(kperp, kpar, sp, 2*wci, 0.9*Oce)
    # Fulop eq. 22 (k^2 vA^2/omega0^2 form, consistent with eq. 24) and eq. 21 (alpha factor)
    alpha = (E_over-1)/(1+Z); K_par = kpar*c/Oce
    ex = np.exp(-Oce/((kpar*c-w)*cZ))
    g22 = w*np.pi/(4*cZ)*(wpr**2/wpi**2)*(k**2*vA**2/w**2)*ex
    g21 = g22*(alpha+K_par)/alpha
    return w, g21, g22
def fulop_f(cZ, alpha):
    def fg(pz, pp):
        if pz <= 0.5: return 0.0, 0.0, 0.0
        f = nr*alpha/(2*np.pi*cZ*pz)*np.exp(-pz/cZ-alpha*pp**2/(2*pz))
        fz = f*(-1/pz-1/cZ+alpha*pp**2/(2*pz**2)); fp = f*(-alpha*pp/pz)
        return f, fz, fp
    return fg
if __name__ == "__main__":
    lnL = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    Eo = 30.0
    cZ = np.sqrt(3*(Z+5)/np.pi)*lnL; alpha = (Eo-1)/(1+Z)
    print(f"lnL={lnL} cZ={cZ:.2f} alpha={alpha:.1f} omega_ci={wci:.3e} omega_ce={Oce:.3e}  analytic max eq24 = {np.pi/16*wpr**2/Oce*np.exp(-1):.3e} = {np.pi/16*wpr**2/Oce*np.exp(-1)/wci:.2f} wci")
    rows = []
    th = np.deg2rad(85.0)
    for k in (150, 250, 350, 450, 550, 650, 750, 850):
        w, g21, g22 = analytic(k, th, lnL, Eo)
        kpar, kperp = k*np.cos(th), k*np.sin(th)
        fg = fulop_f(cZ, alpha)
        pres = Oce/(kpar*c-w)
        pgrid = np.linspace(0.02, 6*np.sqrt(pres/alpha), 160)
        P, Wt, gk = gain_power(w, kperp, kpar, sp, Oce, -1, fg, pgrid, (1.0, 8*cZ+10))
        rows.append(dict(k=k, w=w, w_over_wci=w/wci, eq21=g21, eq22=g22, QL_energy=gk, ratio_QL_eq21=gk/g21, pres=pres))
        print(f"k={k:4d} w/wci={w/wci:7.2f} p_res={pres:6.1f} | eq22/wci={g22/wci:7.3f} eq21/wci={g21/wci:7.3f} | energy-balance/wci={gk/wci:7.3f} ratio(E-bal/eq21)={gk/g21:5.2f}", flush=True)
    json.dump(dict(lnL=lnL, cZ=cZ, rows=rows), open(f"../data/fulop_lnL{lnL:g}.json", "w"), indent=1)
