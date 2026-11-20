"""Reproduce Fulop eq. 24: gamma_max = (pi/16) wpr^2/wce exp(-1) = 1.3e-9 n_r/B_T at k_par c = 2 wce/cZ, omega0 = wce/cZ (n_e=5e19, B=2 T, n_r/n_e=5e-3)."""
import json
from fulop_validation import *
import qlib
from scipy.optimize import brentq
lnL = 15.; cZ = np.sqrt(18/np.pi)*lnL; Eo = 30.; alpha = (Eo-1)/2
kpar = 2*Oce/(cZ*c); w0 = Oce/cZ
rows=[]
for kperp in (300, 500, 750, 1000):
    w = omega_of_k(kperp, kpar, sp, 2*wci, 0.9*Oce); k = np.hypot(kperp, kpar)
    ex = np.exp(-Oce/((kpar*c-w)*cZ)); g22 = w*np.pi/(4*cZ)*(wpr**2/wpi**2)*(k**2*vA**2/w**2)*ex*(alpha+kpar*c/Oce)/alpha
    fg = fulop_f(cZ, alpha); pres = Oce/(kpar*c-w); pgrid = np.linspace(0.02, 6*np.sqrt(pres/alpha), 220)
    E = qlib.pol(w, kperp, kpar, sp); W1, _ = wave_W(w, sp, kperp, kpar, E)
    res = {}
    for mode in ("Ex", "full"):
        Ec_ = E if mode == "full" else np.array([E[0], 0, 0], complex)
        integ = np.zeros(len(pgrid))
        for j, pp in enumerate(pgrid):
            gf = lambda pz: w-kpar*c*pz/np.sqrt(1+pz*pz+pp*pp)+Oce/np.sqrt(1+pz*pz+pp*pp)
            pzs = np.linspace(1, 8*cZ+10, 3001); gv = gf(pzs)
            for i2 in np.where(np.sign(gv[:-1]) != np.sign(gv[1:]))[0]:
                pz = brentq(gf, pzs[i2], pzs[i2+1]); gam = np.sqrt(1+pz*pz+pp*pp); dg = abs((gf(pz+1e-6)-gf(pz-1e-6))/2e-6)
                f, fz, fp = fg(pz, pp); dfdg = (kpar*c*fz-Oce/pp*fp)/w
                A = qlib.A_n(Ec_, kperp, Oce, gam, 1, np.arctan2(pp, pz), -1)
                integ[j] += -(me*c**2/2)*np.pi*(e/(me*c))**2*A**2/dg*dfdg*2*np.pi*pp
        res[mode] = -np.trapezoid(integ, pgrid)/(2*W1)
    rows.append(dict(kperp=kperp, kpar=kpar, w_over_w0=w/(Oce/cZ), eq21=g22, Ex=res["Ex"], full=res["full"]))
    print(f"kperp={kperp} k_par={kpar:.0f} w/w0={w/(Oce/cZ):.2f}: eq21={g22:.3e}  Ex-only={res['Ex']:.3e} ({res['Ex']/g22:.2f})  full={res['full']:.3e} ({res['full']/g22:.2f}) ; eq24 max={np.pi/16*wpr**2/Oce*np.exp(-1):.3e}", flush=True)
json.dump(rows, open("../data/fulop_eq24.json", "w"), indent=1)
