"""Linear/QL power absorption of the DIII-D helicon (476 MHz, k_par = 41/m, n=+1 normal Doppler) by runaways with the P1 kinetic distribution shape
(JAX-in-Cell stationary tail, results/P1/data/full_fp.npz) and by bounded model distributions.  Writes data/diiid_ql.json."""
import json, sys
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from qlib import *
import os
B = 1.4; ne = 0.8e19; f_rf = 476e6; w = 2*np.pi*f_rf; sp = pl.species(B, ne); Oce = e*B/me
# ---- P1 distribution (gamma, theta) pooled over the stationary window
d = np.load(os.path.expanduser("results/P1/data/full_fp.npz")); t = d["t"]; sel = t >= 60
U = d["eta"][sel].reshape(-1, 3).astype(float); p = np.linalg.norm(U, axis=1); th = np.arccos(np.clip(U[:, 0]/p, -1, 1)); gam = np.sqrt(1+p**2)   # x is along B in P1
wts = np.tile(d["wt"], sel.sum()); wts /= wts.sum()
pe = np.linspace(2, 60, 117); te = np.linspace(0, 0.8, 81)
H, _, _ = np.histogram2d(p, th, bins=[pe, te], weights=wts)
pc = 0.5*(pe[1:]+pe[:-1]); tc = 0.5*(te[1:]+te[:-1]); dp = pe[1]-pe[0]; dt = te[1]-te[0]
# density per d^3p: dN/(2 pi p^2 sin(th) dp dth)
F = H/(2*np.pi*pc[:, None]**2*np.sin(tc)[None, :]*dp*dt)
Fs = gaussian_filter(F, sigma=(1.5, 1.5), mode="nearest")
interp = RegularGridInterpolator((pc, tc), Fs, bounds_error=False, fill_value=0.0)
def dist(nRE):
    """fgrad(pz, pp) for p_par<0 (counter-propagating to the wave), normalised to n_RE total (all p in the binned range)."""
    norm = np.sum(F*2*np.pi*pc[:, None]**2*np.sin(tc)[None, :]*dp*dt)
    def G(a, b):
        return interp([[np.hypot(a, b), np.arctan2(b, a)]])[0]
    def fg(pz, pp):
        a = abs(pz); b = max(pp, 1e-3); h = 0.3
        f = G(a, b)
        ga = (G(a+h, b)-G(max(a-h, 0.05), b))/(a+h-max(a-h, 0.05)); gb = (G(a, b+h)-G(a, max(b-h, 1e-3)))/(b+h-max(b-h, 1e-3))
        s = nRE/norm
        return f*s, np.sign(pz)*ga*s, gb*s
    return fg, norm
if __name__ == "__main__":
    rows = []
    fg, norm = dist(1e15)
    # mean P1 stats
    g_mean = np.sum(wts*gam); print("P1 pooled <gamma>", g_mean, " fraction gamma 14.5-20.5 & theta<0.54:", np.sum(wts[(gam > 14.5) & (gam < 20.5) & (th < 0.54)]))
    kpar_list = [41.0, 41.0+0.57*np.arange(-8, 9)]
    res = {}
    for kpar in (41.0, 35.3, 38.15, 43.85, 46.7):
        kperp, _ = pl.solve_kperp(w, kpar, sp); E = pol(w, kperp, kpar, sp)
        pg = np.linspace(0.3, 14.0, 90)
        P, Wt, gk = gain_power(w, kperp, kpar, sp, Oce, +1, fg, pg, (-60.0, -1.0), E=E)
        res[kpar] = dict(kperp=kperp, P_per_E2=P, W_per_E2=Wt, gamma_amp=gk, Gamma_P=2*gk)
        print(f"k_par={kpar}: k_perp={kperp:.1f} Gamma_P(power rate) = {2*gk:.3e} /s per 1e15 total RE (negative = growth)  P={P:.3e} W/m3 per (V/m)^2  W={Wt:.3e}", flush=True)
    json.dump({str(k): v for k, v in res.items()}, open("../data/diiid_ql.json", "w"), indent=1)
