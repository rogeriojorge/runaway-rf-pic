"""b_thr -> coupled power.  Models from results/P2/NOTES.md:
 (single pass)  P = u_w v_g A,  v_g,par in [0.09,0.28] c (cold fast wave, data/mapping.json), beam cross-section A in [1,4] m^2
 (damping)      P = Gamma_P u_w V_p,  V_p = 18 m^3, Gamma_P in [1e3, 1e7] 1/s (total wave-energy loss rate: collisional/Landau/antenna-edge)
 u_w = b_tot^2 B0^2/(4 mu0) with b_tot^2 = sum_j b_j^2 (peak line amplitudes, power adds incoherently) = N_eff b_pk^2."""
import numpy as np
mu0 = 4e-7*np.pi; B0 = 1.4; c = 2.99792458e8; Vp = 18.0
def u_w(btot): return btot**2*B0**2/(4*mu0)
def P_single(btot, vg=0.28, A=1.0): return u_w(btot)*vg*c*A
def P_damp(btot, Gam): return u_w(btot)*Gam*Vp
NEFF = {"gauss34": 8.52, "gauss45": 11.28, "gauss70": 17.55}
def table(bpk, neff):
    bt = bpk*np.sqrt(neff)
    rows = dict(b_tot=bt, single_lo=P_single(bt, 0.09, 1.0), single_hi=P_single(bt, 0.28, 4.0))
    for G in (1e3, 1e4, 1e5, 1e6, 1e7): rows[f"G{G:g}"] = P_damp(bt, G)
    return rows
def gamma_needed(P, bpk, neff): return P/(u_w(bpk*np.sqrt(neff))*Vp)
if __name__ == "__main__":
    for bpk in (1.0e-4, 1.5e-4):
        for n, ne in NEFF.items():
            r = table(bpk, ne); print(f"b_pk={bpk:.1e} {n}: b_tot={r['b_tot']:.2e} single [{r['single_lo']/1e3:.3g},{r['single_hi']/1e3:.3g}] kW; "+"; ".join(f"G={k[1:]}: {v/1e3:.3g} kW" for k, v in r.items() if k.startswith("G")))
