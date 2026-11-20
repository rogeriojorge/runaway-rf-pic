"""Coupled wave-energy model:  dW/dt = P_in - Gamma_th W V - n_res V p_zone(b) m c^2,  W = u_w(b) per unit volume, b from W.
p_zone(b): wave energy absorbed per resonant (zone) runaway per second.  Three closures:
  'cap'  : measured/bounded, p = [P_QL(b) rho(sigma)] || P_cap  (harmonic combination: diffusion-limited, then replenishment-limited at P_cap)
  'QL'   : validated unsaturated quasilinear rate with the overlap factor rho(sigma) (the Step-2 hypothesis, no replenishment limit)
  'S2'   : Step-2 monoenergetic rate Gamma_RE = 1.1e8 /s per 1e15 m^-3 with rho(sigma)
All energies in J, powers in W."""
import numpy as np, json, os
mu0 = 4e-7*np.pi; MC2 = 8.1871057769e-14
B0 = 1.475; V = 18.0; NEFF = 11.3; B_THR = 1.0e-4          # per-line threshold (P3), 0.6-1.6e-4
# overlap factor rho(sigma) = D_eff/D_QL, Gaussian 4.5 comb, onset windows (P3 S_gauss45_f0.pkl); sigma = b/B_UNIT^... sigma(b)=sqrt(b/bu)
SIG = np.array([0.0, 0.45, 0.53, 0.62, 0.72, 0.85, 0.99, 1.16, 1.36, 1.59, 1.87, 2.19, 2.56, 3.0])
RHO = np.array([1e-5, 1e-4, 6e-4, 2.2e-3, 5.6e-3, 0.15, 0.70, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])   # capped at 1 (values >1 are short-window overshoot, P3)
BU = 1e-4/0.99**2
def sigma_of_b(b): return np.sqrt(np.asarray(b)/BU)
def rho(b): return np.exp(np.interp(sigma_of_b(b), SIG, np.log(RHO)))
def u_w(b): return NEFF*np.asarray(b)**2*B0**2/(4*mu0)           # Step-2 convention (magnetic energy only)
# quasilinear absorption (this work, qlib/diiid_ql.py): Gamma_P = 9.5e6 /s per 1e15 m^-3 total RE (P1 shape); zone fraction 0.466
GQL_PER_TOT = 9.5e6/1e15; FZONE = 0.466; WFAC = 1.12
def P_QL_zone(b):
    """QL power per zone particle (W) at wave level b"""
    return GQL_PER_TOT*WFAC*u_w(b)/FZONE
def p_zone(b, closure="cap", Pcap=5.0, gS2=1.1e8/1e15):
    """absorbed power per resonant particle, W"""
    if closure == "S2": return gS2*rho(b)*u_w(b)
    pQL = P_QL_zone(b)*rho(b)
    if closure == "QL": return pQL
    cap = Pcap*MC2
    return 1.0/(1.0/np.maximum(pQL, 1e-300)+1.0/cap)
def loss(b, Gth, nres, closure="cap", Pcap=5.0, Vp=V):
    return Gth*u_w(b)*Vp + nres*Vp*p_zone(b, closure, Pcap)
def solve_b(Pin, Gth, nres, closure="cap", Pcap=5.0, Vp=V, nb=4000):
    bs = np.geomspace(1e-6, 1e-2, nb); L = loss(bs, Gth, nres, closure, Pcap, Vp)
    idx = np.where((L[:-1]-Pin)*(L[1:]-Pin) < 0)[0]
    roots = []
    for i in idx:
        f = (np.log(Pin)-np.log(L[i]))/(np.log(L[i+1])-np.log(L[i]))
        b = np.exp(np.log(bs[i])+f*(np.log(bs[i+1])-np.log(bs[i])))
        stable = L[i+1] > L[i]
        roots.append((b, stable))
    return roots
def b_of_P(Pin, Gth, nres, closure="cap", Pcap=5.0, Vp=V):
    r = [b for b, s in solve_b(Pin, Gth, nres, closure, Pcap, Vp) if s]
    return r
def pth_at_thr(Gth, Vp=V): return Gth*u_w(B_THR)*Vp
if __name__ == "__main__":
    for G in (4e4, 9e4, 2.7e5, 6e5, 1.4e6):
        print(f"Gamma_th={G:.1e}: P_th(b_thr)={pth_at_thr(G)/1e3:.1f} kW; absorption cap per 1e15 zone RE (Pcap=5.0): {1e15*V*2.2*MC2/1e3:.2f} kW; QL(b_thr) per 1e15: {1e15*V*p_zone(B_THR,'QL')/1e3:.0f} kW; S2 per 1e15: {1e15*V*p_zone(B_THR,'S2')/1e3:.0f} kW")
