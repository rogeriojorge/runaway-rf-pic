"""DIII-D mapping: omega_b scale, decorrelation rates, b_cross candidates, and wave power <-> amplitude. Writes data/mapping.json."""
import os, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import bw, plasma as pl, theory as th, drift
from exp import frad_vec, solve_gamma

case = bw.Case(); wn = case.wn; Om = case.Om
line = case.line(41.0); kzn = line["kz"]
res = {}
# ---- group velocity from the cold-plasma dispersion function D(w,kperp,kpar)=det M
def Dfun(w, kp, kz):
    S, D, P = pl.stix(w, case.sp)
    M = pl.wavematrix(kp*pl.c/w, kz*pl.c/w, S, D, P)
    return np.linalg.det(M).real
w0, kp0, kz0 = case.w, line["kperp"], 41.0
h = 1e-6
dDdw = (Dfun(w0*(1+h), kp0, kz0)-Dfun(w0*(1-h), kp0, kz0))/(2*h*w0)
dDdkp = (Dfun(w0, kp0*(1+h), kz0)-Dfun(w0, kp0*(1-h), kz0))/(2*h*kp0)
dDdkz = (Dfun(w0, kp0, kz0*(1+h))-Dfun(w0, kp0, kz0*(1-h)))/(2*h*kz0)
vg = -np.array([dDdkp, dDdkz])/dDdw
res["v_g_perp_par_over_c"] = (vg/pl.c).tolist(); res["v_ph_par_over_c"] = float(w0/kz0/pl.c)
print("cold dispersion: k_perp = %.2f 1/m ; v_g(perp,par)/c = %s ; omega/kpar/c = %.3f" % (kp0, vg/pl.c, w0/kz0/pl.c))

# ---- bounce frequency scale
thetas = [0.05, 0.1, 0.2, 0.3]
tab = {}
for t_ in thetas:
    g = case.gamma_res(kzn, -1, t_)
    Xi = th.omega_b(line, wn, g, t_, 1.0)**2*Om**2
    tab[t_] = dict(gamma_res=g, KE_MeV=(g-1)*0.511, Xi=Xi, wb_1e5=np.sqrt(Xi*1e-5), wb_1e4=np.sqrt(Xi*1e-4), wb_1e3=np.sqrt(Xi*1e-3),
                   island_hw_gamma_1e4=2*np.sqrt(Xi*1e-4)/Om/abs(th.kinematics(wn, kzn, g, t_)["ddelta"]))
    A, dg, dth = drift.sweep_rate(case, kzn, g, t_)
    tab[t_]["A_sweep"] = A*Om**2
    # collisional decorrelation: frequency diffusion D_delta from pitch scattering, Gamma_c = D_delta^(1/3)
    p = np.sqrt(g*g-1); nuD = (case.Zeff+1)*g/p**3/case.tau_c                    # 1/s
    Ddel = 0.5*(kzn*np.sqrt(1-1/g**2)*np.sin(t_)*Om)**2*nuD                       # rad^2/s^3  (beta ~ 1)
    tab[t_]["nu_D"] = nuD; tab[t_]["Gamma_coll"] = Ddel**(1/3)
res["table"] = {str(k): v for k, v in tab.items()}
for k, v in tab.items():
    print(f"theta={k}: gamma={v['gamma_res']:.3f} KE={v['KE_MeV']:.2f} MeV  Xi=omega_b^2/b={v['Xi']:.3e}  wb(b=1e-4)={v['wb_1e4']:.2e}/s  island_hw(gamma,1e-4)={v['island_hw_gamma_1e4']:.3f}  A_sweep={v['A_sweep']:.2e}  nu_D={v['nu_D']:.3e}/s  Gamma_coll={v['Gamma_coll']:.2e}/s")

# ---- b_cross candidates: b = R_c * Gamma^2 / Xi at theta=0.2 (R_c filled in from the measured crossover; here R_c=1 and the factor reported)
t_ = 0.2; v = tab[t_]; Xi = v["Xi"]
cand = {
  "E-sweep + friction + RR (A)": v["A_sweep"],
  "collisional frequency diffusion (Gamma_c^2)": v["Gamma_coll"]**2,
  "launch bandwidth 10 MHz (Delta omega=2pi 10 MHz)^2": (2*np.pi*10e6)**2,
  "toroidal mode spacing dk=1/R (0.44/m): (dk c cos th)^2": (0.44*pl.c*np.cos(t_))**2,
  "transit through a 1 m wave region: (c/1m)^2": (pl.c/1.0)**2,
  "transit through a 0.1 m wave region: (c/0.1m)^2": (pl.c/0.1)**2,
}
res["b_cross_unit_R"] = {k: dict(rate2=x, b=x/Xi) for k, x in cand.items()}
for k, x in cand.items():
    print(f"b at R=1 for {k:60s}: {x/Xi:.3e}   (decorrelation rate {np.sqrt(x):.3e}/s)")

# ---- power <-> amplitude
mu0 = pl.mu0; B0 = case.B0
def b_from_P(P, vflux, area):          # single pass: P = u_w * v_flux * area, u_w = b^2 B0^2/(4 mu0) (time-avg magnetic, peak amplitude b)
    return np.sqrt(4*mu0*P/(vflux*area))/B0
def b_from_damping(P, Gamma_P, V):     # steady state: P = Gamma_P * u_w * V
    return np.sqrt(4*mu0*P/(Gamma_P*V))/B0
P = 150e3
out = {}
vpar = abs(vg[1]); out["vg_par"] = vpar
for A_ in (1.0, 2.0, 4.0):
    out[f"single_pass_area_{A_}m2"] = b_from_P(P, vpar, A_)
for GP in (1e3, 1e4, 1e5, 1e6, 1e7):
    out[f"damping_{GP:.0e}"] = b_from_damping(P, GP, 18.0)
res["power_to_b_150kW"] = out
print("b(150 kW): " + ", ".join(f"{k}={v:.2e}" for k, v in out.items()))
# Kang's 155 kW formula check with their numbers: b=4.75e-4, B0=3.5, V_p assumed -> implied |omega_i|/2pi
Wk = (4.75e-4*3.5)**2/(2*mu0)
res["kang_check"] = dict(W_J_m3=Wk, implied_rate_per_s_for_Vp=lambda_ if False else None) if False else dict(W_J_m3=Wk)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mapping.json"), "w"), indent=1, default=float)
