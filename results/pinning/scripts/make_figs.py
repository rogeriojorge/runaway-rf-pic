import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pin_model as pm
D = "../data/"
# --- Fig: Fulop validation
rows = json.load(open(D+"fulop_check2.json")); r85 = [r for r in rows if r["theta"] == 85]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
k = [r["k"] for r in r85]
ax[0].plot(k, [r["Ex"]/r["eq21"] for r in r85], "o-", label="energy balance, E_x coupling only (paper's)")
ax[0].plot(k, [r["full"]/r["eq21"] for r in r85], "s-", label="energy balance, full cold polarisation")
ax[0].axhline(1, color="k", ls="--"); ax[0].set_xlabel("k [1/m] (theta_k = 85 deg)"); ax[0].set_ylabel("growth rate / Fulop eq. 21"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.2)
ax[0].set_title("Fulop 2006 case (n_e=5e19, B=2 T, n_r/n_e=5e-3, D)", fontsize=9)
cd = json.load(open(D+"collisional_damping.json"))
ax[1].bar(range(len(cd)), [c["gamma_over_nu"] for c in cd], color="C0"); ax[1].axhline(1.5, color="r", ls="--", label="Fulop eq. 25: 1.5/tau_ei (= 1.5 nu_ei)")
ax[1].set_yscale("log"); ax[1].set_xticks(range(len(cd))); ax[1].set_xticklabels(["Fulop\nk=450,85deg", "Fulop\nk=150,85deg", "parallel-ish\nk=450,20deg", "DIII-D\nfast wave"], fontsize=8)
ax[1].set_ylabel("collisional amplitude damping / nu_ei"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("../fig1_validation.png", dpi=130); plt.close(fig)
# --- Fig: P per zone particle vs b from sims + models
S = {}
for f in glob.glob(D+"pin_s*_b*_lam*.json"):
    o = json.load(open(f))
    if o["bpk"] < 1e-8 or "dt0125" in f: continue
    S.setdefault(o["lam"], []).append((o["bpk"], o["P_wave"]))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
bb = np.geomspace(1e-5, 1e-3, 200)
for lam, mk in zip(sorted(S, reverse=True), ["o", "s", "^"]):
    d = np.array(sorted(S[lam])); ax[0].plot(d[:, 0], d[:, 1]/lam, mk+"-", label=f"sim lam={lam:g}: P_wave/lam")
ax[0].plot(bb, pm.p_zone(bb, "QL")/pm.MC2, "k--", label="unsaturated QL x rho(sigma)")
ax[0].axhspan(1.1, 7.7, color="r", alpha=0.12, label="replenishment-limited range 1.1-7.7"); ax[0].axhline(5.0, color="r", ls=":"); ax[0].axvline(pm.B_THR, color="gray", ls="-.")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("b_pk per line"); ax[0].set_ylabel("wave work per zone electron, mc^2/s (physical-time equivalent)")
ax[0].legend(fontsize=7); ax[0].set_title("Driven replenished test-particle ensemble (reduced model)", fontsize=9)
# P_wave/lam vs lam at two b
for b0 in (1e-4, 3e-4):
    pts = sorted([(lam, [p for bb_, p in S[lam] if abs(bb_-b0)/b0 < 1e-6][0]) for lam in S if any(abs(bb_-b0)/b0 < 1e-6 for bb_, p in S[lam])])
    pts = np.array(pts); ax[1].loglog(pts[:, 0], pts[:, 1]/pts[:, 0], "o-", label=f"b={b0:.0e}")
ax[1].set_xlabel("time-compression lam (1 = physical)"); ax[1].set_ylabel("P_wave/lam [mc^2/s]"); ax[1].legend(); ax[1].axhspan(1.1, 7.7, color="r", alpha=0.12)
ax[1].set_title("approach to the replenishment-limited regime as lam -> 1", fontsize=9)
fig.tight_layout(); fig.savefig("../fig2_Pwave_vs_b.png", dpi=130); plt.close(fig)
# --- Fig: pinning feasibility map
n = np.geomspace(1e11, 1e17, 240); G = np.geomspace(3e4, 2e6, 240); NN, GG = np.meshgrid(n, G)
fig, ax = plt.subplots(figsize=(7, 5.4))
R = NN*pm.V*5.0*pm.MC2/(GG*pm.u_w(pm.B_THR)*pm.V)
cs = ax.contourf(NN, GG, np.log10(R), levels=np.arange(-5, 2.1, 1), cmap="RdYlBu"); plt.colorbar(cs, label="log10( P_abs,cap / P_th(b_thr) ),  cap = 5 mc^2/s per zone electron")
for Pc, ls_, lab in [(1.1, ":", "cap 1.1"), (5.0, "-", "cap 5.0 (nominal)"), (7.7, "--", "cap 7.7 (upper bound)")]:
    Rr = NN*pm.V*Pc*pm.MC2/(GG*pm.u_w(pm.B_THR)*pm.V); c_ = ax.contour(NN, GG, Rr, levels=[0.3], colors="k", linewidths=1.6, linestyles=ls_); c_.collections[0].set_label(lab) if hasattr(c_, "collections") else None
ax.plot([], [], "k:", label="ratio = 0.3, cap 1.1"); ax.plot([], [], "k-", label="ratio = 0.3, cap 5.0"); ax.plot([], [], "k--", label="ratio = 0.3, cap 7.7")
ax.axvline(4e15, color="g", ls="--"); ax.text(4.4e15, 3.3e4, "all I_p=0.9 MA\nin the zone", fontsize=7, color="g")
ax.axhspan(9e4, 6e5, facecolor="none", edgecolor="k", hatch="//", alpha=0.3); ax.text(1.3e11, 7.5e5, "Step-2 16-84% band of Gamma_th", fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n_res (resonant zone density) [m^-3]"); ax.set_ylabel("Gamma_th [1/s]"); ax.legend(fontsize=7, loc="upper left")
ax.set_title("Pinning needs the absorption cap to reach ~0.3 of the thermal threshold power", fontsize=9)
fig.tight_layout(); fig.savefig("../fig3_feasibility.png", dpi=130); plt.close(fig)
print("ok")
