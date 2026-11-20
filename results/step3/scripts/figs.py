import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, stats, dd, fd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M = load("map"); Gs = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
# fig2: <dg^2>(t)/(Lw^2/6) versus beat periods, seed-averaged, for four sigma
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
for a, sg in zip(ax, (0.5, 1.0, 1.5, 3.0)):
    for k, G in enumerate(Gs):
        zs = M[(sg, G)]; n = min(len(z["ts"]) for z in zs)
        m = np.mean([z["m2"][:n]/z["Lw"]**2*6 for z in zs], 0); b = zs[0]["beats"][:n]
        a.loglog(b, np.maximum(m, 1e-4), label=f"{G:g}", color=plt.cm.viridis(k/6))
    a.axhline(1, c="k", ls=":"); a.set_title(f"$\\sigma$ = {sg:g}"); a.set_xlabel("beat periods"); a.set_xlim(0.3, 1000)
ax[0].set_ylabel(r"$\langle\Delta\gamma^2\rangle/(L_w^2/6)$"); ax[0].legend(title=r"$\Gamma_\phi/\Delta\omega$", fontsize=7); ax[0].set_ylim(1e-4, 100)
plt.tight_layout(); plt.savefig(f"{fd}/fig2_m2_vs_time.png", dpi=130)
# fig3: sustained rate vs Gamma (valid cells), with fill
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, sg in enumerate((0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
    S = [stats(M[(sg, G)]) for G in Gs[1:5]]; x = Gs[1:5]
    ax[0].errorbar(x, [s["sl"] for s in S], [s["sl_e"] for s in S], marker="o", ms=4, label=f"$\\sigma$={sg:g}", capsize=2)
    ax[1].errorbar(Gs[:5], [stats(M[(sg, G)])["fill"] for G in Gs[:5]], [stats(M[(sg, G)])["fill_e"] for G in Gs[:5]], marker="o", ms=4, capsize=2)
xx = np.array([0.01, 0.03, 0.1, 0.3]); ax[0].plot(xx, np.minimum(7*xx, 1), "k--", lw=1, label="min(1, 7 $\\Gamma/\\Delta\\omega$)")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\Gamma_\phi/\Delta\omega$"); ax[0].set_ylabel(r"late growth rate of $\langle\Delta\gamma^2\rangle$ / QL"); ax[0].legend(fontsize=7)
ax[1].axhline(1, c="k", ls=":"); ax[1].set_xscale("symlog", linthresh=0.01); ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\Gamma_\phi/\Delta\omega$"); ax[1].set_ylabel(r"fill $\langle\Delta\gamma^2\rangle(T)/(L_w^2/6)$, $T$=1000 periods")
plt.tight_layout(); plt.savefig(f"{fd}/fig3_sustained_rate.png", dpi=130)
