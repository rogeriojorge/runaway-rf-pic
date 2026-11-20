"""All P2 figures from data/*.pkl|json.  Run: python3 make_figures.py [which...]"""
import os, sys, json, pickle
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp import OUT, FIG
import bw, theory as th, plasma as pl
import analysisB

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
case = bw.Case()
MAP = json.load(open(os.path.join(OUT, "mapping.json")))
Xi = MAP["table"]["0.2"]["Xi"]; A_sweep = MAP["table"]["0.2"]["A_sweep"]
b_unitR = A_sweep/Xi              # b_phys at R=1 for the DIII-D sweep (E=8.5Ec, friction, RR), theta=0.2

def load_rows(name):
    try:
        return analysisB.analyse(name)
    except Exception as e:
        print("missing", name, e); return []

def fit_hR(R, h, w=None):
    """h = h0/(1+(R/Rc)^p); crude log-space least squares."""
    from scipy.optimize import least_squares
    f = lambda q: np.log(q[0]/(1+(R/q[1])**q[2])) - np.log(h)
    r = least_squares(f, [0.9, 2.0, 1.0], bounds=([0.2, 0.05, 0.3], [1.5, 100, 3]))
    return r.x

def fig_main():
    rows1000 = load_rows("lam1000"); rows30 = load_rows("lam30"); rowsc = load_rows("lam100"); rows1 = load_rows("lam10")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    def plot(rows, **kw):
        if not rows: return
        R = np.array([r["R"] for r in rows]); h = np.array([r["ratio"] for r in rows]); se = np.array([r["se_ratio"] for r in rows])
        ax.errorbar(R*b_unitR, h, yerr=se, **kw)
    plot(rows1000, fmt="o", color="C3", ms=5, label=r"$\lambda=10^3$ (time-compressed drivers)")
    plot(rows30, fmt="s", color="C0", ms=5, mfc="none", label=r"$\lambda=30$")
    plot(rows1, fmt="D", color="k", ms=5, mfc="none", label=r"$\lambda=10$")
    plot(rowsc, fmt="^", color="C2", ms=5, label=r"$\lambda=100$")
    allr = rows30+rowsc+rows1
    if allr:
        R = np.array([r["R"] for r in allr]); h = np.array([r["ratio"] for r in allr])
        q = fit_hR(R, h); print("fit h0, Rc, p =", q)
        Rg = np.logspace(-2.5, 4.5, 200); ax.plot(Rg*b_unitR, q[0]/(1+(Rg/q[1])**q[2]), "k-", lw=1, label=rf"fit $h_0/(1+(R/R_c)^p)$: $R_c$={q[1]:.1f}, p={q[2]:.2f}")
        json.dump(dict(h0=q[0], Rc=q[1], p=q[2]), open(os.path.join(OUT, "hR_fit.json"), "w"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 3); ax.set_xlim(1e-10, 1e-2)
    ax.set_xlabel(r"wave amplitude $b=\delta B/B_0$ (DIII-D mapping, $E=8.5E_c$, $\theta=0.2$)")
    ax.set_ylabel(r"$D_\mathrm{eff}/D_\mathrm{QL}$ per resonance passage")
    # regions and markers
    ax.axvspan(3e-5, 2e-3, color="C1", alpha=0.12, label="DIII-D estimate (150 kW)")
    ax.axvspan(1e-5, 1e-4, color="C4", alpha=0.12, label=r"Guo 2018 ($10^{-5}$-$10^{-4}$)")
    ax.axvline(4.75e-4, color="C5", ls="--", label="Kang 2026 trapping $4.75\\times10^{-4}$")
    ax.axvline(2*b_unitR, color="C3", ls=":", label=r"$b_\times$ (sweep, $R_c\approx2$)")
    sec = ax.secondary_xaxis("top", functions=(lambda b: b/b_unitR, lambda R: R*b_unitR)); sec.set_xlabel(r"$R=\omega_b^2/|\dot\delta|$")
    ax.legend(fontsize=7.5, loc="lower left"); ax.set_title("Single coherent whistler line, DIII-D resonance (476 MHz, k$_\\parallel$=41/m, 7.9 MeV)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_Deff_over_DQL.png")); plt.close(fig)


def fig_A():
    d = pickle.load(open(os.path.join(OUT, "expA.pkl"), "rb"))["out"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    cm = plt.cm.viridis(np.linspace(0, 1, len(d)))
    summary = []
    for c, r in zip(cm, d):
        x = r["ts"]*r["omega_b"]
        ratio = r["meas"]/np.maximum(r["lin"], 1e-300)
        axs[0].plot(x, ratio, color=c, lw=1)
        DQLp = r["lin"][-1]/(2*r["ts"][-1])            # golden-rule plateau: lin(t)/2t at X t >> 1
        r["DQL"] = DQLp
        Deff = r["meas"]/(2*r["ts"])
        axs[1].plot(x, Deff/r["DQL"], color=c, lw=1, label=f"b={r['b']:.1e}" if r is d[0] or r is d[-1] or r is d[7] else None)
        i3 = np.argmin(abs(x-3)); i03 = np.argmin(abs(x-0.3)); i30 = np.argmin(abs(x-30))
        summary.append(dict(b=r["b"], wb=r["omega_b"], short=float(Deff[i03]/r["DQL"]), mid=float(Deff[i3]/r["DQL"]), long=float(Deff[i30]/r["DQL"]),
                            kurt=r["kurt"]))
    for a in axs[:2]:
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$\omega_b t$")
    axs[0].set_ylabel(r"$\langle\Delta\xi^2\rangle_\mathrm{sim}/\langle\Delta\xi^2\rangle_\mathrm{linear}$"); axs[0].set_ylim(1e-3, 5)
    axs[1].set_ylabel(r"$D_\mathrm{eff}(t)/D_\mathrm{QL}$, $D_{\xi\xi}=\langle\Delta\xi^2\rangle/2t$"); axs[1].set_ylim(1e-3, 5); axs[1].legend(fontsize=8)
    axs[0].set_title("15 amplitudes $10^{-6}$-$10^{-2}$ collapse onto one curve"); 
    b = np.array([s["b"] for s in summary])
    for k, lab in (("short", r"$\omega_b t=0.3$"), ("mid", r"$\omega_b t=3$"), ("long", r"$\omega_b t=30$")):
        axs[2].plot(b, [s[k] for s in summary], "o-", ms=4, label=lab)
    axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel("b"); axs[2].set_ylabel(r"$D_\mathrm{eff}/D_\mathrm{QL}$"); axs[2].legend(); axs[2].set_title("collisionless, no decorrelation: ratio set by $\\omega_b t$ only")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_collisionless_time_resolved.png")); plt.close(fig)
    json.dump(summary, open(os.path.join(OUT, "expA_summary.json"), "w"), indent=1)
    for s in summary: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "kurt"}, s["kurt"])


def fig_D():
    d = pickle.load(open(os.path.join(OUT, "expD.pkl"), "rb"))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    tab = []
    for key, m, lab in (("scan_dk", "o", r"$b=10^{-4}$, scan line spacing $\Delta k_\parallel$"), ("scan_b", "s", r"$\Delta k_\parallel=0.44$/m $(=1/R)$, scan $b$")):
        sig = np.array([r["sigma"] for r in d[key]]); Sp = np.array([r["S_end"]*r["L"]**2/(r["L"]+2*r["island_hw"])**2 for r in d[key]])
        S = np.array([r["S_end"] for r in d[key]])
        axs[0].plot(sig, Sp, m+"-", label=lab)
        for r in d[key]:
            wb = r["wb"][0]; ts = r["ts"]*wb; Sg = r["S"]*r["L"]**2/6/r["island_hw"]**2
            tab.append(dict(scan=key, b=r["b"], dk=r["dk"], sigma=r["sigma"], S_end=r["S_end"], Sprime=r["S_end"]*r["L"]**2/(r["L"]+2*r["island_hw"])**2,
                            L_over_hw=r["L"]/r["island_hw"], dg2_over_hw2_end=float(np.mean(Sg[-10:]))))
    axs[0].axvline(1.0, color="k", ls="--"); axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel(r"overlap parameter $\sigma=2(\omega_{b1}+\omega_{b2})/\Delta$"); axs[0].set_ylabel(r"$\langle\Delta\gamma^2\rangle/[(L+2w)^2/6]$ at $t=300/\omega_b$")
    axs[0].legend(fontsize=8); axs[0].set_title("two lines: both scans collapse on $\\sigma$; dashed = Chirikov $\\sigma=1$")
    for r in d["scan_b"]:
        axs[1].plot(r["ts"]*r["wb"][0], r["S"]*r["L"]**2/6/r["island_hw"]**2, label=f"$\\sigma$={r['sigma']:.2f}" if r["sigma"] > 0.3 else None)
    axs[1].set_xlabel(r"$\omega_b t$"); axs[1].set_ylabel(r"$\langle\Delta\gamma^2\rangle/w^2$ ($w$: island half-width)"); axs[1].set_yscale("log"); axs[1].legend(fontsize=7, ncol=2); axs[1].set_title("$\\Delta k=0.44$/m scan in $b$")
    b = np.array([r["b"] for r in d["scan_b"]]); sg = np.array([r["sigma"] for r in d["scan_b"]]); Sp = np.array([r["S_end"] for r in d["scan_b"]])
    axs[2].plot(b, Sp, "o-"); axs[2].axvspan(3e-5, 2e-3, color="C1", alpha=0.12, label="DIII-D estimate"); axs[2].axvline(4.75e-4, color="C5", ls="--", label="Kang 4.75e-4")
    axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel("b (each line)"); axs[2].set_ylabel(r"$S=\langle\Delta\gamma^2\rangle/(L^2/6)$"); axs[2].legend(); axs[2].set_title("toroidal-mode spacing $\\Delta k=1/R_0$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_two_line_overlap.png")); plt.close(fig)
    json.dump(tab, open(os.path.join(OUT, "expD_summary.json"), "w"), indent=1)

def fig_lowth():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, c, lab in (("lam1000", "C3", r"$\theta\in[0.1,0.3]$, $\lambda=10^3$"), ("lowth", "C0", r"$\theta\in[0.02,0.08]$, $\lambda=10^3$"), ("lowth100", "C2", r"$\theta\in[0.02,0.08]$, $\lambda=100$")):
        rows = load_rows(name)
        if not rows: continue
        R = np.array([r["R"] for r in rows]); b = np.array([r["bphys"] for r in rows])
        ax.errorbar(R, [r["ratio"] for r in rows], yerr=[r["se_ratio"] for r in rows], fmt="o-", color=c, ms=4, label=lab+r" ($\xi$ metric)")
        ax.plot(R, [r["ratio_s"] for r in rows], "x:", color=c, label=lab+r" ($\theta^2$ gain)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 5); ax.set_xlabel(r"$R=\omega_b^2/|\dot\delta|$ (label at $\theta=0.2$)"); ax.set_ylabel(r"$D_\mathrm{eff}/D_\mathrm{QL}$"); ax.legend(fontsize=8)
    ax.set_title(r"$\lambda=10^3$: pitch of the resonant runaways ($\xi\approx-1$ means small $\theta$)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_small_pitch.png")); plt.close(fig)


def fig_kicks():
    d = pickle.load(open(os.path.join(OUT, "expB_lam1000.pkl"), "rb"))["out"]
    rows = analysisB.analyse("lam1000"); bs = sorted(d)
    Rs = [r["R"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3))
    pick = [0, 4, 7, 10, 13]
    for i, ip in enumerate(pick):
        b = bs[ip]; r = d[b]; dxi = r["xi_t"][-1]-r["xic"]
        z = (dxi-dxi.mean())/dxi.std()
        axs[0].hist(z, bins=np.linspace(-4, 4, 41), histtype="step", density=True, label=f"R={Rs[ip]:.2g}, kurt={rows[ip]['kurt']:.1f}")
    x = np.linspace(-4, 4, 200); axs[0].plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), "k--", label="Gaussian")
    axs[0].set_xlabel(r"$(\Delta\xi-\langle\Delta\xi\rangle)/\sigma$"); axs[0].set_ylabel("pdf"); axs[0].legend(fontsize=8); axs[0].set_title("kick distribution per resonance passage")
    axs[1].semilogx(Rs, [r["kurt"] for r in rows], "o-"); axs[1].axhline(3, color="k", ls="--"); axs[1].axhline(1.5, color="gray", ls=":")
    axs[1].set_xlabel("R"); axs[1].set_ylabel("kurtosis of $\\Delta\\xi$"); axs[1].set_title("3 = Gaussian, 1.5 = sinusoidal phase kick")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_kick_distribution.png")); plt.close(fig)

def fig_valid():
    v2 = json.load(open(os.path.join(OUT, "val2_resonance.json"))); v4 = json.load(open(os.path.join(OUT, "val4_quasilinear.json")))
    v1 = json.load(open(os.path.join(OUT, "val1_integrator.json"))); vj = json.load(open(os.path.join(OUT, "val1b_jic.json")))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    p = np.array(v2["V2_points_dt32"])
    axs[0].plot(p[:, 1]*case.Om*1e-6, p[:, 2], "o", label="Boris orbits"); axs[0].plot(p[:, 1]*case.Om*1e-6, p[:, 3], "-", label="linear theory (sinc$^2$ lobe)")
    axs[0].set_xlabel(r"detuning $\delta=\omega-k_\parallel v_\parallel-\Omega/\gamma$ [$10^6$/s]"); axs[0].set_ylabel(r"$\langle\Delta u_\perp^2\rangle$"); axs[0].legend(fontsize=8); axs[0].set_title("resonance location and kick amplitude")
    bb = [float(k) for k in v2["V3"]]; 
    axs[1].loglog(bb, [v2["V3"][k]["omega_b_theory"]*case.Om for k in v2["V3"]], "-", label=r"analytic $\omega_b$")
    axs[1].loglog(bb, [v2["V3"][k]["f_small_amp"]*case.Om for k in v2["V3"]], "o", label="orbits (small-amplitude oscillation)")
    axs[1].set_xlabel("b"); axs[1].set_ylabel(r"$\omega_b$ [1/s]"); axs[1].legend(fontsize=8); axs[1].set_title(r"island bounce frequency ($\propto\sqrt{b}$)")
    ts = np.array(v4["ts"])*1e9
    axs[2].plot(ts, v4["meas"], "o", label="orbits, 129 random-phase lines"); axs[2].plot(ts, v4["pred"], "-", label="quasilinear (random-phase)")
    axs[2].set_xlabel("t [ns]"); axs[2].set_ylabel(r"$\langle\Delta u_\perp^2\rangle$"); axs[2].legend(fontsize=8); axs[2].set_title(f"broadband QL: ratio {v4['ratio_mean']:.3f}$\\pm${v4['ratio_se']:.3f}")
    dts = [32, 64, 128]; axs[3].loglog(dts, [vj[f"bw_T/{k}_vs_bw_T/256"] for k in dts], "o-", label="this integrator (tan-exact)")
    axs[3].loglog(dts, [vj[f"jic_T/{k}_vs_bw_T/256"] for k in dts], "s-", label="JAX-in-Cell boris_relativistic")
    axs[3].set_xlabel("steps per gyro-period"); axs[3].set_ylabel("max rel. diff. of $u$ vs T/256 reference"); axs[3].legend(fontsize=8); axs[3].set_title("Boris convergence, same fields")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig0_validation.png")); plt.close(fig)

if __name__ == "__main__":
    which = sys.argv[1:] or ["main"]
    for w in which:
        globals()["fig_"+w]()
