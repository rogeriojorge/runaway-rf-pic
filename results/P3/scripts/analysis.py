"""All P3 figures + the numbers table from data/*.pkl.  python3 analysis.py"""
import os, sys, pickle, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"); FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
def ld(n):
    p = os.path.join(D, n)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None
BU = 1.008182e-4        # b at sigma_nominal = 1 for Delta = 0.57/m (b_unit)
def cross(sig, rho, err, level):
    """first upward crossing of rho = level, log-interpolated in sigma; returns (central, lo, hi) using rho -/+ err."""
    out = []
    for r in (rho, rho-err, rho+err):
        r = np.asarray(r); x = None
        for i in range(1, len(r)):
            if r[i-1] < level <= r[i]:
                f = (level-r[i-1])/(r[i]-r[i-1]); x = np.exp(np.log(sig[i-1])+f*(np.log(sig[i])-np.log(sig[i-1]))); break
        out.append(x)
    return out
SPEC = [("eqN2", "N=2", "C0"), ("eqN3", "N=3", "C1"), ("eqN5", "N=5", "C2"), ("eqN10", "N=10", "C3"), ("gauss34", "Gauss $\\sigma_j$=3.4 (uniform 40° aperture)", "C4"), ("gauss45", "Gauss $\\sigma_j$=4.5 (GENRAY, nominal)", "C5"), ("gauss70", "Gauss $\\sigma_j$=7 (half aperture)", "C6")]
NUM = {}
def fig_main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for key, lab, col in SPEC:
        d = ld(f"Sr2_{key}_f0.pkl")
        if d is None: continue
        r = list(d["res"])
        if key == "gauss45" and ld("Sr2_gauss45_hi.pkl"): r = r + list(ld("Sr2_gauss45_hi.pkl")["res"])
        s = np.array([x["sigma"] for x in r]); rho = np.array([x["rho"] for x in r]); er = np.array([x["rho_err"] for x in r])
        ax[0].errorbar(s, np.clip(rho, 1e-3, None), er, marker="o", ms=3.5, color=col, lw=1, label=lab)
        c5 = cross(s, rho, er, 0.5); c1 = cross(s, rho, er, 0.1)
        bI = lambda sc: float(np.exp(np.interp(np.log(sc), np.log(s), np.log([x["b"] for x in r])))) if sc else None
        NUM[key] = dict(sig_c50=c5, sig_c10=c1, Delta=d["Delta"], bunit=d["bunit"], b_thr50=bI(c5[0]), b_thr10=bI(c1[0]))
        ax[1].plot(np.array([x["b"] for x in r]), np.clip(rho, 1e-3, None), marker="o", ms=3, color=col, lw=1)
    for key, col in (("eqN5", "C2"), ("eqN10", "C3"), ("gauss45", "C5")):
        d = ld(f"S_{key}_f0.pkl")
        if d: ax[0].plot([x["sigma"] for x in d["res"]], np.clip([x["rho"] for x in d["res"]], 1e-3, None), "x", color=col, ms=6, mew=1.3, alpha=0.8)
    ax[0].plot([], [], "kx", label="full-orbit Boris (same ICs/metric)")
    ax[0].set_yscale("log"); ax[0].set_ylim(2e-3, 4); ax[0].set_xlim(0.4, 7.5); ax[0].set_xscale('log'); ax[0].axvspan(0.55, 7.0, color='orange', alpha=0.10)
    ax[0].axhline(1, color="k", lw=0.6); ax[0].axvline(1, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel(r"overlap parameter $\sigma = 2(\omega_{b,j}+\omega_{b,j+1})/\Delta\delta$"); ax[0].set_ylabel(r"$D_{\rm eff}/D_{\rm QL}$ (stationary, onset window)")
    ax[0].legend(fontsize=7.5, loc="lower right"); ax[0].set_title("Collapse on the overlap parameter (collisionless, $\\Delta=0.57$/m)")
    ax[1].axvspan(3e-5, 5e-3, color="orange", alpha=0.15, label="DIII-D amplitude range"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(2e-3, 4)
    ax[1].set_xlabel("peak line amplitude $b=\\delta B/B_0$"); ax[1].set_ylabel(r"$D_{\rm eff}/D_{\rm QL}$"); ax[1].legend(fontsize=8, loc="lower right"); ax[1].set_title("same data against b ($\\Delta = 1/R$, $R=1.75$ m)")
    ax[1].set_xlim(2e-5, 6e-3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_overlap_map.png")); plt.close()
def fig_delta():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4)); Ds = []; bt = []
    for key, col in (("eqN10_D0285", "C0"), ("eqN10", "C1"), ("eqN10_D114", "C2"), ("eqN10_D228", "C3")):
        d = ld(f"Sr2_{key}_f0.pkl")
        if d is None: continue
        r = d["res"]; s = np.array([x["sigma"] for x in r]); rho = np.array([x["rho"] for x in r]); er = np.array([x["rho_err"] for x in r]); b = np.array([x["b"] for x in r])
        ax[0].plot(b, np.clip(rho, 1e-3, None), "o-", ms=3, color=col, label=f"$\\Delta$={d['Delta']:.3g}/m")
        c = cross(s, rho, er, 0.5)
        if c[0]:
            bI = lambda sc: float(np.exp(np.interp(np.log(sc), np.log(s), np.log(b))))
            Ds.append(d["Delta"]); bt.append((bI(c[0]), bI(c[1] or c[0]), bI(c[2] or c[0]))); NUM.setdefault("sigma_c_by_Delta", {})[str(d["Delta"])] = float(c[0])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(2e-3, 4); ax[0].axvspan(3e-5, 5e-3, color="orange", alpha=0.12)
    ax[0].set_xlabel("$b$"); ax[0].set_ylabel("$D_{\\rm eff}/D_{\\rm QL}$"); ax[0].legend(fontsize=8); ax[0].set_title("N=10 equal lines, four line spacings")
    if len(Ds) >= 2:
        Ds = np.array(Ds); bt = np.array(bt); o = np.argsort(Ds); Ds, bt = Ds[o], bt[o]
        ax[1].errorbar(Ds, bt[:, 0], [np.abs(bt[:, 0]-bt[:, 1]), np.abs(bt[:, 2]-bt[:, 0])], fmt="o", color="C3")
        p = np.polyfit(np.log(Ds), np.log(bt[:, 0]), 1); NUM["delta_scaling"] = dict(Delta=Ds.tolist(), b_thr=bt[:, 0].tolist(), slope=float(p[0]))
        xx = np.geomspace(Ds.min(), Ds.max(), 20); ax[1].plot(xx, np.exp(p[1])*xx**p[0], "k-", lw=1, label=f"fit slope {p[0]:.2f}"); ax[1].plot(xx, bt[1, 0]*(xx/Ds[1])**2, "r--", lw=1, label="$\\propto\\Delta^2$ (prediction)")
        ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("line spacing $\\Delta k_\\parallel$ (1/m)"); ax[1].set_ylabel("$b_{\\rm thr}$ ($D_{\\rm eff}/D_{\\rm QL}=0.5$)"); ax[1].legend(); ax[1].set_title("Threshold scaling")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_delta_scaling.png")); plt.close()
def BU_of(d): return d["bunit"]
def fig_window():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, col, ls in (("gauss45", "C5", "-"), ("eqN10", "C3", "--")):
        d = ld(f"Sr2_{key}_f0.pkl")
        if d is None: continue
        r = d["res"]; s = np.array([x["sigma"] for x in r])
        for lab, k, al in (("onset (10-100 beats)", "rho", 1.0), ("100 beats", "rho_100", 0.7), ("400 beats", "rho_400", 0.45)):
            ax[0].plot(s, np.clip([x[k] for x in r], 1e-3, None), ls, color=col, alpha=al, marker="o", ms=3, label=f"{key}: {lab}")
    ax[0].set_yscale("log"); ax[0].set_ylim(5e-3, 3); ax[0].set_xlabel("$\\sigma$"); ax[0].set_ylabel("$D_{\\rm eff}/D_{\\rm QL}$ (slope over the second half of the window)"); ax[0].legend(fontsize=7); ax[0].set_title("window dependence")
    ax[1].axis("off"); ax[1].text(0.02, 0.5, "Long-time (>= 1e3 beats) spreading with the reduced\nintegrator is NOT converged in dt (m2 at 1e4 beats for\nsigma=2.2: 0.009, 0.015, 0.043, 0.052 for dt*omega_b =\n0.1, 0.05, 0.025, 0.0125): numerical damping.\nOnly windows <= 400 beats (dt*omega_b=0.0125) are used.", fontsize=10, va="center")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_window_longtime.png")); plt.close()

def fig_coll():
    d = ld("X_coll_gauss45.pkl"); n = ld("X_noise_gauss45.pkl"); ni = ld("X_noisei_gauss45.pkl")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # analytic scales
    nuD = 0.0390; dth = 2.4e9            # nu_D (1/s) at gamma=16.4 and |d delta/d theta| (rad/s per rad)
    Gc = lambda f: (dth**2*nuD*f)**(1/3)
    for ia, (dat, lab, xl) in enumerate(((d, "collisions", "collision multiplier $f$ ($\\nu_D = f\\times$ DIII-D value)"), (n, "common-mode phase noise", "$\\Gamma_\\phi$ (1/s)"), (ni, "independent per-line phase noise", "$\\Gamma_\\phi$ (1/s)"))):
        if dat is None: continue
        sig = sorted(set(x["sigma_nom"] for x in dat)); lv = np.array(sorted(set(x["level"] for x in dat)))
        for i, sgm in enumerate(sig):
            y = [([x["rho"] for x in dat if x["sigma_nom"] == sgm and x["level"] == l] or [np.nan])[0] for l in lv]
            xs = np.where(lv > 0, lv, lv[lv > 0].min()/30)
            ax[ia].plot(xs, np.clip(y, 1e-3, None), "o-", ms=3, color=plt.cm.plasma(i/len(sig)*0.85), label=f"$\\sigma$={sgm}")
        ax[ia].set_xscale("log"); ax[ia].set_yscale("log"); ax[ia].set_ylim(5e-3, 3); ax[ia].set_xlabel(xl); ax[ia].set_ylabel("$D_{\\rm eff}/D_{\\rm QL}$"); ax[ia].set_title(lab+" (Gauss 4.5)")
        if ia == 0:
            ax[ia].axvline(1, color="k", ls="--", lw=1); ax[ia].text(1.3, 0.006, "DIII-D $f=1$", rotation=90, fontsize=8)
            ax2 = ax[ia].secondary_xaxis("top", functions=(lambda f: Gc(np.maximum(f, 1e-9))/1e6, lambda g: (g*1e6)**3/(dth**2*nuD))); ax2.set_xlabel("$\\Gamma_c=(\\partial_\\theta\\delta)^{2/3}\\nu_D^{1/3}$ (Mrad/s)")
        else:
            ax2 = ax[ia].twiny(); ax2.set_xscale("log"); ax2.set_xlim(ax[ia].get_xlim()); ax2.set_xticks([np.pi*3e6, np.pi*1e7]); ax2.set_xticklabels(["3 MHz FWHM", "10 MHz FWHM"], fontsize=7)
        ax[ia].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_collisions_linewidth.png")); plt.close()
def fig_validation():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, xi0 in enumerate((-0.98, 0.0)):
        p = os.path.join(D, f"val_coll_xi{xi0}.json")
        if os.path.exists(p):
            v = json.load(open(p)); t = np.array(v["nu_t"])
            ax[0].plot(t, v["m1"], "o", ms=4, color=f"C{i}", label=f"<xi>, xi0={xi0}"); ax[0].plot(t, v["th1"], "-", color=f"C{i}")
            ax[1].plot(t, v["m2"], "o", ms=4, color=f"C{i}", label=f"<xi^2>, xi0={xi0}"); ax[1].plot(t, v["th2"], "-", color=f"C{i}")
    ax[0].set_xlabel("$\\nu_D t$"); ax[0].set_ylabel("$\\langle\\xi\\rangle$"); ax[0].legend(); ax[0].set_title("Langevin pitch operator without wave\n(points) vs Lorentz theory (lines)")
    ax[1].set_xlabel("$\\nu_D t$"); ax[1].set_ylabel("$\\langle\\xi^2\\rangle$"); ax[1].legend()
    # reduced vs Boris
    for key, col in (("eqN5", "C2"), ("eqN10", "C3"), ("gauss45", "C5")):
        b = ld(f"S_{key}_f0.pkl"); r = ld(f"Sr2_{key}_f0.pkl")
        if b and r:
            ax[2].plot([x["rho"] for x in b["res"]], [x["rho"] for x in r["res"]], "o", color=col, ms=4, label=key)
    ax[2].plot([0, 2.6], [0, 2.6], "k-", lw=0.8); ax[2].set_xlabel("$D_{\\rm eff}/D_{\\rm QL}$ full-orbit Boris"); ax[2].set_ylabel("reduced resonant model"); ax[2].legend(); ax[2].set_title("reduced model vs full orbit (same grid)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig0_validation.png")); plt.close()

def fig_power():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import power as pw
    G = np.geomspace(1e3, 3e7, 200)
    cen = pw.P_damp(1.0e-4*np.sqrt(11.28), G)/1e3; lo = pw.P_damp(0.6e-4*np.sqrt(8.52), G)/1e3; hi = pw.P_damp(1.6e-4*np.sqrt(17.55), G)/1e3
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.fill_between(G, lo, hi, color="C0", alpha=0.25, label="predicted $P_{\\rm thr}$ (overlap threshold, band: $b_{\\rm thr}$ 0.6-1.6e-4, $N_{\\rm eff}$ 8.5-17.5)")
    ax.plot(G, cen, "C0-", lw=1.5)
    ax.axvspan(1.5e6, 1.9e7, color="C3", alpha=0.12); ax.text(2e6, 2e4, "single-pass wave\n(equivalent $\\Gamma_P=v_gA/V_p$)", fontsize=8, color="C3")
    ax.axvspan(4e3, 4e4, color="C2", alpha=0.12); ax.text(4.5e3, 2e4, "Spitzer-like\ncollisional loss\nonly", fontsize=8, color="C2")
    for P, lab, st, mk in ((43, "201951: 43 kW, small drop", "C1", "o"), (76, "201952: 76 kW, strong reduction", "C1", "s"), (70, "201953: 70 kW (300/300 ms)", "C1", "^"), (130, "ideal 201948: 130 kW, complete scattering", "k", "D"), (120, "ideal 201950: 120 kW, incomplete", "gray", "v")):
        ax.axhline(P, color=st, lw=0.8, ls="--", alpha=0.7); ax.plot([1.2e3], [P], mk, color=st, ms=6); ax.text(1.5e3, P*1.06, lab, fontsize=6.5, color=st)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.3, 5e4); ax.set_xlim(1e3, 3e7)
    ax.set_xlabel("total wave-energy loss rate $\\Gamma_P$ (1/s)  [$P = \\Gamma_P\\,u_w V_p$, $V_p$=18 m$^3$]"); ax.set_ylabel("coupled power (kW)")
    ax.legend(fontsize=7, loc="lower right"); ax.set_title("Threshold power from the overlap amplitude vs DIII-D shots")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_power_threshold.png")); plt.close()
    # numbers
    out = {}
    for lab, bb, ne in (("central", 1.0e-4, 11.28), ("low", 0.6e-4, 8.52), ("high", 1.6e-4, 17.55)):
        out[lab] = dict(b_tot=bb*np.sqrt(ne), **{f"G{g:g}_kW": pw.P_damp(bb*np.sqrt(ne), g)/1e3 for g in (1e3, 1e4, 1e5, 1e6, 1e7)},
                        single_kW=[pw.P_single(bb*np.sqrt(ne), 0.09, 1.0)/1e3, pw.P_single(bb*np.sqrt(ne), 0.28, 4.0)/1e3])
    # Gamma_P window consistent with 43-76 kW
    gl = [g for g in G if pw.P_damp(1.6e-4*np.sqrt(17.55), g)/1e3 >= 43][0]; gh = [g for g in G[::-1] if pw.P_damp(0.6e-4*np.sqrt(8.52), g)/1e3 <= 76][0]
    out["Gamma_window_43_76kW"] = [float(gl), float(gh)]
    cenG = lambda P: P*1e3/(pw.u_w(1.0e-4*np.sqrt(11.28))*pw.Vp)
    out["Gamma_central_43kW_76kW"] = [float(cenG(43)), float(cenG(76))]
    NUM["power"] = out

def ldP(N, lam):
    if N == 2 and lam == 1:
        d = (ld("Pr_eqN2_lam1_dt075_s07.pkl") or []) + (ld("Pr_eqN2_lam1_dt075_s14.pkl") or [])
        return sorted(d, key=lambda x: x["sigma"]) or None
    return ld(f"Pr_eqN{N}_lam{lam}_f0.pkl")
def fig_passage():
    fig, axs = plt.subplots(2, 3, figsize=(15, 8)); cols = {1000: "C0", 100: "C1", 10: "C2", 1: "C3"}
    for ia, N in enumerate((1, 2, 5)):
        Lc2 = (0.0577*N)**2/6
        for row in (0, 1):
            ax = axs[row, ia]
            for lam in (1000, 100, 10, 1):
                d = ldP(N, lam)
                if not d or "m2" not in d[0]: continue
                y = [np.sqrt(x["m2"]) if row == 0 else x["m2"]/Lc2 for x in d]
                ax.plot([x["sigma"] for x in d], y, "o-", color=cols[lam], ms=4, label=f"reduced, $\\lambda$={lam} (R={d[0]['R']:.0f}, Q={d[0]['Q']:.1e} at $\\sigma$={d[0]['sigma']})")
            db = ld(f"P_eqN{N}_lam1000.pkl")
            if db: ax.plot([x["sigma"] for x in db], [np.sqrt(x["m2"]) if row == 0 else x["m2"]/Lc2 for x in db], "ks", ms=6, label="full-orbit Boris, $\\lambda$=1000")
            ax.set_yscale("log"); ax.set_xlabel("overlap $\\sigma$")
            ax.set_ylabel("rms $\\Delta\\gamma$ per passage" if row == 0 else "$\\langle\\Delta\\gamma^2\\rangle/(L_c^2/6)$  (band-fill fraction)")
            ax.set_title(f"N={N} equal lines swept by E, friction, RR")
            if row == 0: ax.legend(fontsize=6.5)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig6_passage_sweep.png")); plt.close()

def fig_fill():
    d = ld("Sfill.pkl")
    if not d: return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, (n, r) in enumerate(d.items()):
        ax.errorbar([x[0] for x in r], np.clip([x[1] for x in r], 2e-3, None), [x[2] for x in r], marker="o", ms=3.5, label=n.replace("eqN", "N="))
    ax.axvline(1, color="k", ls=":", lw=0.8); ax.set_yscale("log"); ax.set_xlabel("$\\sigma$"); ax.set_ylabel("band-fill fraction $\\langle\\Delta\\gamma^2\\rangle/(L_c^2/6)$ at $t=300/\\omega_b$"); ax.legend(); ax.set_title("N equal lines: onset of global spreading (P2 convention)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1b_band_fill.png")); plt.close()

if __name__ == "__main__":
    for f in (fig_main, fig_delta, fig_window, fig_coll, fig_validation, fig_power, fig_passage, fig_fill):
        try: f()
        except Exception as e: import traceback; traceback.print_exc()
    json.dump(NUM, open(os.path.join(D, "numbers.json"), "w"), indent=1, default=float)
    print(json.dumps(NUM, indent=1, default=float))
