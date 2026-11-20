"""Draw the README figures from the data in results/ (no reruns). Usage: python scripts/make_figures.py"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito palette
BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#555555", "#56B4E9"
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, GREEN, RED, PURPLE, SKY]),
    "figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight", "lines.linewidth": 1.8,
})
DW = r"$\Gamma_\phi/\Delta\omega$"


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def baseline():
    d = json.load(open(R / "P1" / "summary_sde.json"))
    k = sorted(d, key=float)
    x = [float(a) for a in k]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, [d[a]["mode"] for a in k], "o-", label="mode of the tail")
    ax.plot(x, [d[a]["mean"] for a in k], "s-", label="mean")
    ax.axhline(13.13, color=GREY, ls="--", lw=1.2)
    ax.text(0.02, 13.4, "noise-free ODE fixed point, 13.1", color=GREY, fontsize=9)
    ax.axhspan(16.8, 17.1, color=GREEN, alpha=0.3, lw=0)
    ax.text(0.02, 17.5, "DREAM mode, 16.8 to 17.1", color=GREEN, fontsize=9)
    ax.set_xlabel(r"pitch-noise strength $\kappa$ (1 = Ito-consistent)")
    ax.set_ylabel(r"tail energy $\gamma$")
    ax.legend(loc="center right")
    save(fig, "baseline_tail.png")


def threshold():
    d = json.load(open(R / "P3" / "data" / "numbers.json"))
    s = d["delta_scaling"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
    D, bt = np.array(s["Delta"]), np.array(s["b_thr"])
    a.loglog(D, bt, "o", color=BLUE, ms=7)
    a.loglog(D, bt[1] * (D / D[1]) ** s["slope"], "--", color=GREY, label=rf"$\propto\Delta^{{{s['slope']:.2f}}}$")
    a.set_xticks(D)
    a.set_xticklabels([f"{v:g}" for v in D])
    a.minorticks_off()
    a.set_xlabel(r"line spacing $\Delta k_\parallel$ (m$^{-1}$)")
    a.set_ylabel(r"threshold amplitude $b_{\rm thr}$ per line")
    a.legend()
    names = ["eqN2", "eqN3", "eqN5", "eqN10", "gauss34", "gauss45", "gauss70"]
    labels = ["N=2", "N=3", "N=5", "N=10", "Gauss 3.4", "Gauss 4.5", "Gauss 7"]
    med = [np.median(d[n]["sig_c50"]) for n in names]
    lo = [med[i] - min(d[n]["sig_c50"]) for i, n in enumerate(names)]
    hi = [max(d[n]["sig_c50"]) - med[i] for i, n in enumerate(names)]
    b.errorbar(range(len(names)), med, yerr=[lo, hi], fmt="o", color=ORANGE, capsize=3, ms=7)
    b.axhline(1, color=GREY, ls=":")
    b.set_xticks(range(len(names)))
    b.set_xticklabels(labels, rotation=35, ha="right")
    b.set_ylabel(r"Chirikov parameter $\sigma$ at 50% jump")
    b.set_ylim(0.6, 1.5)
    save(fig, "overlap_threshold.png")


def bounded_chaos():
    z = {}
    for f in glob.glob(str(R / "step3" / "data" / "map_s*_g*_seed*.npz")):
        a = np.load(f)
        z.setdefault((float(a["sigma_nom"]), float(a["Gphi_over_D"])), []).append(a)
    gs = [0.0, 0.01, 0.03, 0.1, 0.3]
    cols = [RED, ORANGE, GREEN, BLUE, PURPLE]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, sg in zip(axs, (1.0, 3.0)):
        for g, c in zip(gs, cols):
            zs = z[(sg, g)]
            n = min(len(a["ts"]) for a in zs)
            m = np.mean([a["m2"][:n] / a["Lw"] ** 2 * 6 for a in zs], 0)
            ax.loglog(zs[0]["beats"][:n], np.maximum(m, 1e-4), color=c, label=f"{g:g}")
        ax.axhline(1, color=GREY, ls=":", lw=1.2)
        ax.set_xlim(0.3, 1000)
        ax.set_xlabel("time (beat periods)")
        ax.text(0.04, 0.93, rf"$\sigma$ = {sg:g}", transform=ax.transAxes)
    axs[0].set_ylim(1e-3, 30)
    axs[0].set_ylabel(r"$\langle\Delta\gamma^2\rangle/(L_w^2/6)$")
    axs[1].legend(title=DW, loc="lower right")
    save(fig, "bounded_chaos.png")


def sustained():
    d = np.load(R / "step3" / "data" / "map_map_summary.npz")
    sig, gs = d["sig"], d["Gs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.5))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(sig)))
    for i, (s, c) in enumerate(zip(sig, cols)):
        ok = (d["fill"][i] < 2) & (gs > 0)
        a.errorbar(gs[ok], d["sl"][i][ok], d["sl_e"][i][ok], color=c, marker="o", ms=4, capsize=2, label=f"{s:g}")
        b.errorbar(np.where(gs > 0, gs, 3e-3), d["fill"][i], d["fill_e"][i], color=c, marker="o", ms=4, capsize=2)
    x = np.array([0.01, 0.03, 0.1, 0.3])
    a.plot(x, np.minimum(7 * x, 1), "--", color=GREY, label=r"min(1, 7$\Gamma_\phi/\Delta\omega$)")
    a.set(xscale="log", yscale="log", xlabel=DW, ylabel="late growth rate / quasilinear")
    a.legend(title=r"$\sigma$", ncol=2, fontsize=8)
    b.axhline(1, color=GREY, ls=":")
    b.set(xscale="log", yscale="log", xlabel=DW + "  (leftmost point: coherent)", ylabel=r"fill at 1000 periods")
    save(fig, "sustained_scattering.png")


def diiid():
    s = np.load(R / "step2" / "data" / "mc_samples.npz")
    thermal = s["coll"] + s["land"] + s["ttmp"]
    mu = 4 * 4e-7 * np.pi
    bt = s["bthr"] * np.sqrt(s["Neff"])
    p = (thermal + s["edge_non"]) * s["Vp"] * bt ** 2 * s["B"] ** 2 / mu / 1e3
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(np.log10(p), bins=32, color=SKY, edgecolor="white")
    for v, t in ((43, "43"), (76, "76"), (150, "150")):
        ax.axvline(np.log10(v), color=RED, lw=1.5)
        ax.text(np.log10(v), ax.get_ylim()[1] * 0.97, f"{t} kW", color=RED, ha="right", va="top", rotation=90, fontsize=9)
    ax.axvline(np.log10(np.median(p)), color=BLUE, ls="--")
    ax.text(np.log10(np.median(p)) + 0.03, ax.get_ylim()[1] * 0.97, f"median {np.median(p):.0f} kW", color=BLUE, va="top", fontsize=9)
    ax.set_xlabel(r"power to reach the overlap threshold (kW, $\log_{10}$ axis)")
    ax.set_ylabel("samples")
    ticks = [1, 2, 3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{t}$" for t in ticks])
    save(fig, "diiid_threshold_power.png")


def pinning():
    S = {}
    for f in glob.glob(str(R / "pinning" / "data" / "pin_s*_b*_lam*.json")):
        o = json.load(open(f))
        if o["bpk"] < 1e-8 or "dt0125" in f:
            continue
        S.setdefault(o["lam"], []).append((o["bpk"], o["P_wave"] / o["lam"]))
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for (lam, pts), m in zip(sorted(S.items(), reverse=True), "osv"):
        pts = np.array(sorted(pts))
        ax.loglog(pts[:, 0], pts[:, 1], m + "-", label=rf"time compression $\lambda$ = {lam:g}")
    ax.axhspan(1.1, 7.7, color=RED, alpha=0.15, lw=0)
    ax.text(1.2e-5, 5.5, "replenishment limit", color=RED, fontsize=9)
    ax.axvline(1e-4, color=GREY, ls="-.", lw=1.2)
    ax.text(1.05e-4, 2e-1, r"$b_{\rm thr}$", color=GREY)
    ax.set_xlabel(r"wave amplitude per line $b$")
    ax.set_ylabel(r"wave work per electron ($mc^2$/s)")
    ax.legend(loc="lower right")
    save(fig, "pinning_refuted.png")


if __name__ == "__main__":
    for fn in (baseline, threshold, bounded_chaos, sustained, diiid, pinning):
        fn()
