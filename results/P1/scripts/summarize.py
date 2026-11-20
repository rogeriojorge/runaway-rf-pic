"""Metrics tables + figures for P1. Run from this directory with the env-L2 venv."""
import json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import metrics
D = "../data/"; O = "../"; DR = "../../codes/DREAM/data/"
def win(n, w, **k):
    if not os.path.exists(D+n+".npz"): return None
    return metrics.series(D+n+".npz", window=w, smooth=4, **k)
W = (50, 80)
rows = {}
for n in ["full_g40", "full_fp", "dt2p5e-4", "dt4e-3", "n32768", "n2048", "nb32768", "wspread1", "seed1", "seed2", "noion_fp", "drag_fp"]:
    r = win(n, W)
    if r is None: continue
    p, b = r["pooled"], r["blocks"]
    rows[n] = dict(mean=p["mean"], median=p["median"], p90=p["p90"], mode=p["mode"], theta_mode=p["theta_mode"],
                   mode_blk=[x["mode"] for x in b], mean_blk=[x["mean"] for x in b])
    print(n, {k: (round(v, 3) if not isinstance(v, list) else [round(y, 2) for y in v]) for k, v in rows[n].items()})
json.dump(rows, open(O+"summary.json", "w"), indent=1)
# figure 1: distributions and time series
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
d = np.load(DR+"b_fp_L2.npz"); pg = d["pgrid"]; Fg = d["Fg"][-1]; gam = np.sqrt(1+pg**2)
ax[0].plot(gam, Fg/np.trapezoid(Fg, gam), "k", label="DREAM kinetic, pmax 40 (t=100 s)")
for n, c_ in [("full_g40", "C0"), ("full_fp", "C1")]:
    E = np.load(D+n+".npz")["eta"][-15:].reshape(-1, 3).astype(float); g = np.sqrt(1+(E**2).sum(1))
    h, e = np.histogram(g, bins=np.linspace(1, 80, 160), density=True); ax[0].step(.5*(e[1:]+e[:-1]), h, c_, label="JAX-in-Cell "+n)
ax[0].axvline(13.134, color="gray", ls=":", label="test-particle ODE 13.13"); ax[0].set_xlim(0, 70)
ax[0].set_xlabel("gamma"); ax[0].set_ylabel("dN/dgamma (norm.)"); ax[0].legend(fontsize=7); ax[0].set_title("Stationary tail, B=2 T, E=5 mV/m")
for n, c_ in [("full_g40", "C0"), ("full_fp", "C1")]:
    r = metrics.series(D+n+".npz", smooth=4); ax[1].plot(r["t"], r["mean"], c_, label=n+" mean"); ax[1].plot(r["t"], r["mode"], c_+"--", label=n+" mode")
for tag, c_ in [("b_m2", "k"), ("b_fp_L2", "0.5")]:
    q = np.load(DR+tag+".npz"); ax[1].plot(q["t"], q["mg"], c_, lw=.8, label="DREAM "+tag+" mean"); ax[1].plot(q["t"], q["gpk"], color=c_, ls="--", lw=.8)
ax[1].axhline(13.134, color="gray", ls=":"); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("gamma"); ax[1].legend(fontsize=6); ax[1].set_title("Relaxation")
for n, lab in [("full_g40", "Z+1=2 (full)"), ("noion_fp", "e-e only (pitch scattering x1/2)")]:
    E = np.load(D+n+".npz")["eta"][-15:].reshape(-1, 3).astype(float); g = np.sqrt(1+(E**2).sum(1))
    h, e = np.histogram(g, bins=np.linspace(1, 120, 240), density=True); ax[2].step(.5*(e[1:]+e[:-1]), h, label=lab)
ax[2].set_xlabel("gamma"); ax[2].legend(fontsize=7); ax[2].set_title("Sensitivity to the pitch-scattering strength")
plt.tight_layout(); plt.savefig(O+"p1_tail_vs_dream.png", dpi=120)
# figure 2: noise-strength scan (Langevin reduction) and kernel/DREAM reference points
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); ks = ["0", "0.25", "0.5", "1"]; res = {}
for k in ks:
    r = metrics.series(D+"sde_k%s.npz" % k, window=(50, 80), smooth=4); res[k] = r["pooled"]
    E = np.load(D+"sde_k%s.npz" % k)["eta"][-15:].reshape(-1, 3).astype(float); g = np.sqrt(1+(E**2).sum(1))
    if k != "0": ax[0].hist(g, bins=np.linspace(5, 60, 111), density=True, histtype="step", label="kappa=%s" % k)
ax[0].axvline(13.134, color="gray", ls=":", label="ODE (kappa=0)"); ax[0].legend(fontsize=8); ax[0].set_xlabel("gamma"); ax[0].set_ylabel("dN/dgamma")
ax[0].set_title("Pitch-noise strength kappa (drift fixed)")
kk = [float(k) for k in ks]
ax[1].plot(kk, [res[k]["mode"] for k in ks], "o-", label="mode"); ax[1].plot(kk, [res[k]["mean"] for k in ks], "s-", label="mean")
ax[1].plot(kk, [res[k]["p90"] for k in ks], "^-", label="90th pct")
ax[1].axhspan(16.8, 17.1, color="C0", alpha=.2); ax[1].plot([1], [rows["full_g40"]["mode"]], "k*", ms=10, label="JAX-in-Cell kernel (mode)")
ax[1].set_xlabel("kappa"); ax[1].set_ylabel("gamma"); ax[1].legend(fontsize=8); ax[1].set_title("band = DREAM mode 16.8-17.1")
plt.tight_layout(); plt.savefig(O+"p1_noise_scan.png", dpi=120)
json.dump({k: res[k] for k in ks}, open(O+"summary_sde.json", "w"), indent=1)
