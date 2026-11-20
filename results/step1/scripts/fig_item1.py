import pickle, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/step1/"
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
# (a) capture fraction vs dt (Boris, N=1, sigma=0.7, lambda=1000)
dts, fr, er = [], [], []
for dt in ("0.5", "0.25", "0.125"):
    n = o = 0
    for f in glob.glob(R+f"data/s1/bor_eqN1_lam1000_s0.7_dt{dt}_np128_seed*_r20_xw6.pkl"):
        dg = pickle.load(open(f, "rb"))["dg"]; med = np.median(dg); n += len(dg); o += int((dg > med+0.01).sum())
    dts.append(float(dt)); fr.append(o/n); er.append(np.sqrt(fr[-1]*(1-fr[-1])/n))
ax[0].errorbar(np.array(dts)*2*np.pi*16.4/32, fr, er, fmt="o-"); ax[0].axhline(0.063, color="r", ls="--", label="reduced + kappa (6.3%)"); ax[0].axhline(0.0, color="g", ls=":", label="reduced, no kappa (0%)")
ax[0].set_xscale("log"); ax[0].set_ylim(0, 0.15); ax[0].set_xlabel("Boris step (1/Omega units)"); ax[0].set_ylabel("captured fraction"); ax[0].set_title("N=1, sigma=0.7, lambda=1000"); ax[0].legend(fontsize=7)
# (b) histogram of dg: Boris vs reduced
b = np.concatenate([pickle.load(open(f, "rb"))["dg"] for f in glob.glob(R+"data/s1/bor_eqN1_lam1000_s0.7_dt0.25_np128_seed*_r20_xw6.pkl")])
bins = np.linspace(-0.02, 0.06, 81)
ax[1].hist(b, bins, histtype="step", density=True, label="full orbit (Boris, 512)", color="k")
import os
ax[1].set_yscale("log"); ax[1].set_xlabel("dgamma (wave - same-integrator control)"); ax[1].set_title("kick distribution"); ax[1].legend(fontsize=7)
red = R+"data/item1_red_hist.npz"
if os.path.exists(red):
    z = np.load(red); ax[1].hist(z["k0"], bins, histtype="step", density=True, color="g", label="reduced, no kappa"); ax[1].hist(z["k1"], bins, histtype="step", density=True, color="r", label="reduced + kappa"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(R+"figs/item1_capture_branch.png", dpi=130)
