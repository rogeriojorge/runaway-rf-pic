import pickle, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/step1/data/"
rows = []
for f in glob.glob(R+"item2_lam*.pkl")+glob.glob(R+"s1/item2_lam*.pkl"):
    rows += pickle.load(open(f, "rb"))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for a, lam in zip(ax, (100, 30, 10)):
    for kap, mk in ((0.0, "o"), (1.0, "s")):
        rr = sorted([r for r in rows if r["lam"] == lam and r["kappa"] == kap], key=lambda r: r["dtf"])
        if rr: a.plot([r["dtf"] for r in rr], [r["mean"] for r in rr], mk+"-", label=("with kappa" if kap else "no kappa (P3 model)"))
        for r in rr: a.plot([r["dtf"]]*2, [r["q"][0], r["q"][2]], "-", color="gray", lw=.6)
    a.set_xscale("log"); a.set_xlabel("omega_b dt"); a.set_title(f"lambda = {lam}, N=5, sigma=2.2"); a.axhline(0, color="k", lw=.4)
ax[0].set_ylabel("mean dg (bars: 5-95%)"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/step1/figs/item2_dt_convergence.png", dpi=130)
