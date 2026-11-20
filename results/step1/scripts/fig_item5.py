import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/"
d = json.load(open(R+"step1/data/item5_mean_gamma.json"))
T = [20, 40]; fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for i, q in enumerate(("mean", "med", "p90")):
    key = {"mean": "mean_p60", "med": "med_p60", "p90": "p90_p60"}[q]
    for T_ in (10, 20, 40):
        r = d[str(T_)]; a, b = r["b_m1"][q], r["b_m2"][q]; j = r[key]
        ax[i].plot([240, 480, 1e3], [a, b, b+(b-a)], "o-", label=f"DREAM t={T_} s" if i == 0 else None)
        ax[i].plot([1e3], [j], "k*", ms=9 if T_ == 40 else 5)
    ax[i].set_xscale("log"); ax[i].set_title({"mean": "mean gamma", "med": "median", "p90": "p90"}[q]); ax[i].set_xlabel("DREAM p-points (1e3 = 1st-order extrapolation)")
ax[0].legend(fontsize=7); ax[0].set_ylabel("gamma (g40 start, p_max 60)")
fig.suptitle("DREAM grid convergence vs JAX-in-Cell (stars, p<60): offset is inside the DREAM grid error", fontsize=9)
plt.tight_layout(); plt.savefig(R+"step1/figs/item5_dream_grid_convergence.png", dpi=130)
