import glob, pickle, json, numpy as np, collections, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/step1/"
runs = collections.defaultdict(list)
for f in sorted(glob.glob(R+"data/s1/st_*.pkl")):
    for d in pickle.load(open(f, "rb")):
        runs[(d["name"], round(d["nbeats"]), round(d["sigma"], 2), float(d.get("kappa", 0.0)))].append(d)
def slope_ratio(ts, m2, Dql, f0, f1):
    i0, i1 = int(f0*len(ts)), int(f1*len(ts)); return np.polyfit(ts[i0:i1], m2[i0:i1], 1)[0]/Dql
rows = []
for (nm, nb, sg, kp), L in sorted(runs.items()):
    w = np.array([d["NP"] for d in L], float); m2 = sum(d["m2"]*d["NP"] for d in L)/w.sum(); d0 = L[0]
    rr = lambda a, b: float(np.mean([slope_ratio(d["ts"], d["m2"], d["Dql"], a, b) for d in L]))
    # fill at a set of periods (from the single trajectory sampled at 64 times)
    beats = d0["beats"]; Lw2 = d0["Lw"]**2/6
    fills = {int(p): float(np.interp(p, beats, m2)/Lw2) for p in (16, 64, 256, 1000, 10000) if p <= beats[-1]*1.0001}
    rows.append(dict(kappa=kp, name=nm, nbeats=nb, sigma=sg, nseeds=len(L), NP=int(w.sum()), ratio_late=rr(.5, 1), ratio_mid=rr(.25, .5), ratio_early=rr(0, .25), fill_end=float(m2[-1]/Lw2), fills=fills,
                     ratio_late_by_seed=[slope_ratio(d["ts"], d["m2"], d["Dql"], .5, 1) for d in L]))
json.dump(rows, open(R+"data/item3_summary.json", "w"), indent=1)
for r in rows: print(f"k={r['kappa']:.0f} {r['name']:7s} T={r['nbeats']:6d} periods sigma={r['sigma']:.2f} seeds={r['nseeds']} NP={r['NP']:4d}  D/DQL late={r['ratio_late']:+.3f} mid={r['ratio_mid']:+.3f} early={r['ratio_early']:+.3f}  fill(end)={r['fill_end']:.2f}")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for nm, mk in (("eqN5", "s"), ("eqN10", "o"), ("gauss45", "^")):
    for nb, ls in ((1000, "-"), (10000, "--")):
        rr_ = [r for r in rows if r["name"] == nm and r["nbeats"] == nb and r["kappa"] == 1.0]
        if not rr_: continue
        s = [r["sigma"] for r in rr_]
        ax[0].plot(s, [r["fill_end"] for r in rr_], mk+ls, label=f"{nm}, {nb} periods")
        ax[1].plot(s, [r["ratio_late"] for r in rr_], mk+ls, label=f"{nm}, {nb}")
ax[0].set_xlabel("overlap sigma"); ax[0].set_ylabel("<dgamma^2>(T) / (L_band^2/6)"); ax[0].set_title("band filling at the end of the window"); ax[0].legend(fontsize=6); ax[0].set_yscale("log")
ax[1].axhline(0, color="k", lw=.5); ax[1].set_xlabel("overlap sigma"); ax[1].set_ylabel("late-half D_eff/D_QL"); ax[1].set_title("late-time diffusion rate"); ax[1].legend(fontsize=6)
for (nm, nb, sg, kp), L in sorted(runs.items()):
    if kp != 1.0: continue
    if nm == "eqN10" and nb == 1000 and sg in (1.02, 1.4, 1.6, 2.2, 3.0) or (nm == "eqN10" and nb == 10000):
        d = L[0]; ax[2].loglog(d["beats"], d["m2"]/(d["Lw"]**2/6), label=f"sigma={sg} ({nb})")
ax[2].set_xlabel("beat periods"); ax[2].set_ylabel("fill"); ax[2].set_title("eqN10: fill vs time"); ax[2].legend(fontsize=5)
plt.tight_layout(); plt.savefig(R+"figs/item3_closing.png", dpi=130)
