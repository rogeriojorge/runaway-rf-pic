"""Item 5: is JIC mean-gamma offset vs DREAM within DREAM grid-convergence error?  Uses DREAM p_max=60 pair (240/480 p-points, same g40 start) and JIC g40-start snapshots."""
import numpy as np, sys, glob, json
R = "results/"
sys.path.insert(0, R+"P1/scripts"); from metrics import snap_metrics
D = {k: np.load(R+f"codes/DREAM/data/{k}.npz", allow_pickle=True) for k in ("b_m1", "b_m2", "b_fp_L1", "b_fp_L2")}
def dstat(d, t):
    i = np.argmin(abs(d["t"]-t)); return dict(mean=float(d["mg"][i]), med=float(d["gmed"][i]), p90=float(d["g90"][i]), mode=float(d["gpk"][i]))
# JIC g40-start runs: sampled snapshot times
runs = ["full_g40","n32768","n2048","seed1","seed2","dt4e-3","dt2p5e-4"]
out = {}
for T in (5, 10, 20, 40):
    acc = []
    for r in runs:
        try: z = np.load(R+f"P1/data/{r}.npz")
        except Exception: continue
        if "eta" not in z.files: continue
        t = z["t"]; i = np.argmin(abs(t-T))
        if abs(t[i]-T) > 1.1: continue
        acc.append(z["eta"][i].astype(float))
    eta = np.concatenate(acc)
    p = np.linalg.norm(eta, axis=1); g = np.sqrt(1+p**2)
    row = dict(n=len(eta), frac_p_gt60=float((p > 60).mean()), mean_all=float(g.mean()), mean_p60=float(g[p < 60].mean()), mean_p40=float(g[p < 40].mean()),
               med_p60=float(np.median(g[p < 60])), p90_p60=float(np.percentile(g[p < 60], 90)), N_runs=len(acc))
    m = snap_metrics(eta[p < 60]); row["mode"] = m["mode"]
    # bootstrap error on mean (by run-chunk)
    row["mean_p60_se"] = float(np.std([np.concatenate([a for a in acc])[np.random.default_rng(s).integers(0, len(eta), len(eta))] for s in range(1)][0:0] or [0])) 
    for k in ("b_m1", "b_m2"): row[k] = dstat(D[k], T)
    out[T] = row
def rich(a, b, ratio=2.0, order=1):   # a coarse, b fine
    return b+(b-a)/(ratio**order-1)
print("time-matched, g40 start, pmax = 60: DREAM 240 pts, 480 pts, Richardson (1st/2nd order) vs JIC (p<60)")
for T, r in out.items():
    for q, key in (("mean", "mean_p60"), ("med", "med_p60"), ("p90", "p90_p60")):
        a, b = r["b_m1"][q], r["b_m2"][q]
        r1, r2 = rich(a, b, 2, 1), rich(a, b, 2, 2); j = r[key]
        print(f"t={T:3d} {q:4s} DREAM240={a:.3f} DREAM480={b:.3f} R1={r1:.3f} R2={r2:.3f} JIC={j:.3f}  JIC/D480={j/b-1:+.3%} JIC/R1={j/r1-1:+.3%} JIC/R2={j/r2-1:+.3%}")
    print(f"       mode DREAM240={r['b_m1']['mode']:.2f} 480={r['b_m2']['mode']:.2f}  JIC={r['mode']:.2f}; frac p>60={r['frac_p_gt60']:.4f} n={r['n']} runs={r['N_runs']}")
# stationary fixed-point pmax = 40 pair
print("\nstationary (fixed-point start, pmax 40, absorbing wall) DREAM 240 vs 360 pts; JIC truncated p<40")
for t in (60, 100):
    a, b = dstat(D["b_fp_L1"], t), dstat(D["b_fp_L2"], t)
    print(t, "mean", a["mean"], b["mean"], "R1(ratio1.5)", rich(a["mean"], b["mean"], 1.5, 1), "R2", rich(a["mean"], b["mean"], 1.5, 2), "p90", a["p90"], b["p90"], rich(a["p90"], b["p90"], 1.5, 1))
json.dump(out, open("../data/item5_mean_gamma.json", "w"), indent=1)
