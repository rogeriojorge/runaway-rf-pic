import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pin_model as pm
D = "../data/"
fo = json.load(open(D+"fullorbit_transition_5e-5.json")); f1 = json.load(open(D+"ql_early_full.json"))
pts = []
for k, v in fo.items(): pts.append((v["sigma"], v["slope_early"]/v["QL"], v["slope_late"]/v["QL"]))
for k, v in f1.items(): b = float(k); pts.append((float(pm.sigma_of_b(b)), None, v["slope"]/v["QL"]))
pts.sort(key=lambda x: x[0])
fig, ax = plt.subplots(figsize=(6, 4.2))
s = [p[0] for p in pts]; ax.plot(s, [max(p[2], 1e-3) for p in pts], "s-", label="full orbit, window 0.5-1 T (late)")
se = [(p[0], p[1]) for p in pts if p[1] is not None]; ax.plot(*zip(*[(a, max(b, 1e-3)) for a, b in se]), "o--", label="full orbit, window 0.1-0.5 T (early)")
sg = np.linspace(0.4, 3, 100); ax.plot(sg, [float(np.interp(x, pm.SIG, pm.RHO)) for x in sg], "k:", label="D_eff/D_QL (P3 stationary, capped 1)")
ax.set_yscale("log"); ax.set_xlabel("sigma (Chirikov, adjacent lines)"); ax.set_ylabel("wave work rate / QL prediction"); ax.axvline(1, color="gray", ls="-."); ax.legend(fontsize=8)
ax.set_title("Unsaturated absorption across the overlap threshold (collisionless, full orbit)", fontsize=9)
fig.tight_layout(); fig.savefig("../fig4_transition.png", dpi=130)
print(pts)
