import json, numpy as np, collections
R = "results/step1/"
import glob, pickle
runs = collections.defaultdict(list)
for f in sorted(glob.glob(R+"data/s1/st_*.pkl")):
    for d in pickle.load(open(f, "rb")): runs[(d["name"], round(d["nbeats"]), round(d["sigma"], 1), float(d.get("kappa", 0.0)))].append(d)
def sl(d, a, b):
    ts, m2 = d["ts"], d["m2"]; i0, i1 = int(a*len(ts)), int(b*len(ts)); return np.polyfit(ts[i0:i1], m2[i0:i1], 1)[0]/d["Dql"]
def cell(name, nb, sg, a, b, kp=1.0):
    L = runs.get((name, nb, sg, kp)); 
    if not L: return "  -  "
    w = np.array([d["NP"] for d in L], float); v = sum(sl(d, a, b)*d["NP"] for d in L)/w.sum()
    return f"{v:+.2f}"
sigs = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.6, 3.0, 4.0]
out = []
for kp in (1.0, 0.0):
  for name in ("eqN10", "eqN5", "gauss45"):
    for (a, b, lab) in ((0, 1, "whole window [0,T]"), (.5, 1, "late half [T/2,T]")):
        out.append(f"\n{name} (kappa={kp:.0f}): D_eff/D_QL, {lab}\n")
        out.append("| beat periods T | " + " | ".join(f"{s:.1f}" for s in sigs) + " |"); out.append("|---|" + "---|"*len(sigs))
        for nb in (16, 64, 256, 1000, 10000):
            cells = [cell(name, nb, s, a, b, kp) for s in sigs]
            if all(c.strip() == "-" for c in cells): continue
            out.append(f"| {nb} | " + " | ".join(cells) + " |")
open(R+"data/item3_tables.md", "w").write("\n".join(out)); print("\n".join(out))
