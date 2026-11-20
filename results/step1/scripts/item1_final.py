"""Item 1 final table: reduced (kappa 0 / Strang-split kappa 1, dtf given) vs full-orbit Boris pickles (same S).  usage: item1_final.py dtf"""
from pc2 import *
import glob
dtf = float(sys.argv[1]); R = os.path.expanduser("results/step1/data/")
q = lambda dg: np.round(np.percentile(dg, [5, 50, 95]), 4)
cases = [("eqN1", 0.7, 20.0, 4096, "bor_eqN1_lam1000_s0.7_dt0.25_np128_seed*_r20_xw6.pkl"),
         ("eqN2", 0.7, 0.0, 1024, "bor_eqN2_lam1000_s0.7_dt1.0_np128_seed*_r0_xw6_f1r1.pkl"),
         ("eqN2", 1.4, 0.0, 1024, "bor_eqN2_lam1000_s1.4_dt1.0_np128_seed*_r0_xw6_f1r1.pkl"),
         ("eqN5", 1.2, 0.0, 256, "bor_eqN5_lam1000_s1.2_dt1.0_np64_seed*_r0_xw6_f1r1.pkl"),
         ("eqN5", 1.6, 0.0, 256, "bor_eqN5_lam1000_s1.6_dt1.0_np64_seed*_r0_xw6_f1r1.pkl"),
         ("eqN5", 2.2, 0.0, 256, "bor_eqN5_lam1000_s2.2_dt1.0_np64_seed*_r0_xw6_f1r1.pkl")]
rows = []
for name, sg, ramp, NP, pat in cases:
    b = np.concatenate([pickle.load(open(f, "rb"))["dg"] for f in glob.glob(R+"s1/"+pat)])
    S = setup(name, 1000, sg, NP=NP, seed=1, ramp_wb=ramp)
    row = dict(name=name, sigma=sg, ramp=ramp, boris=dict(NP=len(b), rms=float(np.sqrt(np.mean(b**2))), mean=float(b.mean()), q=q(b).tolist()))
    for kap in (0.0, 1.0):
        d = red_dg(S, dtf if name != "eqN1" else 0.03, kappa=kap, kD=0.0)["dg"]
        row[f"red_k{kap:g}"] = dict(NP=len(d), rms=float(np.sqrt(np.mean(d**2))), mean=float(d.mean()), q=q(d).tolist())
    if name == "eqN1":
        for k in ("boris", "red_k0", "red_k1"):
            pass
        bb = b; med = np.median(bb); row["boris"]["capt"] = float(np.mean(bb > med+0.01))
        for kap in (0.0, 1.0):
            dd = red_dg(S, 0.03, kappa=kap, kD=0.0)["dg"]; m = np.median(dd); row[f"red_k{kap:g}"]["capt"] = float(np.mean(dd > m+0.01)); row[f"red_k{kap:g}"]["median"] = float(m)
        row["boris"]["median"] = float(med)
    rows.append(row); print(name, sg, {k: ({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in row.items() if k.startswith(("boris", "red"))}, flush=True)
json.dump(rows, open(R+"item1_final.json", "w"), indent=1)
