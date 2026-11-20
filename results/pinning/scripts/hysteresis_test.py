"""Bonus: does the loss L(b)=Gamma u_w V + n V p(b) become non-monotonic (-> bistability / hysteresis in P_in)?  Variants: rho capped at 1 (baseline),
rho with the long-window closing of the diffusive window for sigma>1.6 (P3: 0.85<sigma<1.6-2.2), unsaturated QL and capped closures, n_res up to 1e17."""
import numpy as np, json
import pin_model as pm
base = pm.RHO.copy()
close = base.copy(); close[pm.SIG >= 1.87] = [0.3, 0.05, 0.02, 0.01][:int((pm.SIG >= 1.87).sum())] if (pm.SIG >= 1.87).sum() <= 4 else 0.01
res = []
for name, rho in [("rho capped at 1", base), ("rho closing for sigma>1.6", close)]:
    pm.RHO = rho
    for closure in ("QL", "cap"):
        for n in (1e13, 1e14, 1e15, 1e16, 1e17):
            for G in (4e4, 2.7e5):
                bs = np.geomspace(1e-6, 1e-2, 6000); L = pm.loss(bs, G, n, closure)
                dL = np.diff(L); nonmono = bool((dL < 0).any())
                bmax_dec = None if not nonmono else (float(bs[:-1][dL < 0].min()/pm.B_THR), float(bs[:-1][dL < 0].max()/pm.B_THR))
                Pwin = None
                if nonmono:
                    i = np.where(dL < 0)[0]; Pwin = (float(L[i[0]]), float(L[i[-1]+1]))
                res.append(dict(rho=name, closure=closure, nres=n, Gth=G, nonmonotonic=nonmono, b_over_bthr_range=bmax_dec, L_range_W=Pwin))
                if nonmono: print(name, closure, f"n={n:.0e} G={G:.1e}: NON-MONOTONIC, b/bthr in {bmax_dec}, L range {Pwin}")
print("number of non-monotonic cases:", sum(r["nonmonotonic"] for r in res), "of", len(res))
json.dump(res, open("../data/hysteresis_test.json", "w"), indent=1)
