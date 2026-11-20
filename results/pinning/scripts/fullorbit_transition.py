"""Full-orbit wave work rate (collisionless, no drive, P1 zone ensemble, Gaussian comb) across the overlap threshold: ratio to the QL energy-balance prediction."""
import numpy as np, json, sys
from fullorbit_check import run_full
import pin_model as pm
out = {}
for b in [float(x) for x in sys.argv[1:]]:
    T = 1.5e-6
    r = run_full(b, 0.0, 768, T, drive=False, nrec_blocks=20)
    t = np.linspace(0, T, 21)[1:]; Ew = np.array(r['Ew_t']); sem = r['Ew_sem']
    m = (t > 0.1*T)&(t < 0.5*T); sl = np.polyfit(t[m], Ew[m], 1)[0]; m2 = t > 0.5*T; sl2 = np.polyfit(t[m2], Ew[m2], 1)[0]
    q = pm.P_QL_zone(b)/pm.MC2
    print(f"FULL b={b:.1e} sigma={float(pm.sigma_of_b(b)):.2f}: slope(0.1-0.5T)={sl:.3e} slope(0.5-1T)={sl2:.3e} mc2/s; QL={q:.3e}; ratios {sl/q:.3f} {sl2/q:.3f}; Ew_end={Ew[-1]:.3e}+-{sem:.1e}", flush=True)
    out[str(b)] = dict(sigma=float(pm.sigma_of_b(b)), slope_early=float(sl), slope_late=float(sl2), QL=float(q), t=t.tolist(), Ew=Ew.tolist(), sem=sem)
    json.dump(out, open("../data/fullorbit_transition_%s.json" % sys.argv[1], "w"))
