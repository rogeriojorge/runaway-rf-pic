"""Pinning metric: width of the P_in interval over which b stays within +-10% of b_thr, R_pin = P_in(1.1 b_thr)/P_in(0.9 b_thr) (no absorption: 1.49 = (1.1/0.9)^2),
and the absorbed fraction f_abs = P_abs/P_in at b_thr.  Also P_in needed to reach 2 b_thr relative to the no-runaway case."""
import numpy as np, json
import pin_model as pm
rows = []
for closure, Pc in [("cap", 1.1), ("cap", 5.0), ("cap", 7.7), ("QL", None)]:
    for G in (4e4, 9e4, 2.7e5):
        for n in (1e13, 1e14, 1e15, 4e15, 1e16, 1e17):
            kw = dict(closure=closure, Pcap=Pc if Pc else 5.0)
            P = lambda b: pm.loss(b, G, n, **kw)
            Rpin = P(1.1*pm.B_THR)/P(0.9*pm.B_THR); fabs = n*pm.V*pm.p_zone(pm.B_THR, **kw)/P(pm.B_THR)
            P0 = lambda b: pm.loss(b, G, 0.0, **kw)
            rows.append(dict(closure=closure, Pcap=Pc, Gth=G, nres=n, Rpin=float(Rpin), fabs_at_bthr=float(fabs), Pth_kW=float(P0(pm.B_THR)/1e3), P_at_2bthr_over_no_RE=float(P(2*pm.B_THR)/P0(2*pm.B_THR))))
json.dump(rows, open("../data/pin_metrics.json", "w"), indent=1)
for r in rows:
    if r["nres"] in (1e14, 1e15, 4e15, 1e16, 1e17) and r["Gth"] in (4e4, 2.7e5):
        print(f"{r['closure']:3s} cap={r['Pcap']} G={r['Gth']:.1e} n={r['nres']:.0e}  P_th={r['Pth_kW']:.0f} kW  R_pin={r['Rpin']:.2f} (1.49 = none)  f_abs(b_thr)={r['fabs_at_bthr']:.3f}  P(2b_thr)/P0={r['P_at_2bthr_over_no_RE']:.2f}")
