"""Item 4: single-line passage D_eff/D_QL (gamma metric) vs R, window Xs*sqrt(A) and ramp, reduced model.  usage: plateau.py LAM NP Xs tr_sA"""
from pc2 import *
lam = float(sys.argv[1]); NP = int(sys.argv[2]); Xs = float(sys.argv[3]); trsA = float(sys.argv[4])
res = []
for sg in (0.01, 0.02, 0.03, 0.045, 0.065, 0.09, 0.12):
    S = setup("eqN1", lam, sg, NP=NP, Xs=Xs, tr_sA=trsA, Xw=0.0, ic="cross")
    ql = ql_value(S)
    dtf = 0.04*min(1.0, np.sqrt(S["R"]))
    r = red_dg(S, dtf)
    m2 = float(np.mean(r["dg"]**2)); se = float(np.std(r["dg"]**2)/np.sqrt(NP))
    res.append(dict(sigma=sg, R=S["R"], ratio=m2/ql, se=se/ql, Xs=Xs, tr_sA=trsA, lam=lam, mean=float(r["dg"].mean()), nst=r["nst"]))
    print(f"red lam={lam:g} Xs={Xs:g} tr_sA={trsA:g} sigma={sg} R={S['R']:.3g} ratio={m2/ql:.4f}+-{se/ql:.4f} mean_dg={r['dg'].mean():+.2e} steps={r['nst']}", flush=True)
json.dump(res, open(f"../data/plateau_red_lam{lam:g}_Xs{Xs:g}_tr{trsA:g}.json", "w"))
