"""Diagnostic: parallel-propagating circularly polarised wave (k_perp=0): capture fraction in Boris vs reduced.  usage: para.py sign NP sigma..."""
from pc2 import *
import plasma
sign = float(sys.argv[1]); NP = int(sys.argv[2]); sigs = [float(s) for s in sys.argv[3:]]
orig = case.line
def line_par(kpar):
    kz = kpar*pl.c/case.Om
    s2 = np.sqrt(0.5)
    E0 = np.array([1, 1j*sign, 0])*s2*(case.wn/kz)       # E in B0 c units per unit b
    Bv = -1j*sign*np.array([1, 1j*sign, 0])*s2
    return dict(kperp=0.0, kpar=kpar, kx=0.0, kz=kz, E=E0, B=Bv)
case.line = line_par
for sg in sigs:
    S = setup("eqN1", 1000, sg, NP=NP, ramp_wb=20.0, ic="start")
    # R=35 requires b scaling: report R and F
    b = boris_dg(S, 1.0); r = red_dg(S, 0.05)
    for nm, d in (("Boris", b), ("red", r)):
        dg = d["dg"]; med = np.median(dg); print(f"sign={sign} sigma={sg} R={S['R']:.3g} wb={S['wb']:.3g} {nm}: median={med:+.5f} std={dg.std():.5f} outlier(>med+3*w_est)={np.mean(abs(dg-med)>0.01):.3f}", flush=True)
