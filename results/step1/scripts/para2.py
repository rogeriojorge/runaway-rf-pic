"""Parallel (k_perp=0) wave, N=1: Boris (exact) vs reduced with kappa in {0,+1,-1}; same ICs (ramp 0)."""
from pc2 import *
sign = float(sys.argv[1]); NP = int(sys.argv[2]); sg = float(sys.argv[3]); NAME = sys.argv[4] if len(sys.argv) > 4 else "eqN1"
def line_par(kpar):
    kz = kpar*pl.c/case.Om; s2 = np.sqrt(0.5)
    return dict(kperp=0.0, kpar=kpar, kx=0.0, kz=kz, E=np.array([1, 1j*sign, 0])*s2*(case.wn/kz), B=-1j*sign*np.array([1, 1j*sign, 0])*s2)
case.line = line_par
S = setup(NAME, 1000, sg, NP=NP, ramp_wb=0.0, ic="start")
import theory as thy
l = line_par(41.0); g = case.gamma_res(l["kz"], -1, 0.2); P1, Q1 = thy.coupling(l, wn, g, 0.2); print("P1,Q1", abs(P1), abs(Q1), "R,S", abs(P1+np.conj(Q1)), abs(P1-np.conj(Q1)), "arg(S/R)", np.angle((P1-np.conj(Q1))/(P1+np.conj(Q1))))
q = lambda dg: np.round(np.percentile(dg, [5, 25, 50, 75, 95]), 4)
for kap in (0.0, 1.0):
    r = red_dg(S, 0.05, kappa=kap, kD=0.0); dg = r["dg"]; med = np.median(dg)
    print(f"RED {NAME} kappa={kap:+.0f}: rms {np.sqrt(np.mean(dg**2)):.4f} mean {dg.mean():+.4f} median {med:+.5f} quantiles {q(dg)}", flush=True)
b = boris_dg(S, 1.0); dg = b["dg"]; med = np.median(dg); print(f"BORIS {NAME}: rms {np.sqrt(np.mean(dg**2)):.4f} mean {dg.mean():+.4f} median {med:+.5f} quantiles {q(dg)} NP={NP}")
