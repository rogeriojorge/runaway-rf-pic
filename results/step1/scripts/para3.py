from pc2 import *
sign = 1.0; NP = 512; sg = float(sys.argv[1]); NAME = sys.argv[2]
def line_par(kpar):
    kz = kpar*pl.c/case.Om; s2 = np.sqrt(0.5)
    return dict(kperp=0.0, kpar=kpar, kx=0.0, kz=kz, E=np.array([1, 1j*sign, 0])*s2*(case.wn/kz), B=-1j*sign*np.array([1, 1j*sign, 0])*s2)
case.line = line_par
S = setup(NAME, 1000, sg, NP=NP, ramp_wb=0.0, ic="start")
q = lambda dg: np.round(np.percentile(dg, [5, 25, 50, 75, 95]), 4)
for ex in (0, 1):
    os.environ["EXACTPAR"] = str(ex)
    for kap in (0.0, 1.0):
        r = red_dg(S, 0.05, kappa=kap, kD=0.0); dg = r["dg"]
        print(f"{NAME} sigma={sg} exactpar={ex} kappa={kap}: rms {np.sqrt(np.mean(dg**2)):.4f} median {np.median(dg):+.4f} q {q(dg)}", flush=True)
