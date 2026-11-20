"""Reduced model with/without the kappa term at the same S as borisrun (ramp 0): usage redcmp.py NAME LAM NP dtf sig..."""
from pc2 import *
name = sys.argv[1]; lam = float(sys.argv[2]); NP = int(sys.argv[3]); dtf = float(sys.argv[4]); sigs = [float(x) for x in sys.argv[5:]]
out = []
for sg in sigs:
    S = setup(name, lam, sg, NP=NP, seed=1, ramp_wb=0.0)
    for kap in (0.0, 1.0):
        r = red_dg(S, dtf, kappa=kap); dg = r["dg"]
        print(f"RED {name} lam={lam:g} sigma={sg} R={S['R']:.3g} kappa={kap} NP={NP} dtf={dtf}: rms={np.sqrt(np.mean(dg**2)):.4f} mean={dg.mean():+.4f} median={np.median(dg):+.4f} std={dg.std():.4f}  steps={r['nst']:.2e} t={r['time']:.0f}s", flush=True)
        out.append(dict(name=name, lam=lam, sigma=sg, R=S["R"], kappa=kap, dtf=dtf, NP=NP, dg=dg))
        pickle.dump(out, open(f"../data/redcmp_{name}_lam{lam:g}_dtf{dtf}.pkl", "wb"))
