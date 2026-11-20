"""Fine-dt reduced comparison (kappa 0/1) for N=2,5 at lambda=1000 against Boris runs (same S: ramp 0).  usage: redcmp2.py NAME NP dtf sig..."""
from pc2 import *
name = sys.argv[1]; NP = int(sys.argv[2]); dtf = float(sys.argv[3]); sigs = [float(x) for x in sys.argv[4:]]
q = lambda dg: np.round(np.percentile(dg, [5, 25, 50, 75, 95]), 4)
out = []
for sg in sigs:
    S = setup(name, 1000, sg, NP=NP, seed=1, ramp_wb=0.0)
    for kap in (0.0, 1.0):
        r = red_dg(S, dtf, kappa=kap, kD=0.0); dg = r["dg"]
        print(f"RED {name} sigma={sg} R={S['R']:.3g} kappa={kap} dtf={dtf} NP={NP}: rms={np.sqrt(np.mean(dg**2)):.4f} mean={dg.mean():+.4f} q={q(dg)} t={r['time']:.0f}s", flush=True)
        out.append(dict(name=name, sigma=sg, kappa=kap, dtf=dtf, dg=dg))
        pickle.dump(out, open(f"../data/redcmp2_{name}_dtf{dtf}.pkl", "wb"))
