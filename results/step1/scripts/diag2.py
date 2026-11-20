"""reduced passage dt scan.  usage: diag2.py NAME LAM NP dtf1,dtf2,.. sig1 sig2 ..."""
from pcommon import *
name = sys.argv[1]; lam = float(sys.argv[2]); NP = int(sys.argv[3]); dtfs = [float(x) for x in sys.argv[4].split(",")]; sigs = [float(s) for s in sys.argv[5:]]
out = []
for s in sigs:
    for d in dtfs:
        o = red_passage(name, lam, s, dtf=d, NP=NP); out.append(o)
        print(f"{name} lam={lam:g} sig={s} dtf={d} NP={NP} R={o['R']:.3g} rms={np.sqrt(np.mean(o['dg']**2)):.5f} mean={o['dg'].mean():+.5f} std={o['dg'].std():.5f} steps={o['nst']:.2e} t={o['time']:.0f}s", flush=True)
        pickle.dump(out, open(os.path.expanduser(os.environ.get("OUT", "~/jic_step1/data"))+f"/d2_{name}_lam{lam:g}{os.environ.get('TAG','')}.pkl", "wb"))
