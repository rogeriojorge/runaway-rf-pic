"""Item 2: N=5, sigma=2.2 passage, reduced model, dt convergence and kappa on/off.  usage: item2.py LAM NP dtf1,dtf2,.. """
from pc2 import *
lam = float(sys.argv[1]); NP = int(sys.argv[2]); dtfs = [float(x) for x in sys.argv[3].split(",")]
S = setup("eqN5", lam, 2.2, NP=NP, seed=1, ramp_wb=0.0)
out = []
for kap in [float(x) for x in os.environ.get('KAPS', '0,1').split(',')]:
    for d in dtfs:
        r = red_dg(S, d, kappa=kap); dg = r["dg"]
        row = dict(lam=lam, sigma=2.2, R=S["R"], kappa=kap, dtf=d, NP=NP, rms=float(np.sqrt(np.mean(dg**2))), mean=float(dg.mean()), std=float(dg.std()), q=np.percentile(dg, [5, 50, 95]).tolist(), nst=r["nst"], time=r["time"], dg=dg)
        out.append(row); print(f"item2 lam={lam:g} R={S['R']:.3g} kappa={kap} dtf={d}: rms={row['rms']:.4f} mean={row['mean']:+.4f} q5,50,95={np.round(row['q'],3)} steps={r['nst']:.2e} t={r['time']:.0f}s", flush=True)
        pickle.dump(out, open(os.path.expanduser(os.environ.get("OUT", "../data"))+f"/item2_lam{lam:g}{os.environ.get('TAG', '')}.pkl", "wb"))
