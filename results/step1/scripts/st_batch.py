"""Item 3 production: reduced stationary RK4 runs.  usage: st_batch.py NAME NPERIODS DTFAC NP SEED sigma1 sigma2 ..."""
from red_st import *
name = sys.argv[1]; nb = float(sys.argv[2]); dtf = float(sys.argv[3]); NP = int(sys.argv[4]); seed = int(sys.argv[5]); sigs = [float(x) for x in sys.argv[6:]]
out = []
for sg in sigs:
    kap = float(os.environ.get('KAPPA', 0.0)); d = run_st(name, sg, nb, dtf, NP, seed=seed, kappa=kap); d.pop("g_final", None); out.append(d)
    print(f"{name} sigma_nom={sg} sigma={d['sigma']:.2f} periods={nb:g} dtfac={dtf} NP={NP} seed={seed} steps={d['nst']:.2e} t={d['time']:.0f}s ratio[.5-1]={slope_ratio(d,.5,1):.3f} [.25-.5]={slope_ratio(d,.25,.5):.3f} [0-1]={slope_ratio(d,0,1):.3f} fill={d['R_fill']:.2f}", flush=True)
    pickle.dump(out, open(os.path.expanduser(os.environ.get("OUT", "~/jic_step1/data/s1"))+f"/st_{name}_p{nb:g}_dt{dtf}_np{NP}_s{seed}_{sigs[0]}_k{os.environ.get('KAPPA','0')}.pkl", "wb"))
