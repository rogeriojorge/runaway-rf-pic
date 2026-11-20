"""Boris passage with envelope.  usage: diag3.py NAME LAM NP RAMP_WB sig1 ...   env DTFAC"""
from pcommon import *
name = sys.argv[1]; lam = float(sys.argv[2]); NP = int(sys.argv[3]); ramp = float(sys.argv[4]); sigs = [float(s) for s in sys.argv[5:]]
dtfac = float(os.environ.get("DTFAC", 1.0)); out = []
for s in sigs:
    o = boris_passage_env(name, lam, s, dtfac=dtfac, NP=NP, ramp_wb=ramp); out.append(o)
    print(f"BORIS-ENV {name} lam={lam:g} sig={s} dtfac={dtfac} ramp={ramp} NP={NP} R={o['R']:.3g} rms={np.sqrt(np.mean(o['dg']**2)):.5f} mean={o['dg'].mean():+.5f} std={o['dg'].std():.5f} steps={o['nst']:.2e} t={o['time']:.0f}s", flush=True)
    pickle.dump(out, open(os.path.expanduser(os.environ.get("OUT", "~/jic_step1/data"))+f"/d3_{name}_lam{lam:g}_dt{dtfac:g}_r{ramp:g}{os.environ.get('TAG','')}.pkl", "wb"))
