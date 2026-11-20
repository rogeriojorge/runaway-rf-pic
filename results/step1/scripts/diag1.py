"""Item 1 first diagnostic: same-integrator control vs RK4 control (kick definition) in the Boris passage.  usage: diag1.py NAME LAM NP sig1 sig2 ..."""
from pcommon import *
name = sys.argv[1]; lam = float(sys.argv[2]); NP = int(sys.argv[3]); sigs = [float(s) for s in sys.argv[4:]]
dtfac = float(os.environ.get("DTFAC", 1.0)); tag = os.environ.get("TAG", "")
res = []
for s in sigs:
    o = boris_passage(name, lam, s, dtfac=dtfac, NP=NP)
    sm = summarize(o); res.append(o)
    print(json.dumps(sm), f"steps={o['nst']:.2e} t={o['time_wave']:.0f}s", flush=True)
    pickle.dump(res, open(os.path.expanduser(os.environ.get("OUT", "~/jic_step1/data"))+f"/d1_{name}_lam{lam:g}_dt{dtfac:g}{tag}.pkl", "wb"))
