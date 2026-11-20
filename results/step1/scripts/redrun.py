"""Reduced-model passage, one config per call.  usage: redrun.py NAME LAM SIG DTF NP RAMP [NUF] ; saves ../data/<tag>.pkl"""
from pc2 import *
name = sys.argv[1]; lam = float(sys.argv[2]); sig = float(sys.argv[3]); dtf = float(sys.argv[4]); NP = int(sys.argv[5]); ramp = float(sys.argv[6])
nuf = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
fr = int(os.environ.get('FRIC', 1)); rrf = int(os.environ.get('RR', 1))
S = setup(name, lam, sig, NP=NP, ramp_wb=ramp, fric=fr, rr=rrf)
kap = float(os.environ.get('KAPPA', 0.0))
r = red_dg(S, dtf, nuf=nuf, kappa=kap)
for k in ("ks", "a", "ls", "wbj", "ddl", "G", "theta", "p", "phi", "z0n", "phis"): S.pop(k, None)
r.update(S); tag = f"red_{name}_lam{lam:g}_s{sig}_dtf{dtf}_np{NP}_r{ramp:g}_f{fr}r{rrf}_k{kap:g}"
pickle.dump(r, open(os.path.expanduser(os.environ.get("OUT", "~/jic_step1/data/s1"))+f"/{tag}.pkl", "wb"))
print(tag, "R=%.3g nst=%.2e" % (S["R"], r["nst"]), stats(r), f"{r['time']:.0f}s", flush=True)
