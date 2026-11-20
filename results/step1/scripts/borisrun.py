"""Boris passage, one config per call.  usage: borisrun.py NAME LAM SIG DTFAC NP SEED RAMP [XW] ; saves ../data/s1/bor_*.pkl"""
from pc2 import *
name = sys.argv[1]; lam = float(sys.argv[2]); sig = float(sys.argv[3]); dtfac = float(sys.argv[4]); NP = int(sys.argv[5]); seed = int(sys.argv[6]); ramp = float(sys.argv[7])
xw = float(sys.argv[8]) if len(sys.argv) > 8 else 6.0
fr = int(os.environ.get('FRIC', 1)); rrf = int(os.environ.get('RR', 1))
S = setup(name, lam, sig, NP=NP, seed=seed, ramp_wb=ramp, Xw=xw, fric=fr, rr=rrf)
r = boris_dg(S, dtfac)
for k in ("ks", "a", "ls", "wbj", "ddl", "G", "theta", "p", "phi", "z0n", "phis"): S.pop(k, None)
r.update(S); tag = f"bor_{name}_lam{lam:g}_s{sig}_dt{dtfac}_np{NP}_seed{seed}_r{ramp:g}_xw{xw:g}_f{fr}r{rrf}"
pickle.dump(r, open(os.path.expanduser(os.environ.get("OUT", "~/jic_step1/data/s1"))+f"/{tag}.pkl", "wb"))
dg = r["dg"]; med = np.median(dg); out = dg > med+0.01
print(tag, "R=%.3g nst=%.2e" % (S["R"], r["nst"]), stats(r), "outfrac=%.3f" % out.mean(), f"{r['time']:.0f}s", flush=True)
