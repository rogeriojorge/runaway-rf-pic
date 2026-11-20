import re, os, glob, pickle, numpy as np
R = os.path.expanduser("results/step1/")
head = open("/tmp/head.md").read()
a = open("/tmp/body_ab.md").read()
c = open("/tmp/body_c.md").read()
# --- lambda=1 Richardson
def lam1_rows():
    rows = []
    for f in sorted(glob.glob(R+"data/s1/red_eqN2_lam1_*k1.pkl")+glob.glob(R+"data/s1/red_eqN2_lam1_*.pkl")):
        pass
    for f in sorted(set(glob.glob(R+"data/s1/red_eqN2_lam1_*_k1.pkl"))):
        d = pickle.load(open(f, "rb")); dg = d["dg"]
        m = re.search(r"s([0-9.]+)_dtf([0-9.]+)_np(\d+)", f)
        rows.append((float(m.group(1)), float(m.group(2)), int(m.group(3)), float(np.sqrt(np.mean(dg**2))), float(dg.mean()), float(dg.std())))
    return sorted(rows)
rows = lam1_rows()
if rows:
    txt = "rms dg (mean) per omega_b dt: " + "; ".join(f"sigma {s}: dt {dt} (NP {np_}) {r:.4f} ({m:+.4f})" for s, dt, np_, r, m, sd in rows) + ". "
else:
    txt = ""
a = a.replace("LAMBDA1_PLACEHOLDER", txt + LAM1_TAIL if False else txt + "LAM1_TAIL")
open(R+"NOTES.md", "w").write(head + a + c)
