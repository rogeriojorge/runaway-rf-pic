"""Collect data/*.npz -> tables + figures.  Map metric s* = mixing-equivalent D_eff/D_QL (see NOTES)."""
import glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pn import ql_reference
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
dd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"); fd = os.path.join(dd, "..", "figures")
from qlref import ql_fp
def refcurve(G): return ql_fp(max(G, 1.0))
LO, HI = 0.1/6, 0.5/6          # displacement levels (units Lw^2): 0.1 and 0.5 of the complete-mixing value Lw^2/6
def metrics(z):
    """s_k : mixing-equivalent D_eff/D_QL = ratio of the QL-FP (same smoothed spectrum) time to the simulated time to go from m=LO to m=HI.
       (HI not reached within T -> evaluated at T: an upper bound on the sustained rate).
       s_late: rate of growth of <dg^2> over [T/2, T] divided by the QL-FP prediction at the same displacement (1 = QL, 0 = bounded).
       fill: <dg^2>(late 10%)/(Lw^2/6); t_hi in beat periods."""
    TAU, MREF = refcurve(float(z["Gphi_over_D"]))
    m = z["m2"]/z["Lw"]**2; t = z["ts"]; Dql = float(z["Dql"]); Lw2 = float(z["Lw"])**2
    mm = np.maximum.accumulate(m)
    il = np.where(mm >= LO)[0]; ih = np.where(mm >= HI)[0]; reached = len(ih) > 0
    j = ih[0] if reached else len(t)-1; i = il[0] if len(il) and il[0] < j else 0
    tau = lambda mv: np.interp(mv, MREF, TAU)
    mi = np.mean(m[max(i-2, 0):i+3]); mj = np.mean(m[max(j-2, 0):j+3]) if reached else np.mean(m[-5:])
    s = (tau(mj)-tau(mi))/(Dql*(t[j]-t[i])/Lw2)
    k0 = len(t)//2; p = np.polyfit(t[k0:], m[k0:], 1); mT = np.mean(m[-20:])
    tq = tau(mT); dm = (np.interp(tq*1.05, TAU, MREF)-np.interp(tq*0.95, TAU, MREF))/(0.1*tq)   # dm/dtau of the ref at the reached displacement
    s_late = p[0]/(Dql/Lw2*dm)
    return s, reached, t[j]/z["Tbeat"], float(mT*6), s_late
def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{dd}/{tag}_s*_g*_seed*.npz")):
        z = dict(np.load(f)); key = (float(z["sigma_nom"]), float(z["Gphi_over_D"])); out.setdefault(key, []).append(z)
    return out
def stats(zs):
    v = np.array([[float(a[0]), float(a[1]), a[2], a[3], a[4]] for a in (metrics(z) for z in zs)]); n = len(v)
    se = lambda x: x.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
    return dict(s=v[:, 0].mean(), s_e=se(v[:, 0]), reached=v[:, 1].mean(), t_hi=v[:, 2].mean(), fill=v[:, 3].mean(), fill_e=se(v[:, 3]), sl=v[:, 4].mean(), sl_e=se(v[:, 4]), n=n)
if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "map"
    M = load(tag); sig = sorted({k[0] for k in M}); Gs = sorted({k[1] for k in M})
    keys = ["s", "s_e", "reached", "t_hi", "fill", "fill_e", "sl", "sl_e", "n"]
    R = {k: np.full((len(sig), len(Gs)), np.nan) for k in keys}
    for i, sg in enumerate(sig):
        for j, g in enumerate(Gs):
            if (sg, g) in M:
                st = stats(M[(sg, g)])
                for k in keys: R[k][i, j] = st[k]
    def table(name, key, fmt="{:.2f}"):
        print(name, "   Gamma/Delta ->", Gs)
        for i, sg in enumerate(sig): print(f"  sigma {sg:3.1f}: " + " ".join(fmt.format(R[key][i, j]) for j in range(len(Gs))))
    table("s_k = mixing-equivalent D_eff/D_QL", "s"); table("  +- seed s.e.", "s_e"); table("s_late = late growth rate / QL prediction", "sl"); table("  +- seed s.e.", "sl_e")
    table("fill = m2(T)/(Lw^2/6)", "fill"); table("  +- seed s.e.", "fill_e"); table("t_hi (beat periods to reach 0.5 of complete mixing; T=1000 means not reached)", "t_hi", "{:.0f}"); table("n seeds", "n", "{:.0f}")
    np.savez(f"{dd}/map_{tag}_summary.npz", sig=sig, Gs=Gs, **R)
    X = np.arange(len(Gs)); Y = np.arange(len(sig))
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for a_, key, ttl, norm in ((ax[0], "s", r"$s_k$: mixing-rate / QL", matplotlib.colors.LogNorm(0.03, 4)), (ax[1], "sl", r"$s_{late}$: late growth / QL", matplotlib.colors.Normalize(0, 1.5)), (ax[2], "fill", "fill <dg^2>(T)/(Lw^2/6)", matplotlib.colors.LogNorm(0.1, 30))):
        Z = np.clip(R[key], 1e-3, None); im = a_.pcolormesh(np.arange(len(Gs)+1)-0.5, np.arange(len(sig)+1)-0.5, Z, norm=norm, cmap="viridis", shading="flat")
        a_.set_xticks(X); a_.set_xticklabels([f"{g:g}" for g in Gs]); a_.set_yticks(Y); a_.set_yticklabels([f"{x:g}" for x in sig]); a_.set_title(ttl, fontsize=10)
        for i in Y:
            for j in X: a_.text(j, i, f"{R[key][i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
        a_.set_xlabel(r"$\Gamma_\phi/\Delta\omega$"); a_.set_ylabel(r"$\sigma$ (Chirikov)"); plt.colorbar(im, ax=a_)
    plt.tight_layout(); plt.savefig(f"{fd}/fig1_map_{tag}.png", dpi=130)
