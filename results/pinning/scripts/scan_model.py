"""Scan the coupled model: b(P_in) for n_res = 1e11..1e16 and Gamma_th band; pinning metrics; hysteresis search.  Writes data/model_scan.json and figures."""
import json, numpy as np
import pin_model as pm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PIN = np.geomspace(1e4, 1e6, 241)
NRES = [1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
GTH = {"4e4 (min)": 4e4, "9e4 (16%)": 9e4, "2.7e5 (median)": 2.7e5, "6e5 (84%)": 6e5, "1.4e6 (max)": 1.4e6}
BT = lambda b: b*np.sqrt(pm.NEFF)
def curve(G, n, closure, Pcap=5.0):
    """upsweep (lowest stable root) and downsweep (highest stable root) of b/b_thr"""
    up, dn, nroots = [], [], []
    for P in PIN:
        r = sorted(b for b, s in pm.solve_b(P, G, n, closure, Pcap) if s)
        up.append(r[0]/pm.B_THR if r else np.nan); dn.append(r[-1]/pm.B_THR if r else np.nan); nroots.append(len(r))
    return np.array(up), np.array(dn), np.array(nroots)
def plateau(x, y, thresh=0.15, band=(0.7, 1.4)):
    """P_in interval where local elasticity dln y/dln x < thresh and y within band*b_thr"""
    e = np.gradient(np.log(y), np.log(x)); ok = (e < thresh) & (y > band[0]) & (y < band[1]) & np.isfinite(e)
    if ok.sum() < 3: return None
    idx = np.where(ok)[0]; return x[idx[0]], x[idx[-1]], float(np.mean(y[idx])), float(np.min(e[idx]))
if __name__ == "__main__":
    out = {}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(NRES)))
    for ia, (closure, ttl) in enumerate([("cap", "measured-bounded absorption (this work)"), ("QL", "unsaturated QL (Step-2 hypothesis, validated rate)"), ("S2", "Step-2 estimate 1.1e8 /s per 1e15")]):
        for ig, (gl, G) in enumerate([("2.7e5 (median)", 2.7e5), ("9e4 (16%)", 9e4)] if False else [("2.7e5 (median)", 2.7e5)]):
            pass
    summary = []
    for gi, (gname, G) in enumerate([("4e4 (min)", 4e4), ("2.7e5 (median)", 2.7e5), ("1.4e6 (max)", 1.4e6)]):
        for ci, (closure, ttl) in enumerate([("cap", "measured-bounded"), ("QL", "unsaturated QL")]):
            ax = axs[ci, gi]
            for n, col in zip(NRES, cols):
                up, dn, nr = curve(G, n, closure)
                ax.plot(PIN/1e3, up, color=col, lw=1.8, label=f"n_res={n:.0e}")
                if np.nanmax(np.abs(up-dn)) > 1e-3: ax.plot(PIN/1e3, dn, color=col, lw=1, ls=":")
                pl_ = plateau(PIN, up)
                summary.append(dict(closure=closure, Gth=G, nres=n, plateau=pl_, multi=int(nr.max()), bthr_ratio_at_1MW=float(up[-1]), bthr_ratio_at_100kW=float(np.interp(1e5, PIN, up))))
            ax.axhline(1, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
            ax.axvline(pm.pth_at_thr(G)/1e3, color="gray", ls="-.", lw=1)
            ax.set_title(f"{ttl}\nGamma_th={gname} /s", fontsize=9)
            if ci == 1: ax.set_xlabel("P_in [kW]")
            if gi == 0: ax.set_ylabel("b_pk / b_thr")
    axs[0, 0].legend(fontsize=7, loc="upper left"); axs[0, 0].set_ylim(0.1, 20)
    fig.suptitle("Wave amplitude vs coupled power.  dashed: b_thr (P3, 1.0e-4 per line).  dash-dot: P_th(b_thr)=Gamma_th u_w V.  top: replenishment-limited absorption; bottom: unsaturated QL", fontsize=10)
    fig.tight_layout(); fig.savefig("../fig_main_b_vs_Pin.png", dpi=130)
    json.dump(summary, open("../data/model_scan.json", "w"), indent=1, default=float)
    for s in summary:
        print(s["closure"], f"G={s['Gth']:.1e} n={s['nres']:.0e}", "plateau", None if s["plateau"] is None else (round(s["plateau"][0]/1e3, 1), round(s["plateau"][1]/1e3, 1), round(s["plateau"][2], 2), round(s["plateau"][3], 3)), "max#roots", s["multi"], f"b/bthr(100kW)={s['bthr_ratio_at_100kW']:.2f} (1MW)={s['bthr_ratio_at_1MW']:.2f}")
