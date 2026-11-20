"""Stationary-tail metrics, defined as in results/codes/DREAM/scripts/caseb.py::moments:
mean gamma (number weighted), median/p90 of gamma, mode of dN/dgamma over p>3 on a dp bin grid,
pitch at the mode = arccos(<xi> in that p bin)."""
import numpy as np

def snap_metrics(eta, w=None, dp=0.25, pmax=60., smooth=0):
    """eta (n,3) in units of m c, x along B. Returns dict. Pools all rows (use pooled snapshots for the mode)."""
    p = np.linalg.norm(eta, axis=1); g = np.sqrt(1+p**2); xi = eta[:, 0]/np.maximum(p, 1e-30)
    w = np.ones(len(p)) if w is None else w
    w = w/w.sum()
    edges = np.arange(0, pmax+dp, dp); pc = .5*(edges[1:]+edges[:-1])
    h, _ = np.histogram(p, edges, weights=w)
    gam = np.sqrt(1+edges**2); dg = np.diff(gam); Fg = h/dg
    if smooth:
        k = np.exp(-.5*(np.arange(-3*smooth, 3*smooth+1)/smooth)**2); k /= k.sum()
        Fg = np.convolve(Fg, k, mode="same")
    m = pc > 3
    j = np.argmax(np.where(m, Fg, -1)); gm = np.sqrt(1+pc[j]**2)
    sel = (p >= edges[j]) & (p < edges[j+1])
    mx = np.average(xi[sel], weights=w[sel]) if sel.any() else np.nan
    o = np.argsort(g); cw = np.cumsum(w[o]); 
    q = lambda f: float(np.interp(f, cw, g[o]))
    return dict(mean=float(np.sum(w*g)), median=q(.5), p90=q(.9), mode=float(gm), theta_mode=float(np.arccos(np.clip(mx, -1, 1))),
                rms_theta=float(np.sqrt(np.sum(w*np.arccos(np.clip(xi, -1, 1))**2))), frac_lost=float(np.sum(w*(g < 3))))

def series(path, window=None, **kw):
    d = np.load(path); t = d["t"]; E = d["eta"].astype(np.float64); w = d["wt"]
    rows = [snap_metrics(E[i], w, **kw) for i in range(len(t))]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["t"] = t
    if window:
        sel = (t >= window[0]) & (t <= window[1])
        pooled = E[sel].reshape(-1, 3); ww = np.tile(w, sel.sum())
        out["pooled"] = snap_metrics(pooled, ww, smooth=kw.get("smooth", 0))
        out["pooled_n"] = int(sel.sum())
        # block means of snapshot metrics, for error bars
        idx = np.where(sel)[0]; b = np.array_split(idx, 4)
        out["blocks"] = [snap_metrics(E[bi].reshape(-1, 3), np.tile(w, len(bi)), smooth=kw.get("smooth", 0)) for bi in b]
    return out
