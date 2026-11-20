"""Per-passage variance versus the stationary-phase quasilinear value, from expB_*.pkl."""
import sys, pickle
from exp import *
import drift

def ql_prediction(case, line, theta0, lam, b):
    """Linear-theory <dxi^2>, per particle, for one passage at sweep rate A_i (stationary phase): 1/2 F^2 b^2 2pi/A * (dxi/du_perp)^2."""
    wn = case.wn; kzn = line["kz"]
    g = solve_gamma(case, kzn, np.zeros_like(theta0), theta0)
    p = np.sqrt(g**2-1)
    u = np.stack([p*np.sin(theta0), np.zeros_like(theta0), -p*np.cos(theta0)], 1)
    d = drift.drift_vec(case, u, scale=lam)
    # d delta/dt along the drift
    eps = 1e-6
    dn = np.linalg.norm(d, axis=1)
    def dl(uv):
        gg = np.sqrt(1+np.sum(uv**2, 1)); return wn - kzn*uv[:, 2]/gg - 1/gg
    Ai = (dl(u+eps*d/dn[:, None]) - dl(u-eps*d/dn[:, None]))/(2*eps)*dn                  # Omega^2 units
    F, Fv = frad_vec(line, wn, g, theta0, full=True)
    dx = dxi_duperp(case, kzn, g, theta0)
    pred = dx**2*0.5*(F*b)**2*2*np.pi/np.abs(Ai)
    pred_s = (Fv*b)**2*2*np.pi/np.abs(Ai)/(g**2-1)/np.cos(theta0)**2      # QL gain of theta^2 per passage (vector variance / u_par^2)
    return pred, Ai, F, g, pred_s

def analyse(name):
    d = pickle.load(open(os.path.join(OUT, f"expB_{name}.pkl"), "rb"))
    case = bw.Case(); line = case.line(41.0); lam = d["lam"]
    rows = []
    for b, r in sorted(d["out"].items()):
        dxi = r["xi_t"][-1] - r["xic"]
        coll_var = float(np.mean((r["xi_ctl"]-r["xic"])**2)) if r.get("xi_ctl") is not None else 0.0
        pred, Ai, F, g, pred_s = ql_prediction(case, line, r["theta0"], lam, b)
        ds = r["thf"]**2 - r["thc"]**2                       # gain of theta^2 (2-D pitch vector variance) relative to the unperturbed run
        omb = th.omega_b(line, case.wn, g.mean(), np.mean(r["theta0"]), b)          # Omega units, mean theta
        R = omb**2/np.mean(np.abs(Ai))
        rows.append(dict(b=b, lam=lam, bphys=b/lam, R=R, meas=float(np.mean(dxi**2)-coll_var), pred=float(np.mean(pred)),
                         ratio=float((np.mean(dxi**2)-coll_var)/np.mean(pred)), mean_dxi=float(np.mean(dxi)),
                         se_ratio=float(np.sqrt(np.std(dxi**2)**2+np.std((r['xi_ctl']-r['xic'])**2 if coll_var else 0*dxi)**2)/np.sqrt(len(dxi))/np.mean(pred)),
                         kurt=float(np.mean((dxi-dxi.mean())**4)/np.var(dxi)**2),
                         meas_s=float(np.mean(ds)), pred_s=float(np.mean(pred_s)), ratio_s=float(np.mean(ds)/np.mean(pred_s)),
                         se_ratio_s=float(np.std(ds)/np.sqrt(len(ds))/np.mean(pred_s)),
                         frac_big=float(np.mean(ds > 4*np.mean(pred_s)))))
    return rows

if __name__ == "__main__":
    for rw in analyse(sys.argv[1]):
        print("b=%.2e bphys=%.2e R=%.3g  xi-metric ratio=%.3f +- %.3f | theta^2 gain ratio=%.3f +- %.3f  kurt=%.2f frac_big=%.2f" % (rw['b'], rw['bphys'], rw['R'], rw['ratio'], rw['se_ratio'], rw['ratio_s'], rw['se_ratio_s'], rw['kurt'], rw['frac_big']))
