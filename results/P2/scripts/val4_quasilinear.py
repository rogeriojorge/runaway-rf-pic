"""V4: broadband random-phase spectrum at small amplitude recovers the quasilinear (Kennel-Engelmann/golden-rule) kick variance."""
import sys
from exp import *
case = bw.Case(); wn = case.wn
nl = 129
kz0, dk = 41.0, 0.08                       # 1/m: lines 35.6..46.4 1/m, spacing 0.08 (=2.4e7 rad/s in delta)
kpars = kz0 + dk*(np.arange(nl)-nl//2)
b = 3e-8                                   # per line; sqrt(nl)*b = 3.4e-7 rms total
lines0, amps0, ls = bw.make_lines(case, list(kpars), np.full(nl, b), np.zeros(nl))
omb1 = th.omega_b(ls[nl//2], wn, case.gamma_res(ls[nl//2]["kz"], -1, 0.2), 0.2, b)*case.Om
print("single-line omega_b [1/s]", omb1, " spacing in delta [1/s]", dk*pl.c*np.cos(0.2))
N = 512; nreal = 16
tmax = 6e-9            # s  (recurrence time 2pi/(dk c) = %.1f ns)
print("recurrence time ns", 2*np.pi/(dk*pl.c)*1e9)
dt = 2*np.pi*16.4/32
nst = int(tmax*case.Om/dt); nrec = nst//24; nst = nrec*24
ts = (np.arange(1, 25)*nrec)*dt/case.Om
# ensemble: delta0 uniform within central +/-25 spacings so that many lines overlap each particle's response
rows_meas, rows_pred = [], []
for rl in range(nreal):
    rng = np.random.default_rng(100+rl)
    phases = rng.uniform(0, 2*np.pi, nl)
    lines, amps, _ = bw.make_lines(case, list(kpars), np.full(nl, b), phases)
    dsp = dk*pl.c/case.Om*np.cos(0.2)
    d0 = rng.uniform(-20*dsp, 20*dsp, N)
    x, u, theta, g = build(case, ls[nl//2], N, (0.1, 0.3), d0, 200+rl, zrange=2*np.pi/(dk*pl.c/case.Om))
    st, rec = bw.simulate((x, u, 0.0), nst, nrec, dt, wn, lines, amps, rec_fn=rec_u)
    s_t = np.array([quantities(r)[0] for r in np.asarray(rec)])        # (24,N)
    s_0 = quantities(u)[0]
    meas = np.mean((s_t - s_0[None, :])**2, 1)
    # linear theory, random-phase expectation: sum_j 1/2 F_j^2 b^2 sinc^2
    pred = np.zeros(24)
    for j, lj in enumerate(ls):
        F = frad_vec(lj, wn, g, theta)
        dl = delta_of(case, lj["kz"], g, theta)
        for it, t in enumerate(ts*case.Om):
            pred[it] += np.mean(0.5*(F*b)**2*(np.sin(dl*t/2)/(dl/2))**2)
    rows_meas.append(meas); rows_pred.append(pred)
    print("realization", rl, "ratio at t>=1ns:", (meas/pred)[ts > 1e-9].round(3)[:6])
M = np.mean(rows_meas, 0); Pm = np.mean(rows_pred, 0)
err = np.std(rows_meas, 0)/np.sqrt(nreal)
sel = ts > 1.0e-9
ratio = M[sel]/Pm[sel]
print("mean ratio (t>1 ns):", ratio.mean(), "+-", (err[sel]/Pm[sel]).mean())
json.dump(dict(ts=ts.tolist(), meas=M.tolist(), pred=Pm.tolist(), err=err.tolist(), ratio_mean=float(ratio.mean()),
               ratio_se=float((err[sel]/Pm[sel]).mean()), nlines=nl, dk=dk, b_line=b, omega_b_single=omb1,
               spacing_delta=dk*pl.c*np.cos(0.2)), open(os.path.join(OUT, "val4_quasilinear.json"), "w"), indent=1)
