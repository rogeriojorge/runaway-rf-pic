"""Exp A: collisionless single coherent line, ensemble spread uniformly across the resonance.
<dxi^2>(t) against linear theory <lin_i(t)> for 15 amplitudes b = 1e-6 .. 1e-2.  No E, friction, RR or collisions."""
import sys, pickle
from exp import *
case = bw.Case(); wn = case.wn
line = case.line(41.0); kzn = line["kz"]
N = int(os.environ.get("NPART", 512)); th_rng = (0.1, 0.3)
bs = np.logspace(-6, -2, 15)
out = []
for ib, b in enumerate(bs):
    omb = th.omega_b(line, wn, case.gamma_res(kzn, -1, 0.3), 0.3, b)          # Omega units, largest over theta range
    omb2 = th.omega_b(line, wn, case.gamma_res(kzn, -1, 0.2), 0.2, b)
    X = 8*omb
    rng = np.random.default_rng(3)
    d0 = rng.uniform(-X, X, N)
    x, u, theta0, g0 = build(case, line, N, th_rng, d0, 21)
    tmax = 40/omb2
    dt = 2*np.pi*16.4/32
    nrec_t = 96
    # log-spaced record times via blocks of equal steps would waste; use linear 96 records
    nst = int(tmax/dt); nrec = max(nst//nrec_t, 1); nst = nrec*nrec_t
    lines, amps, _ = bw.make_lines(case, [41.0], [b], [0.0])
    t0 = time.time()
    st, rec = jax.jit(lambda x, u: bw.simulate((x, u, 0.0), nst, nrec, dt, wn, lines, amps, rec_fn=rec_u))(x, u)
    rec = np.asarray(rec)
    _, xi_t, th_t, g_t = quantities(rec)
    _, xi0, th0_, _ = quantities(np.asarray(u))
    ts = np.arange(1, nrec_t+1)*nrec*dt                      # 1/Omega
    dxi = xi_t - xi0[None, :]
    meas = np.mean(dxi**2, 1)
    # linear theory per particle
    F = frad_vec(line, wn, g0, theta0)
    dx = dxi_duperp(case, kzn, g0, theta0)
    dl = delta_of(case, kzn, g0, theta0)
    lin = np.array([np.mean(dx**2*0.5*(F*b)**2*(np.sin(dl*t/2)/(dl/2))**2) for t in ts])
    # golden-rule QL variance rate: plateau of lin(t)/t averaged over the window 1/X << t << 1/omb
    sel = (ts > 4/X) & (ts < 0.5/omb2)
    DQL = float(np.mean(lin[sel]/ts[sel]))/2*case.Om                  # <dxi^2>/(2t) in 1/s, QL
    row = dict(b=b, omega_b=omb2*case.Om, X=X*case.Om, ts=(ts/case.Om), meas=meas, lin=lin, DQL=DQL, nst=nst, time=time.time()-t0,
               kurt={f"{k}": float(np.mean((dxi[np.argmin(abs(ts*omb2-k))]-dxi[np.argmin(abs(ts*omb2-k))].mean())**4)/np.var(dxi[np.argmin(abs(ts*omb2-k))])**2) for k in (0.3, 3, 30)},
               xi_final_hist=np.histogram(dxi[np.argmin(abs(ts*omb2-3))], bins=41)[0], dxi_t3=dxi[np.argmin(abs(ts*omb2-3))])
    out.append(row)
    i3 = np.argmin(abs(ts*omb2-3)); i03 = np.argmin(abs(ts*omb2-0.3)); i30 = np.argmin(abs(ts*omb2-30))
    print(f"b={b:.2e} wb={omb2*case.Om:.2e} steps={nst} {row['time']:.0f}s | meas/lin at wb t=0.3: {meas[i03]/lin[i03]:.3f}, 3: {meas[i3]/lin[i3]:.3f}, 30: {meas[i30]/lin[i30]:.3f}", flush=True)
    pickle.dump(dict(out=out, N=N), open(os.path.join(OUT, "expA.pkl"), "wb"))
