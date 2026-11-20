"""Exp B: one resonance passage of an accelerated runaway through a single coherent whistler line.

Physical drivers at DIII-D values (E = 8.5 E_c, collisional friction, JIC reduced Landau-Lifshitz radiation reaction),
time-compressed by lambda (all slow rates x lambda, amplitude mapped b_phys = b_sim/lambda at equal R = omega_b^2/|d delta/dt|).
Measured: <(xi_f(b) - xi_f(0))^2> per passage against the linear (quasilinear, stationary-phase) value.
usage: python3 expB_sweep.py LAMBDA OUTNAME b1 b2 ...
"""
import sys
from exp import *
import drift
lam = float(sys.argv[1]); name = sys.argv[2]; Rlist = [float(x) for x in sys.argv[3:]]   # target R = omega_b^2/|d delta/dt| at theta=0.2
case = bw.Case(); wn = case.wn
line = case.line(41.0); kzn = line["kz"]
N = int(os.environ.get("NPART", 256)); dtfrac = int(os.environ.get("DTFRAC", 32))
th_rng = tuple(float(x) for x in os.environ.get("THR", "0.1,0.3").split(","))
A0, dg0, _ = drift.sweep_rate(case, kzn, case.gamma_res(kzn, -1, 0.2), 0.2, scale=lam)      # rad/s^2 units Om^2
A = A0*case.Om**2
print(f"lambda={lam:g}: d(delta)/dt = {A:.3e} rad/s^2 ; sweep rate sqrt(A) = {np.sqrt(A):.3e}/s", flush=True)
E0z, fric = lam*case.EoverEc*case.Ec_n, lam*case.Ec_n
NUF = float(os.environ.get("NUF", 0.0))        # collision multiplier; 1 = DIII-D collisionality, Levy-scaled (nu_D x lambda^1.5 keeps Gamma_c^2/A fixed)
nu_scale = (case.Zeff+1)*case.nu0*lam**1.5*NUF
xi_b = th.omega_b(line, wn, case.gamma_res(kzn, -1, 0.2), 0.2, 1.0)**2        # omega_b^2 per unit b (Omega^2)
blist = [R*A0/xi_b for R in Rlist]
print("amplitudes b_sim:", ["%.3e" % b for b in blist], " b_phys = b_sim/lambda", flush=True)

def run_one(b, rec_n=48):
    om_b = th.omega_b(line, wn, case.gamma_res(kzn, -1, 0.3), 0.3, b)*case.Om            # largest over the theta range (rad/s)
    X = max(10*np.sqrt(A), 6*om_b)                                                       # rad/s
    Xn = X/case.Om
    T = 2.1*Xn/A0                                                                        # in 1/Omega
    dt = 2*np.pi*16.4/dtfrac
    nst = int(T/dt); nrec = max(nst//rec_n, 1); nst = nrec*rec_n
    lines, amps, _ = bw.make_lines(case, [41.0], [b], [0.0])
    x, u, theta0, g0 = build(case, line, N, th_rng, np.full(N, -Xn), 11)
    t0 = time.time()
    f = jax.jit(lambda x, u: bw.simulate((x, u, 0.0), nst, nrec, dt, wn, lines, amps, E0z=E0z, fric=fric, eps_rr=lam, rec_fn=rec_u, nu_scale=nu_scale, key=jax.random.PRNGKey(7)))
    st, rec = f(x, u)
    rec = np.asarray(rec)
    xi_ctl = s_ctl = None
    if nu_scale > 0:     # same ICs and noise key, no wave: collisional spread alone
        l0, a0, _ = bw.make_lines(case, [41.0], [0.0], [0.0])
        f0 = jax.jit(lambda x, u: bw.simulate((x, u, 0.0), nst, nst, dt, wn, l0, a0, E0z=E0z, fric=fric, eps_rr=lam, rec_fn=rec_u, nu_scale=nu_scale, key=jax.random.PRNGKey(7)))
        _, rec0 = f0(x, u)
        s_ctl, xi_ctl, _, _ = quantities(np.asarray(rec0)[-1])
    uc = drift.propagate_control(case, np.asarray(u), nst*dt, scale=lam)
    s_, xi, th_, g = quantities(rec)
    s0, xi0, th0_, g0_ = quantities(np.asarray(u))
    sc, xic, thc, gc = quantities(uc)
    return dict(b=b, X=X, om_b=om_b, nst=nst, T=nst*dt, time=time.time()-t0, xi_t=xi, xi0=xi0, xic=xic, thf=th_[-1], thc=thc, s_f=s_[-1], sc=sc,
                gf=g[-1], gc=gc, theta0=theta0, g0=g0, ts=(np.arange(1, rec_n+1)*nrec*dt), xi_ctl=xi_ctl, s_ctl=s_ctl, nuf=NUF)

import pickle
out = {}
for b in blist:
    r = run_one(b)
    out[b] = r
    dxi = r["xi_t"][-1]-r["xic"]
    print(f"b={b:.3e}: om_b={r['om_b']:.3e} X={r['X']:.3e} steps={r['nst']} time={r['time']:.0f}s  rms dxi={np.sqrt(np.mean(dxi**2)):.3e}", flush=True)
    pickle.dump(dict(lam=lam, A=A, A0=A0, out=out, th_rng=th_rng, N=N, dtfrac=dtfrac), open(os.path.join(OUT, f"expB_{name}.pkl"), "wb"))
