"""V2/V3: resonance location and kick amplitude (linear regime) and trapping-island bounce frequency."""
import os, sys, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
import bw, plasma as pl, theory as th

case = bw.Case(); res = {}
kz_phys = 41.0
th0 = 0.2
b = 2e-9
N = 64
rng = np.random.default_rng(1)
line = case.line(kz_phys)
kzn = line["kz"]; wn = case.wn
gres = case.gamma_res(kzn, -1, th0)
Frad, P1, Q1 = th.frad(line, wn, gres, th0)
print("gamma_res", gres, "KE MeV", (gres-1)*0.511, "F_rad per b", Frad, "|P1|,|Q|", abs(P1), abs(Q1))
res["gamma_res"] = gres; res["F_rad_per_b"] = Frad
kin = th.kinematics(wn, kzn, gres, th0)
res["ddelta_dgamma"] = kin["ddelta"]
res["omega_b_over_sqrtb_per_s"] = th.omega_b(line, wn, gres, th0, 1.0)*case.Om
print("omega_b/sqrt(b) [1/s]", res["omega_b_over_sqrtb_per_s"])

lines, amps, ls = bw.make_lines(case, [kz_phys], [b], [0.0])
def make_ens(g, theta, n, seed):
    r = np.random.default_rng(seed)
    ph = (np.arange(n)+0.5)*2*np.pi/n if seed == 5 else r.uniform(0, 2*np.pi, n)   # stratified phases for V2
    p = np.sqrt(g**2-1)
    u = np.stack([p*np.sin(theta)*np.cos(ph), p*np.sin(theta)*np.sin(ph), -p*np.cos(theta)*np.ones(n)], 1)
    x = np.stack([p*np.sin(theta)*np.sin(ph) if seed == 5 else r.uniform(0, 60, n), np.zeros(n), np.zeros(n) if seed == 5 else r.uniform(0, 60, n)], 1)   # x_gc = 0
    return jnp.asarray(x), jnp.asarray(u)

# ---- V2: kick vs detuning, short time
T = 3.0e5   # 1/Omega  (1.2 ns)
out = {}
for dtfrac in (16, 32, 64):
    dt = 2*np.pi*gres/dtfrac
    nst = int(T/dt); nrec = max(nst//8, 1); nst = nrec*8
    dg_list = np.linspace(-3e-4, 3e-4, 41)*gres      # detuning scan
    pts = []
    for dg in dg_list:
        g = gres+dg
        x, u = make_ens(g, th0, N, 5)
        st, rec = bw.simulate((x, u, 0.0), nst, nrec, dt, wn, lines, amps, rec_fn=bw.std_rec)
        rec = np.asarray(rec)             # (8,N,6): g, uz, s, theta, phi, z
        u0perp = np.sqrt(np.asarray(u)[:, 0]**2+np.asarray(u)[:, 1]**2)
        du = np.sqrt(rec[-1, :, 2]) - u0perp
        # measured radial variance; theory: 1/2 F^2 b^2 sin^2(d t/2)/(d/2)^2
        t = nst*dt
        dl = th.delta(wn, kzn, g, -np.sqrt(g*g-1)*np.cos(th0))
        pred = 0.5*(Frad*b)**2*(np.sin(dl*t/2)/(dl/2))**2 if abs(dl) > 1e-15 else 0.5*(Frad*b*t)**2
        pts.append((float(dg/gres), float(dl), float(np.mean(du**2)), float(pred)))
    pts = np.array(pts)
    out[dtfrac] = pts.tolist()
    ok = pts[:, 3] > 0.05*pts[:, 3].max()
    ratio = pts[ok, 2]/pts[ok, 3]
    print(f"V2 dt=T/{dtfrac}: measured/predicted radial variance over main lobe: mean {ratio.mean():.3f} min {ratio.min():.3f} max {ratio.max():.3f}")
    # measured peak position
    imax = np.argmax(pts[:, 2]); ip = np.argmax(pts[:, 3])
    print("   peak at dgamma/gamma =", pts[imax, 0], "predicted peak", pts[ip, 0], "grid step", pts[1, 0]-pts[0, 0])
    res[f"V2_dt{dtfrac}"] = dict(ratio_mean=float(ratio.mean()), ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                                  peak_meas=float(pts[imax, 0]), peak_pred=float(pts[ip, 0]), grid=float(pts[1, 0]-pts[0, 0]))
res["V2_points_dt32"] = out[32]; res["V2_b"] = b

# ---- V3: bounce frequency from fastest small-amplitude oscillator in a resonant ensemble
bres = {}
for b3 in (1e-6, 1e-5, 1e-4, 1e-3):
    wb = th.omega_b(line, wn, gres, th0, b3)             # in Omega units
    lines3, amps3, _ = bw.make_lines(case, [kz_phys], [b3], [0.0])
    dt = 2*np.pi*gres/32
    Tsim = 12*2*np.pi/wb
    nst = int(Tsim/dt); nrec = max(nst//1024, 1); nst = nrec*1024
    x, u = make_ens(gres, th0, 96, 7)
    st, rec = bw.simulate((x, u, 0.0), nst, nrec, dt, wn, lines3, amps3, rec_fn=bw.std_rec)
    g = np.asarray(rec)[:, :, 0]                          # (1024, N)
    freqs = np.fft.rfftfreq(g.shape[0], d=nrec*dt)*2*np.pi
    sp = np.abs(np.fft.rfft(g - g.mean(0), axis=0))
    fpk = freqs[np.argmax(sp[1:], axis=0)+1]
    amp_pp = np.ptp(g, axis=0)
    # small-amplitude oscillators: smallest peak-to-peak among those with clear oscillation
    idx = np.argsort(amp_pp)[:12]
    f_small = np.median(fpk[idx]); f_max = fpk.max()
    # theoretical full island width: dgamma_max = 2*G/wb ... (pendulum separatrix half-height = 2 wb/|d delta|)
    bres[b3] = dict(omega_b_theory=float(wb), f_small_amp=float(f_small), f_max=float(f_max), ratio=float(f_small/wb),
                    island_halfwidth_gamma_theory=float(2*wb/abs(kin["ddelta"])), gamma_excursion_max=float(amp_pp.max()/2),
                    sim_over_period=float(Tsim*wb/(2*np.pi)))
    print("V3 b=%g  wb_theory=%.4e  f_small=%.4e  ratio=%.3f  max_excursion/halfwidth=%.3f" % (b3, wb, f_small, f_small/wb, (amp_pp.max()/2)/(2*wb/abs(kin['ddelta']))))
res["V3"] = bres
json.dump(res, open("../data/val2_resonance.json", "w"), indent=1)
