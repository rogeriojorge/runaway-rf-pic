"""Item 4: what sets the 'plateau' D_eff/D_QL below the crossover?  Reduced model in the linear regime (R<<1), passage with window X = Xs sqrt(A), initial pitch spread
theta in [th_lo, th_hi] AT THE START (as in P2).  Measured <dg^2> compared with (a) QL evaluated with the start pitch (P2 convention), (b) QL evaluated with each particle's own
state at the resonance crossing (stationary phase with local c = G F b Om and local A), (c) full linear response along the actual control orbit (windowed Fresnel).
usage: plateau2.py LAM Xs th_lo th_hi NP"""
from pc2 import *
import red_env
lam = float(sys.argv[1]); Xs = float(sys.argv[2]); tlo = float(sys.argv[3]); thi = float(sys.argv[4]); NP = int(sys.argv[5])
FR = int(os.environ.get("FRIC", 1)); RRF = int(os.environ.get("RR", 1))
S = setup("eqN1", lam, 0.02, NP=NP, Xs=Xs, Xw=0.0, ic="start", fric=FR, rr=RRF)
r = np.random.default_rng(5); theta = r.uniform(tlo, thi, NP)
kz = S["ls"][0]["kz"]; a = wn/kz
gam = solve_gamma(case, kz, np.full(NP, -S["Xn"]), theta); p = np.sqrt(gam**2-1)
dtf = 0.03*np.sqrt(S["R"]); dt = dtf/S["wb"]; Tsec = S["Tn"]/Om; nst = int(Tsec/dt); nst = (nst//1)*1
red = red_env.Red(S["ks"], S["b"]*S["a"]); red0 = red_env.Red(S["ks"], 0.0*S["a"])
for rd in (red, red0): rd.use_fric = float(FR); rd.use_rr = float(RRF)
uz0 = -p*np.cos(theta); phi = r.uniform(0, 2*np.pi, NP); z0 = np.zeros(NP)
run = lambda rd, rec: jax.jit(lambda: rd.run(gam, uz0, phi, z0, np.zeros(1), dt, nst, rec, lam=lam, nuf=0.0, drive=True, key=0))()
(gf, *_), _ = run(red, nst); (gc, *_), recs0 = run(red0, 1)
dg = np.asarray(gf)-np.asarray(gc)
recs0 = np.asarray(recs0)                        # (nst, NP, 2): gamma, uz
g_t = np.concatenate([gam[None], recs0[:, :, 0]]); uz_t = np.concatenate([uz0[None], recs0[:, :, 1]])
tt = np.arange(g_t.shape[0])*dt
dl = (wn - kz*uz_t/g_t - 1/g_t)*Om; up = np.sqrt(np.maximum(g_t**2-1-uz_t**2, 1e-12)); th_t = np.arctan2(up, -uz_t)
Fv = np.asarray(red0.Fc(jnp.asarray(g_t.ravel()), jnp.asarray(th_t.ravel()))).reshape(g_t.shape)
c_t = up/(g_t-uz_t/a)*Fv*S["b"]*Om
ic = np.argmin(np.abs(dl), 0); ar = np.arange(NP)
A_i = np.array([np.gradient(dl[:, k], tt)[ic[k]] for k in range(NP)]); c_i = c_t[ic, ar]; th_c = th_t[ic, ar]
ql_local = np.pi*c_i**2/A_i
# nominal QL with F, G, A at the start pitch (P2 / P3 convention: resonance at the *initial* theta)
gres = np.array([case.gamma_res(kz, -1, t_) for t_ in theta]); thn = theta
Fn = np.asarray(red0.Fc(jnp.asarray(gres), jnp.asarray(thn))); upn = gres*0+np.sqrt(gres**2-1)*np.sin(thn); uzn = -np.sqrt(gres**2-1)*np.cos(thn)
c_n = upn/(gres-uzn/a)*Fn*S["b"]*Om; ql_nom = np.pi*c_n**2/S["A"]
# linear response along the actual orbit (per particle, averaged over uniform initial phase): 1/2 |int c e^{-i Phi} dt|^2
Phi = np.concatenate([np.zeros((1, NP)), np.cumsum(0.5*(dl[1:]+dl[:-1])*np.diff(tt)[:, None], 0)])
I = np.trapezoid(c_t*np.exp(-1j*Phi), tt, axis=0); lin = 0.5*np.abs(I)**2
print(f"lam={lam:g} Xs={Xs:g} theta0 in [{tlo},{thi}] NP={NP}  R~{S['R']:.3g}")
print(f"  mean theta start {theta.mean():.4f}, at crossing {th_c.mean():.4f} (shift {th_c.mean()-theta.mean():+.4f}); A_i/A_nom mean {np.mean(A_i)/S['A']:.4f}")
print(f"  measured/QL(start-pitch, nominal A)      = {np.mean(dg**2)/np.mean(ql_nom):.4f}")
print(f"  measured/QL(local c,A at crossing)       = {np.mean(dg**2)/np.mean(ql_local):.4f}")
print(f"  linear response (actual orbit)/QL(local) = {np.mean(lin)/np.mean(ql_local):.4f};  measured/linear response = {np.mean(dg**2)/np.mean(lin):.4f}")
print(f"  QL(local)/QL(nominal) = {np.mean(ql_local)/np.mean(ql_nom):.4f}")
