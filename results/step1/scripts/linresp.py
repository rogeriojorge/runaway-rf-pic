"""Linear response along the actual (reduced) control trajectory: <dg^2> = 1/2 |int c(t) exp(-i Phi(t)) dt|^2 with c = G F b Om, Phi = int delta dt.
Compares with the QL stationary-phase formula and the reduced-model measurement.  usage: linresp.py Xs sigma"""
from pc2 import *
import red_env
xs = float(sys.argv[1]); sg = float(sys.argv[2])
S = setup("eqN1", 1000, sg, NP=1, Xs=xs, Xw=0.0, ic="cross"); ql = ql_value(S)
dtf = 0.02*np.sqrt(S["R"]); dt = dtf/S["wb"]; Tsec = S["Tn"]/Om; nst = int(Tsec/dt); nst = (nst//64)*64
kz = S["kzc"] if "kzc" in S else case.line(S["ks"][0])["kz"]
red0 = red_env.Red(S["ks"], 0.0*S["a"])
uz0 = -S["p"]*np.cos(S["theta"]); z0 = S["z0n"]*pl.c/Om
(_, _, *_), recs = jax.jit(lambda: red0.run(S["gam"], uz0, S["phi"], z0, np.zeros(1), dt, nst, 1, lam=S["lam"], nuf=0.0, drive=True, key=0))()
recs = np.asarray(recs)[:, 0, :]; g = np.concatenate([[S["gam"][0]], recs[:, 0]]); uz = np.concatenate([[uz0[0]], recs[:, 1]])
tt = np.arange(len(g))*dt; up = np.sqrt(np.maximum(g*g-1-uz*uz, 1e-12)); th = np.arctan2(up, -uz)
dl = (case.wn - kz*uz/g - 1/g)*Om
Fv = np.asarray(red0.Fc(jnp.asarray(g), jnp.asarray(th)))
a = case.wn/kz; G = up/(g-uz/a)
c = G*Fv*S["b"]*Om
Phi = np.concatenate([[0], np.cumsum(0.5*(dl[1:]+dl[:-1])*np.diff(tt))])
I = np.trapezoid(c*np.exp(-1j*Phi), tt)
i_c = np.argmin(abs(dl)); slope = np.gradient(dl, tt)[i_c]
print(f"Xs={xs} sigma={sg} R={S['R']:.3g}: slope at crossing {slope:.4e} vs S[A] {S['A']:.4e};  c(cross)={c[i_c]:.4e} vs sqrt(pi)... QL-central c {np.sqrt(ql*S['A']/np.pi):.4e}")
print(f"  linear-response over QL-central = {0.5*abs(I)**2/ql:.4f};  gamma at crossing {g[i_c]:.4f}, theta {th[i_c]:.4f}; delta range/sqrtA {dl[0]/np.sqrt(S['A']):.2f}..{dl[-1]/np.sqrt(S['A']):.2f}")
# stationary-phase estimate with local c, A
print(f"  stationary-phase with local c,A: {np.pi*c[i_c]**2/abs(slope)/ql:.4f}")
