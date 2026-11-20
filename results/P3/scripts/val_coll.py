"""Check the Langevin pitch operator without a wave: <xi>(t)=xi0 exp(-nu t), <xi^2>=1/3+(xi0^2-1/3)exp(-3 nu t), |p| conserved.
nu_D = (Z+1) gamma/(p^3 tau_c) (P1 convention, Lorentz operator nu/2 d/dxi (1-xi^2) d/dxi); multiplier nuf applied to the DIII-D value."""
import json
from p3lib import *
Np = 8192; g0 = 16.4; p0 = np.sqrt(g0**2-1)
nuD_phys = (case.Zeff+1)*g0/p0**3/case.tau_c            # 1/s, DIII-D value at this energy
nuf = 4e6
nu = nuD_phys*nuf
for xi0 in (-0.98, 0.0):
    u0 = np.zeros((Np, 3)); u0[:, 2] = p0*xi0; u0[:, 0] = p0*np.sqrt(1-xi0**2)
    x0 = np.zeros((Np, 3))
    nst, nrec = 200000, 20
    lines, aa, _ = bw.make_lines(case, [41.0], [0.0], [0.0])
    f = jax.jit(lambda x, u: bw.simulate((x, u, 0.0), nst, nst//nrec, DT, wn, lines, aa, rec_fn=rec_u, nu_scale=(case.Zeff+1)*case.nu0*nuf, key=jax.random.PRNGKey(3)))
    st, rec = f(jnp.asarray(x0), jnp.asarray(u0)); rec = np.asarray(rec)
    ts = np.arange(1, nrec+1)*(nst//nrec)*DT/Om
    p = np.sqrt(np.sum(rec**2, -1)); xi = rec[..., 2]/p
    out = dict(nuD_phys=nuD_phys, nu=nu, xi0=xi0, pmax_dev=float(np.abs(p/p0-1).max()))
    print(f"xi0={xi0}: nu_D(DIII-D)={nuD_phys:.4f}/s, x{nuf:g} = {nu:.3e}/s, nu*T_end={nu*ts[-1]:.3f}, max|dp/p|={out['pmax_dev']:.1e}")
    m1 = xi.mean(1); m2 = (xi**2).mean(1)
    a1 = xi0*np.exp(-nu*ts); a2 = 1/3+(xi0**2-1/3)*np.exp(-3*nu*ts)
    for i in (0, 4, 9, 14, 19):
        print(f"  nu t={nu*ts[i]:.3f}: <xi>={m1[i]:+.4f} (theory {a1[i]:+.4f})  <xi^2>={m2[i]:.4f} (theory {a2[i]:.4f})")
    out.update(nu_t=(nu*ts).tolist(), m1=m1.tolist(), th1=a1.tolist(), m2=m2.tolist(), th2=a2.tolist())
    json.dump(out, open(f"../data/val_coll_xi{xi0}.json", "w"))
