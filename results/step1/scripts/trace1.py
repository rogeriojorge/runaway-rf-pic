"""Time traces dg(t)=g_wave-g_b0 for a few particles, Boris (N=1, lam, sigma) and matching reduced run."""
from pcommon import *
name = sys.argv[1]; lam = float(sys.argv[2]); sig = float(sys.argv[3]); NP = 8; nrec = 400
kind, Delta, kw = REG[name]; ks, a = spectrum(kind, Delta, **kw)
ls, g, F, kin = line_data(ks); L = len(ks); ic = L//2
G = np.array([k["G_per_F"] for k in kin])
A0, _, _ = drift.sweep_rate(case, ls[ic]["kz"], g[ic], TH0, scale=lam); A = A0*Om**2
Dd = 1e9 if L == 1 else abs(ls[1]["kz"]-ls[0]["kz"])*np.sqrt(g[ic]**2-1)*np.cos(TH0)/g[ic]*Om
b = b_unit(Delta)*sig**2
wb = np.sqrt(G*F*b*a*np.array([abs(k["ddelta"]) for k in kin]))[ic]*Om
X = ((L-1)/2+0.5)*(Dd if L > 1 else 0) + max(10*np.sqrt(A), 6*wb); Xn = X/Om; T = 2.1*Xn/A0
nst = int(T/DT); nst = (nst//nrec)*nrec
r = np.random.default_rng(11); theta = r.uniform(TH0-0.01, TH0+0.01, NP)
gam = solve_gamma(case, ls[ic]["kz"], np.full(NP, -Xn), theta); p = np.sqrt(gam**2-1); phi = r.uniform(0, 2*np.pi, NP)
z0 = r.uniform(0, 2*np.pi/ls[0]["kz"] if L == 1 else 2*np.pi/(Delta*pl.c/Om), NP)
u = np.stack([p*np.sin(theta)*np.cos(phi), p*np.sin(theta)*np.sin(phi), -p*np.cos(theta)], 1); x = np.stack([p*np.sin(theta)*np.sin(phi), np.zeros(NP), z0], 1)
lines, aa, _ = bw.make_lines(case, list(ks), list(b*a), r.uniform(0, 2*np.pi, L))
E0z, fric = lam*case.EoverEc*case.Ec_n, lam*case.Ec_n
def rec(x, u, t): return jnp.stack([bw.lorentz_gamma(u), u[:, 2]], -1)
def run(am):
    fn = jax.jit(lambda x, u: bw.simulate((x, u, 0.0), nst, nst//nrec, DT, wn, lines, am, E0z=E0z, fric=fric, eps_rr=lam, rec_fn=rec))
    return np.asarray(fn(jnp.asarray(x), jnp.asarray(u))[1])
rw, r0 = run(aa), run(0*aa)
t = np.arange(1, nrec+1)*(nst//nrec)*DT/Om
np.savez(f"../data/trace_{name}_lam{lam:g}_s{sig}.npz", t=t, gw=rw[..., 0], g0=r0[..., 0], uzw=rw[..., 1], uz0=r0[..., 1], wb=wb, A=A, X=X, theta=theta, phi=phi)
dg = rw[..., 0]-r0[..., 0]
# detuning of control orbit vs time
dl = case.wn - ls[ic]["kz"]*r0[..., 1]/r0[..., 0] - 1/r0[..., 0]
i_res = np.argmin(abs(dl.mean(1)))
print("t_res index", i_res, "of", nrec, " omega_b", wb, " X/wb", X/wb)
for k in range(NP):
    print(k, "dg at 1/4,1/2,res-10,res+10,3/4,end:", [f"{dg[j,k]:+.4f}" for j in (nrec//4, nrec//2, max(i_res-10,0), min(i_res+10, nrec-1), 3*nrec//4, nrec-1)], "max|dg|", f"{abs(dg[:,k]).max():.4f}")
