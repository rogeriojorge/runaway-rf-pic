"""Quasilinear (Fokker-Planck) reference for a dense, smoothed comb with Lorentzian wings.
d f/d tau = (1/2) d/dx ( D(x) d f/dx ),  x = (gamma - gamma_band_start)/Lw, tau = Dql t / Lw^2 (so D = 1 inside a dense infinite comb).
D(x) = 0.1 * sum_{j=0..9} f_G(x - x_j), f_G Lorentzian of HWHM G (band units) normalised to 1, x_j = (j+0.5)/10, G = Gref/10 with Gref = max(Gamma/Delta, 1).
Domain [-W, 1+W] with reflecting ends.  Returns tau grid and m(tau) = <(x-x0)^2>, x0 uniform on [0,1] (in Lw^2 units)."""
import numpy as np, os
def ql_fp(Gref, W=3.0, N=700, n0=120, nt=160, taumax=40.0):
    cf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"qlfp_G{Gref:g}.npz")
    if os.path.exists(cf): z = np.load(cf); return z["tau"], z["m"]
    xs = np.linspace(-W, 1+W, N); h = xs[1]-xs[0]; xh = 0.5*(xs[1:]+xs[:-1])
    G = Gref/10.0; xj = (np.arange(10)+0.5)/10
    D = 0.1*np.sum((G/np.pi)/((xh[:, None]-xj[None, :])**2+G**2), 1)
    A = np.zeros((N, N))
    for i in range(N-1):
        c = 0.5*D[i]/(h*h); A[i, i] -= c; A[i+1, i+1] -= c; A[i, i+1] += c; A[i+1, i] += c
    lam, V = np.linalg.eigh(A)
    x0i = np.array([int(np.argmin(abs(xs-x))) for x in (np.arange(n0)+0.5)/n0])
    taus = np.geomspace(1e-4, taumax, nt); out = []
    d2 = (xs[:, None]-xs[x0i][None, :])**2
    for t in taus:
        P = (V*np.exp(lam*t))@V[x0i, :].T          # P[x, x0] (symmetric generator; columns are densities up to h)
        out.append(np.mean(np.sum(P*d2, 0)))
    m = np.array(out); np.savez(cf, tau=taus, m=m); return taus, m
if __name__ == "__main__":
    for G in (1.0, 3.0):
        t, m = ql_fp(G); print(G, m[::20], t[::20])
