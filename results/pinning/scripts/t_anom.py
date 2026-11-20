"""Check A_n for n=-1 against a direct Boris test particle: initial resonant energy-gain rate vs |A_-1|."""
import os, sys
os.environ["JAX_ENABLE_X64"]="1"
sys.path.insert(0, os.path.expanduser("results/P3/scripts"))
import numpy as np, jax, jax.numpy as jnp
from p3lib import case, th, wn, bw
import qlib
ln = case.line(41.0)
b = 1e-6
for n, sg, theta in [(1,-1,0.3),(-1,+1,0.3),(-1,+1,0.1)]:
    gam = case.gamma_res(ln['kz'], sg, theta, n=n)
    p = np.sqrt(gam**2-1); up = p*np.sin(theta); upar = sg*p*np.cos(theta)
    # instantaneous gain rate maximised over gyrophase, at t=0 (z=0): scan phases, short integration T = 0.3 gyro periods? use A_n numeric (normalised units, E in B0c)
    E_unit = ln['E']  # physical per unit b (B0 c units); not unit-norm
    ph = np.linspace(0,2*np.pi,512,endpoint=False); zeta = ln['kx']*up
    beta = np.stack([up*np.cos(ph), up*np.sin(ph), np.full_like(ph, upar)],1)/gam
    A = abs(np.mean((beta@ln['E'])*np.exp(1j*zeta*np.sin(ph))*np.exp(-1j*n*ph)))
    # Boris: integrate N particles with random gyrophase and wave phase, measure energy gain over a window where resonant phase is frozen
    Np = 400; rng = np.random.default_rng(1)
    phi = rng.uniform(0,2*np.pi,Np); wph = rng.uniform(0,2*np.pi,Np)
    u = np.stack([up*np.cos(phi), up*np.sin(phi), np.full(Np, upar)],1)
    x = np.stack([up/gam*np.sin(phi)*0, np.zeros(Np), np.zeros(Np)],1)   # x offset irrelevant (absorbed in wave phase); gyro-centre at 0 => x = rho sin(phi)
    x[:,0] = up/gam*np.sin(phi)      # rho = u_perp/gamma in c/Omega units
    lines, aa, _ = bw.make_lines(case, [41.0]*1, [b], [0.0])
    # per-particle phase offset: shift z by wph/kz
    x[:,2] = wph/ln['kz']
    dt = 2*np.pi/64*gam; nst = 200; T = nst*dt
    st, rec = bw.simulate((jnp.asarray(x), jnp.asarray(u), 0.0), nst, 50, dt, wn, lines, aa, rec_fn=lambda x,u,t: bw.lorentz_gamma(u))
    g = np.asarray(rec); dg = g[-1]-gam
    # expected: dg(T) = (A b) * Re(...) -> max over phases ~ A b T if phase frozen (detuning small)
    delta = wn - ln['kz']*upar/gam - n/gam
    print(f"n={n} theta={theta} gam={gam:.3f} detuning={delta:.2e} T*delta={T*delta:.3f}  A*b*T={A*b*T:.3e}  Boris max|dg|={np.max(abs(dg)):.3e}  rms={np.sqrt(np.mean(dg**2)):.3e} (rms expected A b T/sqrt2={A*b*T/np.sqrt(2):.3e})")
