import sys, os, time
sys.path.insert(0, os.path.expanduser("results/P2/scripts"))
from exp import *
case = bw.Case(); wn = case.wn
for L in (1, 8, 24):
  for N in (256,):
    ks = 41.0 + 0.57*(np.arange(L)-L//2)
    lines, amps, _ = bw.make_lines(case, list(ks), [1e-4]*L, np.random.rand(L)*6.28)
    r = np.random.default_rng(1)
    g = 16.4+r.uniform(-.3,.3,N); th=0.2; p=np.sqrt(g**2-1); phi=r.uniform(0,6.28,N)
    u = np.stack([p*np.sin(th)*np.cos(phi), p*np.sin(th)*np.sin(phi), -p*np.cos(th)*np.ones(N)],1)
    x = np.stack([p*np.sin(th)*np.sin(phi), np.zeros(N), r.uniform(0,100,N)],1)
    dt = 2*np.pi*16.4/32
    nst=20000
    f = jax.jit(lambda x,u: bw.simulate((x,u,0.0), nst, nst, dt, wn, lines, amps, rec_fn=rec_u))
    f(jnp.asarray(x), jnp.asarray(u))[0][1].block_until_ready()
    t=time.time(); f(jnp.asarray(x), jnp.asarray(u))[0][1].block_until_ready(); T=time.time()-t
    print(L,N,nst,T, "particle-steps/s %.2e"%(N*nst/T), "line-steps/s %.2e"%(N*nst*L/T), flush=True)
