from val_red import *
ks, a = spectrum("gauss", 0.57, sig_j=4.5); bu = b_unit(0.57); Dd=1.672e8
b=bu*2.2**2
for df in (0.025, 0.0125):
    r=run_S_red(ks,a,b,0.57,win_lines=5,Np=128,seed=1,Tfix=1e4/Dd,nrec=96,dtfrac=df)
    print(f"long s=2.2 dtfrac={df}: m2 at 1e3,1e4:", np.interp([1e3,1e4], r['ts']*Dd, r['m2']).round(4), flush=True)
for s in (1.0, 1.6, 3.5, 6.0):
    b=bu*s**2; out=[]
    for df in (0.2, 0.1, 0.05, 0.025):
        rs=[run_S_red(ks,a,b,0.57,win_lines=5,Np=256,seed=sd,dtfrac=df) for sd in (1,2,3)]
        out.append(np.mean([r['ratio'] for r in rs]))
    print(f"onset sigma={s}: rho for dtfrac 0.2,0.1,0.05,0.025:", np.round(out,3), flush=True)
