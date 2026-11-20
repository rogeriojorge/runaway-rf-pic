from val_red import *
ks, a = spectrum("gauss", 0.57, sig_j=4.5); bu = b_unit(0.57); Dd=1.672e8
s=2.2; b=bu*s**2
for df in (0.2, 0.1, 0.05, 0.025):
    r=run_S_red(ks,a,b,0.57,win_lines=5,Np=128,seed=1,Tfix=1e4/Dd,nrec=96,dtfrac=df)
    print(f"dtfrac={df}: m2 at t=1e3,1e4 beats", np.interp([1e3,1e4], r['ts']*Dd, r['m2']).round(4), " nst",r['nst'], flush=True)
