from val_red import *
ks, a = spectrum("gauss", 0.57, sig_j=4.5); bu = b_unit(0.57)
Dd=1.672e8
for s in (0.9, 1.2, 1.6, 2.2, 3.0):
    b=bu*s**2
    rs=[run_S_red(ks,a,b,0.57,win_lines=5,Np=256,seed=sd,Tfix=400/Dd,nrec=96) for sd in range(1,5)]
    m2=np.mean([r['m2'] for r in rs],0); ts=rs[0]['ts']; D=rs[0]['Dql']
    rho_t=m2/(D*ts)
    idx=[min(95,int(np.searchsorted(ts*Dd,x))) for x in (5,10,20,50,100,200,400)]
    print(f"sig={s} rho(t)=m2/(D t) at beats 5,10,20,50,100,200,400:", np.round(rho_t[idx],2), " m2_end",round(m2[-1],3), flush=True)
