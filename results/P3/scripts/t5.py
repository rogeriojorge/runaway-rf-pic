import pickle
from val_red import *
ks, a = spectrum("gauss", 0.57, sig_j=4.5); bu = b_unit(0.57); Dd=1.672e8
out={}
for s in (0.9, 1.0, 1.2, 1.4, 1.7, 2.2, 3.0):
    b=bu*s**2
    res=[]
    for lb in (1e3, 1e4, 1e5):
        rs=[run_S_red(ks,a,b,0.57,win_lines=5,Np=256,seed=sd,Tfix=lb/Dd,nrec=96) for sd in range(1,4)]
        m2=np.mean([r['m2'] for r in rs],0); ts=rs[0]['ts']; D=rs[0]['Dql']
        res.append((lb,ts,m2,D))
    out[s]=res
    print(f"sig={s}: m2 at beats 1e3,1e4,1e5:", [f"{r[2][-1]:.3f}" for r in res], " Dql*T:", [f"{r[3]*r[1][-1]:.3f}" for r in res], flush=True)
pickle.dump(out,open("../data/long_gauss45.pkl","wb"))
