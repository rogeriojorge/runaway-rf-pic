from runPr import *
ks, a = spectrum("eq", 0.57, 5)
lam=1000.; NP=64
ls, g, F, kin = line_data(ks); ic=2
A0,_,_ = drift.sweep_rate(case, ls[ic]["kz"], g[ic], TH0, scale=lam); A=A0*Om**2
Dd=1.672e8; X=(2.5)*Dd+6*2.6e7; Xn=X/Om; T_n=2.1*Xn/A0; wb=2.6e7; dt=0.1/wb; Tsec=T_n/Om; nst=int(Tsec/dt); nrec=nst//64; nst=nrec*64
r=np.random.default_rng(1); theta=np.full(NP,TH0)
gam=solve_gamma(case, ls[ic]["kz"], np.full(NP,-Xn), theta); p=np.sqrt(gam**2-1)
red=Red(ks, 0.0*a); uz0=-p*np.cos(theta)
(gf,uzf,_,_,_),_=red.run(gam,uz0,r.uniform(0,6.28,NP),r.uniform(0,10,NP),np.zeros(5),dt,nst,nrec,lam=lam,drive=True)
u0v=np.stack([p*np.sin(theta),np.zeros(NP),uz0],1); uc=drift.propagate_control(case,u0v,Tsec*Om,scale=lam)
gc=np.sqrt(1+np.sum(uc**2,1))
print("A",A,"T",Tsec,"nst",nst,"gam0",gam[0],"red gf",np.mean(gf),"ctrl gf",np.mean(gc),"uz red",np.mean(uzf),"uz ctrl",np.mean(uc[:,2]))
dl0=np.mean(case.wn-ls[ic]["kz"]*uz0/gam-1/gam); dlr=np.mean(case.wn-ls[ic]["kz"]*np.asarray(uzf)/np.asarray(gf)-1/np.asarray(gf)); dlc=np.mean(case.wn-ls[ic]["kz"]*uc[:,2]/gc-1/gc)
print("delta start/red end/ctrl end (rad/s):",dl0*Om,dlr*Om,dlc*Om, " A*T=",A*Tsec)
