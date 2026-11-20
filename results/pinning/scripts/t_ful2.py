from fulop_validation import *
import qlib
Eo=30.; lnL=15.; cZ=np.sqrt(18/np.pi)*lnL; alpha=(Eo-1)/2
th=np.deg2rad(85.)
for k in (150,350,550):
    w,g21,g22=analytic(k,th,lnL,Eo); kpar,kperp=k*np.cos(th),k*np.sin(th)
    E=qlib.pol(w,kperp,kpar,sp); print("k",k,"E",np.round(E,4), "|E|",np.linalg.norm(E))
    fg=fulop_f(cZ,alpha); pres=Oce/(kpar*c-w); pgrid=np.linspace(0.02,6*np.sqrt(pres/alpha),160)
    P,Wt,g_full=gain_power(w,kperp,kpar,sp,Oce,-1,fg,pgrid,(1.0,8*cZ+10),E=E)
    # Ex-only coupling
    Ex=E.copy(); 
    W1,_=wave_W(w,sp,kperp,kpar,E)
    def gp_Ex():
        Eonly=np.array([E[0],0,0],complex)
        # monkeypatch pol to use full E for W but Eonly for coupling: reimplement quickly
        integ=np.zeros(len(pgrid))
        for j,pp in enumerate(pgrid):
            gfun=lambda pz: w-kpar*c*pz/np.sqrt(1+pz*pz+pp*pp)+Oce/np.sqrt(1+pz*pz+pp*pp)
            pzs=np.linspace(1,8*cZ+10,3001); gv=gfun(pzs)
            for i in np.where(np.sign(gv[:-1])!=np.sign(gv[1:]))[0]:
                pz=brentq(gfun,pzs[i],pzs[i+1]); gam=np.sqrt(1+pz*pz+pp*pp); dg=abs((gfun(pz+1e-6)-gfun(pz-1e-6))/2e-6)
                f,fz,fp=fg(pz,pp); dfdg=(kpar*c*fz-Oce/pp*fp)/w
                A=qlib.A_n(Eonly,kperp,Oce,gam,1,np.arctan2(pp,pz),-1)
                integ[j]+=-(me*c**2/2)*np.pi*(e/(me*c))**2*A**2/dg*dfdg*2*np.pi*pp
        P=np.trapezoid(integ,pgrid); return -P/(2*W1)
    print(f"k={k}: eq21/wci={g21/wci:.3f} full={g_full/wci:.3f}  Ex-only={gp_Ex()/wci:.3f}")
