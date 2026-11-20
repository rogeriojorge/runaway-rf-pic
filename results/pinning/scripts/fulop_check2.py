"""Ex-only vs full-polarisation energy-balance growth rate against Fulop eq. 21, as function of angle and k (lnL=15, E/Ec=30)."""
import sys, json
from fulop_validation import *
import qlib
def run(k, thdeg, lnL=15., Eo=30., mode="full"):
    cZ=np.sqrt(18/np.pi)*lnL; alpha=(Eo-1)/2; th=np.deg2rad(thdeg)
    w,g21,g22=analytic(k,th,lnL,Eo); kpar,kperp=k*np.cos(th),k*np.sin(th)
    fg=fulop_f(cZ,alpha); pres=Oce/(kpar*c-w); pgrid=np.linspace(0.02,6*np.sqrt(pres/alpha),200)
    E=qlib.pol(w,kperp,kpar,sp); W1,_=wave_W(w,sp,kperp,kpar,E)
    out={}
    for mode in ("full","Ex"):
        Ec=E if mode=="full" else np.array([E[0],0,0],complex)
        integ=np.zeros(len(pgrid))
        for j,pp in enumerate(pgrid):
            gf=lambda pz: w-kpar*c*pz/np.sqrt(1+pz*pz+pp*pp)+Oce/np.sqrt(1+pz*pz+pp*pp)
            pzs=np.linspace(1,8*cZ+10,3001); gv=gf(pzs)
            for i in np.where(np.sign(gv[:-1])!=np.sign(gv[1:]))[0]:
                pz=brentq(gf,pzs[i],pzs[i+1]); gam=np.sqrt(1+pz*pz+pp*pp); dg=abs((gf(pz+1e-6)-gf(pz-1e-6))/2e-6)
                f,fz,fp=fg(pz,pp); dfdg=(kpar*c*fz-Oce/pp*fp)/w
                A=qlib.A_n(Ec,kperp,Oce,gam,1,np.arctan2(pp,pz),-1)
                integ[j]+=-(me*c**2/2)*np.pi*(e/(me*c))**2*A**2/dg*dfdg*2*np.pi*pp
        out[mode]=-np.trapezoid(integ,pgrid)/(2*W1)
    return w,g21,out["full"],out["Ex"]
if __name__=="__main__":
    rows=[]
    for thd in (85,88,89.5):
        for k in (100,150,250,400):
            w,g21,gf,gx=run(k,thd)
            rows.append(dict(theta=thd,k=k,eq21=g21,full=gf,Ex=gx))
            print(f"theta={thd} k={k} w/wci={w/wci:.1f}: eq21/wci={g21/wci:.4f} Ex-only/eq21={gx/g21:.3f} full/eq21={gf/g21:.3f}",flush=True)
    json.dump(rows,open("../data/fulop_check2.json","w"),indent=1)
