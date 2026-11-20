import os, sys
os.environ["JAX_ENABLE_X64"]="1"
sys.path.insert(0, os.path.expanduser("results/P3/scripts"))
import numpy as np
from scipy.special import jv, jvp
from p3lib import case, th, wn, TH0
ln = case.line(41.0)
for gam, theta in [(16.3,0.2),(16.3,0.4),(18.,0.3)]:
    gam = case.gamma_res(ln['kz'],-1,theta)
    F = th.frad(ln, wn, gam, theta)[0]; k = th.kinematics(wn, ln['kz'], gam, theta)
    G = k['G_per_F']
    p=np.sqrt(gam**2-1); uperp=p*np.sin(theta); upar=-p*np.cos(theta)
    # numeric gyro average of dgamma/dt = -E.v ; harmonic n=1
    ph=np.linspace(0,2*np.pi,512,endpoint=False); zeta=ln['kx']*uperp
    v=np.stack([uperp*np.cos(ph),uperp*np.sin(ph),np.full_like(ph,upar)],1)/gam
    Ev=np.einsum('i,ji->j',ln['E'],v)
    for n in (1,):
        A=np.mean(Ev*np.exp(1j*zeta*np.sin(ph))*np.exp(-1j*n*ph))
        # Pokol psi (two sign options)
        Ex,Ey,Ez=ln['E']
        for s in (1,-1):
            psi=Ex*n/zeta*jv(n,zeta)+1j*s*Ey*jvp(n,zeta)+(upar/uperp)*Ez*jv(n,zeta)
            print(f"gam={gam:.3f} th={theta} n={n} G*F={G*F:.5f} |A|={abs(A):.5f} |uperp/gam*psi| s={s}: {uperp/gam*abs(psi):.5f}")
