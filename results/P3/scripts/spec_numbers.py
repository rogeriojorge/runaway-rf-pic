import numpy as np
c=2.99792458e8; f=476e6; w=2*np.pi*f; k0=w/c
Rant=2.27; arc=np.deg2rad(40)*Rant
print(f"k0=w/c={k0:.3f}/m; n_par=3 -> k_par,ant={3*k0:.2f}/m (lambda={2*np.pi/(3*k0)*100:.1f} cm); n_phi(ant)=k_par R={3*k0*Rant:.1f}")
print(f"antenna arc 40deg at R=2.27 = {arc:.3f} m; 30 modules -> pitch {arc/30*100:.1f} cm = {arc/30/(2*np.pi/(3*k0)):.2f} lambda_par per module")
for R in (1.6,1.75,2.0,2.27):
    print(f"R_res={R}: k_par = {3*k0*Rant/R:.1f}/m, line spacing 1/R = {1/R:.3f}/m")
# finite aperture spectra (width in n_par)
L=arc; 
print("uniform aperture: FWHM dk=0.886*2pi/L =",0.886*2*np.pi/L,"/m -> dn_par FWHM",0.886*2*np.pi/L/k0, " sigma_n_par=",0.886*2*np.pi/L/k0/2.355)
print("GENRAY: sigma n_par=0.2 -> sigma_k(ant)=",0.2*k0,"/m -> sigma in line index (n_phi) =",0.2*k0*Rant)
print("uniform aperture sigma in n_phi:", 0.886*2*np.pi/L/2.355*Rant)
# half-length (comb-line power decaying along antenna: effective aperture ~ L/2)
print("effective aperture L/2: sigma n_phi", 0.886*2*np.pi/(L/2)/2.355*Rant)
for s in (3.4,4.5,7.0): print("sigma_j",s,"N_eff=sum a_j^2 =",np.sqrt(2*np.pi)*s)
print("10 MHz band -> equivalent dk_par = 2pi*1e7/c_ =",2*np.pi*1e7/c, "/m at v=c; Gamma_phi for 3MHz FWHM:",np.pi*3e6,"/s; 10 MHz:",np.pi*1e7)
