"""Physical Gamma_phi estimate for the DIII-D helicon (numbers quoted in NOTES step 3).  Gamma = HWHM of the Lorentzian line (rad/s), field coherence exp(-Gamma |tau|)."""
import numpy as np
Dd = 1.67e8            # beat angular frequency at Delta k = 0.57 /m (step1), 2pi/Dd = 37.6 ns
GP = 2.7e5             # step2 median power damping rate [1/s]; 95% range 4e4 - 1.4e6
tlife = 1/GP
print(f"beat period {2*np.pi/Dd*1e9:.1f} ns; wave lifetime 1/Gamma_P = {tlife*1e6:.1f} us = {tlife/(2*np.pi/Dd):.0f} beat periods")
# (a) source linewidth.  Relative phase of two lines that reached a point by paths of delay difference dtau: var = 2 Gs |dtau| (white frequency noise).
# Lines decorrelate (Gamma_phi ~ Gs) only if Gs*dtau >~ 1 with dtau up to the lifetime; common-mode phase is irrelevant.
print("source linewidth needed for dtau ~ lifetime: Gs >~ %.2e rad/s = %.1f kHz (HWHM/2pi)" % (GP, GP/2/np.pi/1e3))
for lw_hz in (1, 1e3, 1e4, 1e5, 1e6):
    Gs = 2*np.pi*lw_hz; print(f"  source HWHM {lw_hz:8.0e} Hz -> Gamma/Dd = {Gs/Dd:.1e};  Gs*lifetime = {Gs*tlife:.2e}")
for fm in (1e6, 3e6, 5e6, 10e6):
    print(f"  FM / noise HWHM {fm/1e6:.0f} MHz (2pi f) -> Gamma/Dd = {2*np.pi*fm/Dd:.2f}")
# (b) plasma density fluctuation phase noise (fast wave k_perp = 70/m)
kperp = 70.0
print("density-fluctuation phase: dphi_rms = kperp L (dn/n)/2 per pass")
for L in (1.0, 2.0):
    for dn in (0.003, 0.01, 0.03):
        dphi = kperp*L*dn/2
        for tc in (1e-6, 1e-5, 1e-4):
            G = dphi**2/tc if tc < tlife else dphi**2/tc   # random walk rate if tc << life; if tc >> life phase is frozen on the lifetime
            print(f"  L={L} dn/n={dn:.3f} dphi={dphi:.2f} rad tau_c={tc:.0e}s  Gamma~dphi^2/tau_c={G:.1e}/s -> /Dd={G/Dd:.1e}  (frozen over life: {tc>tlife})")
# (c) PDF side bands at the deuterium cyclotron spacing (8 MHz): lines at w0 +- m Omega_D, spacing in angular frequency
OD = 2*np.pi*7.6e6; print(f"PDI sideband spacing 2pi*7.6 MHz = {OD:.2e} rad/s = {OD/Dd:.2f} Dd; a broadened pump of width ~ spacing would give Gamma/Dd ~ {OD/Dd:.2f}")
