# P2 notes: single-wave scattering of runaways at DIII-D parameters (go/no-go)

Scripts: `scripts/` (all JAX_ENABLE_X64=1, 4 threads, laptop only). Data: `data/` (8 MB). Nothing committed or pushed.
JIC clone for reading: `~/local/jic-work/P2/JAX-in-Cell` (branch quantum-recoil). No P1 files touched.

## Verdict: GO (conditional) for the narrowed thesis. The kill criterion is not met.
- A single coherent line does NOT follow quasilinear (QL) theory at DIII-D or Kang amplitudes. It departs from QL at
  R = omega_b^2/|d delta/dt| of about 2-3, i.e. b_x ~ 3.5e-9 for the DIII-D acceleration sweep (E = 8.5 Ec, friction, RR).
  DIII-D (3e-5 to 2e-3), Guo (1e-5 to 1e-4) and Kang (4.75e-4) are all 4-6 decades above that.
- Sign matters: for resonant pitch theta >= 0.1 trapping makes the passage scatter LESS than QL, D_eff/D_QL ~ (1.5 to 7)/R,
  about 1e-3 to 1e-5 at DIII-D b. This is opposite to Kang's "1e6 faster". For small pitch (theta 0.02-0.08, xi ~ -1)
  there is an enhancement window above QL (x23 at lambda=1e3, x8 at lambda=100, shrinking with lambda; physical lambda=1 not run)
  near R = 1-10, then suppression again.
- Which amplitude is "the crossover" depends on the decorrelation mechanism, b_x = R_c Gamma^2/Xi (Xi = omega_b^2/b = 1.77e19 s^-2).
  QL is recovered only if the particle sees decorrelation Gamma >= omega_b (4e7/s at b=1e-4). This is NOT supplied by E-sweep
  (1.6e5/s) or collisions (~5e5/s, analytic estimate). It is supplied by a random-phase spectrum or short transit (>= 1e8/s).
  So whether DIII-D is trapped or QL is decided by spectral coherence, which is exactly the overlap question (P3).
- Not a PRL yet by itself: the single-line law is a known pendulum-with-sweep scaling (Fajans-Friedland, Karney-Fisch type);
  what is new is the quantitative DIII-D map plus the line-overlap law (two-line result below). Ambiguity flagged in "Limits".

## Method (independent of JIC L4)
`bw.py`: relativistic Boris in B0 z-hat plus plane wave(s), time 1/Omega, tan-exact rotation. Fields from cold plasma
(`plasma.py`, electrons + D): n_e = 0.8e19 and B=1.4 T reproduce Choudhury's k_perp = 76.6/m at k_par = 41/m (0.5e19 gives 36/m).
Resonance (n=1, v_par<0): gamma = 16.1 (theta=0), 16.4 (theta=0.2): 7.7-7.9 MeV. Extras: static E_par, friction, Langevin pitch scattering
(nu_D=(Z+1)gamma/(p^3 tau_c)), radiation reaction = JIC `landau_lifshitz` (my first reduced form was wrong by gamma and was dropped; integrator RR dgamma/dt = -4.02/s = JIC = AB estimate).
Amplitude b = peak |dB|/B0 of one line. `theory.py`: gyro-averaged resonant force F_rad (Bessel structure by FFT over gyrophase), omega_b, island size.

## Integrator checks (fig0_validation.png, data/val*.json)
| check | result |
|---|---|
| no wave: gamma conserved, gyro-phase | gamma drift 1e-16; phase error 4e-17 rad/step (tan-exact rotation) |
| pure wave exact first integral gamma-(w/kz)(u_z-A_z) | bounded, dI = 5e-4/1.3e-4/3.4e-5 at T/8,T/16,T/32 (dt^2); gamma excursion 0.009 |
| resonance condition and kick amplitude | peak at predicted gamma; sinc^2 lobe matches, ratio 0.995 (T/32), 0.999 (T/64), b=2e-9 |
| island bounce frequency | analytic omega_b vs orbits 1.000, 1.002, 1.000 (b=1e-6..1e-4), 1.09 at 1e-3 |
| broadband random-phase QL (129 lines, 16 realisations) | measured/QL = 0.983 +- 0.015 (Kennel-Engelmann/Pokol structure, golden rule derived from the same force law) |
| vs JAX-in-Cell Boris, same fields | both converge to the same trajectory (dt^2); JIC linearised Boris error is ~14x mine at equal dt (T/32: 0.157 vs 0.011) |
Production dt = T_gyro/32 (V2 kick amplitude 0.5% low).
JIC L4 as comparison model: not applicable. It is k_perp = 0 (parallel circular mode), 10 ns pulses with no E sweep, no collisions;
its QL response is valid only for noise-like bands (>= 32 lines, 2 w_b/dDelta = 3) and says a single coherent line is "trapped, no diffusion",
which agrees with the finding here. Its standard Boris has the gyro-phase error above, relevant to any resonance study.

## DIII-D numbers (data/mapping.json)
omega_b = 4.2e9 sqrt(b) /s at theta=0.2 (3.0e9 at 0.1, 2.1e9 at 0.05); b=1e-4: omega_b = 4.2e7/s, island half-width 0.029 in gamma (0.18%),
Delta u_perp ~ 0.7 vs u_perp 3.3. Sweep A = d delta/dt = 2.5e10 rad/s^2 (E only 3.2e10; friction and RR take 23%). Ec=0.0069 V/m, tau_c=0.25 s, nu_D=0.038/s.
Collisional frequency diffusion Gamma_c = (D_delta)^(1/3) = 4.8e5/s (order of magnitude, constant not fitted).
b at R=1 for each decorrelation rate (multiply by R_c ~ 2.5 for b_x): sweep 1.4e-9; collisions 1.3e-8; 10 MHz bandwidth 2.2e-4; toroidal spacing 1/R0 (0.44/m) 9.5e-4; 1 m transit 5e-3.

## Main result (fig1_Deff_over_DQL.png, data/expB_*_table.txt)
Protocol: ensemble (theta 0.1-0.3) accelerated through one resonance by E, friction and RR, compared with the unperturbed run;
metric <(xi_f(b)-xi_f(0))^2> over the stationary-phase QL value. Drivers time-compressed by lambda (b_phys = b_sim/lambda at equal R).
| R | 0.3 | 1 | 3 | 10 | 30 | 100 | 250 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| D_eff/D_QL, lambda=30-100 | 0.9 | 0.9 | 0.4 | 0.13 | 0.06 | 0.02 | | | |
| lambda=1e3 | 0.87 | 0.98 | 0.5 (R=2) | 0.25 | 0.12 | 0.06 (R=125) | 0.028 | 0.006 | 0.002 |
R_c (ratio 0.5) = 2-3. Plateau 0.9 below that (10% low, within 1-2 sigma, systematic not understood).
At R>>1 D_eff ~ b, independent of the sweep, equal to a bounded island kick; D_QL ~ b^2/A. Kicks: kurtosis 10-50 near R_c (heavy tails),
1.2-2 (bimodal/sinusoidal) for R >= 100 (fig3). lambda dependence is real: at R~100 the ratio is 0.02 (lambda 30), 0.024 (100), 0.06 (1e3) because
the finite island (Delta u_perp vs u_perp) is larger at larger b_sim. Physical DIII-D (R~1e4-1e5, island/u_perp 0.2-1) was NOT simulated (needs 1e8+ steps); the number
at DIII-D b is an extrapolation: D_eff/D_QL ~ 1e-5 to 1e-4 (with R*h between 1.5 and 7).
Collisionless, no decorrelation (fig2): all 15 b = 1e-6..1e-2 collapse on one curve of omega_b t; D_eff/D_QL = 0.94 at 0.3, falls to ~0.04 at omega_b t = 30. There is no QL regime for a single line beyond one bounce time.
Small pitch (fig4): enhancement window described in the verdict; theta^2-gain metric peaks at ~2.4 (lambda=100) and 1.4-3.5 (1e3).
Collisions with RR: RR acts only through the sweep rate (pitch damping 0.005/s). Collision Langevin runs in the lambda-compressed frame were contaminated (the rescaled pitch variance per passage is ~1e-2, far above the wave kick) and are discarded: the collisional shift of b_x
(sweep+collisions: Gamma^2 = A + Gamma_c^2 gives b_x ~ 4e-8) is an analytic estimate only. The planned scan of Gamma_c/omega_b at physical lambda=1 was not run (cost).

## Two lines (fig5, data/expD_summary.json)
Equal lines at k_par = 41 and 41+dk. sigma = 2(omega_b1+omega_b2)/Delta. <dgamma^2>/w^2 (w island half-width, t=300/omega_b) rises from 0.03-0.2 at sigma <= 0.56 to 0.8-0.9 at sigma = 1.0-1.3,
then decays (adiabatic merging, sigma >> 1). Scans in dk (b=1e-4) and in b (dk=0.44/m) collapse on sigma. Jump at sigma_c ~ 0.8-1.0: Chirikov sigma=1 confirmed to ~20%.
At the toroidal mode spacing dk = 1/R0 = 0.44/m, sigma = 1 at b = 5.6e-5, inside the DIII-D range.

## Where each amplitude sits (assumptions explicit)
- Power to amplitude, 150 kW, u_w = b^2 B0^2/(4 mu0) (peak b, time average), cold v_g,par = 0.28c: single pass through 1-4 m^2: b = 7e-5 to 3.4e-5.
  Steady state P = Gamma_P u_w V_p, V_p = 18 m^3: Gamma_P = 1e7, 1e6, 1e5, 1e4, 1e3 /s gives b = 4.6e-5, 1.5e-4, 4.6e-4, 1.5e-3, 4.6e-3.
  Spitzer collisional damping alone (nu_ei = 4e3/s) gives ~2e-3. Landau and antenna loss unknown: DIII-D b is uncertain over 3e-5 to 5e-3. Kang's literal formula (|omega_i|/2pi) differs by about 2pi in rate.
- Guo 1e-5 to 1e-4 sits at R ~ 7e3-7e4 for a coherent line, but his Gaussian k_par line (random-phase QL) assumes Gamma = Delta k c >> omega_b; consistent with our law only if that spectrum is truly incoherent.
- Kang b > 4.75e-4 (E0 = 3.55e-5 in B0 c units, gamma_r = 3): same locking criterion omega_b^2 ~ |d delta/dt| for a field ~3e5 x larger than DIII-D E (my estimate of the threshold ~26 E0 = 9e-4 is within 2x). Their capture is for particles starting near resonance at small pitch; our theta>=0.1 passage shows no capture.

## Limits (honest)
1. Uniform plasma, one plane wave, no toroidal geometry, no wave damping/refraction; b is a free parameter.
2. lambda-compression is exact only for the pendulum limit; finite-island effects change the ratio by up to x3. DIII-D point is extrapolated.
3. Collisional effect on b_x is analytic, not simulated.
4. Two-line result uses the 1-D chirikov reduction on 2D (gamma, theta) dynamics, N=256, one realisation; sigma_c resolved to ~20%.
5. Plateau 0.9 below crossover unexplained at the 10% level.
6. Pitch metric for xi ~ -1 is ambiguous; the xi and theta^2 metrics differ by up to x2 at R>100 and disagree on the enhancement size at small pitch.
