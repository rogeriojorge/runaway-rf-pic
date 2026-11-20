# P3 notes: does the DIII-D helicon "power threshold" equal the onset of resonance-island overlap?

Scripts: `scripts/` (JAX_ENABLE_X64=1; laptop CPU only, at most 4 threads, no GPU, office not used because its load was 85-100).
Data: `data/` (< 1 MB). Nothing committed or pushed. The P2 integrator (`results/P2/scripts/bw.py`, plasma.py, theory.py) is imported unchanged.
The P2 JIC clone (`~/local/jic-work/P2/JAX-in-Cell`, branch quantum-recoil) had been deleted during the session; I re-cloned it (commit e4ec59a, same content as 200567b) because `bw.py` and `drift.py` import `landau_lifshitz` from it.

## Verdict (short)
**Undetermined.** The model gives a sharp, testable amplitude threshold in a stationary measure, but the mapping to power and the phase coherence of the launch are unknown, and the sweep (passage) protocol shows no sharp threshold.
1. The overlap threshold is real and sharp in the model. D_eff/D_QL collapses on one overlap parameter sigma = 2(w_b,j + w_b,j+1)/Delta_delta for N = 5, 10 and three realistic spectra. D_eff/D_QL = 0.1 at sigma = 0.85 and 0.5 at sigma = 1.0-1.12. For the DIII-D comb (line spacing 1/R, R = 1.75 m) b_thr = 1.0e-4 per toroidal line, total b_tot = 3.4e-4 (bands below). The predicted scaling b_thr ~ Delta^2 holds: fitted exponent 1.9 over a factor 8 in Delta.
2. Collisions and radiation reaction do not move the threshold at DIII-D. Collisional detrapping, nu_D/(island width)^2 = 8-200 /s, is 1e-7 to 5e-6 of omega_b = 4e7 /s. A shift of the 10%-level onset needs collisionality about 1e4 times DIII-D, and of the 50% level about 1e5 times.
3. The threshold exists only for a phase-coherent comb. Phase noise of Gamma_phi = 1e6 /s (Lorentzian FWHM 0.3 MHz) already lifts D_eff/D_QL at sigma = 0.7 from 0.01 to 0.1; 3 MHz FWHM moves sigma_c(0.5) from 1.04 to 0.82 (b_thr x 0.6); 10 MHz FWHM removes it (sigma_c(0.5) < 0.7). The papers give no linewidth, so whether the launch is coherent is **undetermined**.
4. Power. The coupled power at threshold is P_thr from 0.2 kW (loss rate 1e3 /s, lower corner) to 60 MW (single pass, upper corner) depending on the unknown wave-energy loss rate. The data (43 kW small drop, 76 kW strong reduction) are consistent with the model for a loss rate Gamma_P between 1.4e4 and 3.5e5 /s over the full uncertainty band (5.4e4-9.6e4 /s for the central values; wave lifetime 3-70 microseconds, Q = 1e4-2e5) and inconsistent with a single-pass wave (P_thr >= 0.3 MW). That cannot be distinguished with current data. The wave field is not measured.
5. Negative/caution results: (a) in a passage (sweep) protocol the overlap onset is NOT a sharp jump in the kick per passage. The kick rises smoothly with b and is bounded by island and band size, and it does not follow the quasi-linear (QL) value. (b) For a coherent comb the diffusive window closes again at large b (sigma > about 1.6-2.2 in 100-400 beat windows). (c) In the sweep protocol at N = 5 the wave captures and drags the whole ensemble at sigma = 2.2 (b = 4.9e-4) once R >= 3.5e3 (a locking effect, reduced model, lambda = 100); its behaviour at the physical sweep is untested. (d) Long-time (>= 1e3 line beats) spreading could not be converged with my reduced integrator and is not used; the closing of the diffusive window at large sigma is therefore established only for 100-400 beat windows.

## 1. Realistic antenna spectrum (task 1)
Sources: Choudhury PRL 136, 025101 and PPCF 68, 105047 (`papers/txt`). Numbers are from the text; `scripts/spec_numbers.py` reproduces the derived ones.

| item | value | source |
|---|---|---|
| frequency | 476 MHz, "10 MHz band"; klystron 1 dB bandwidth +-1.5 MHz (a gain bandwidth, not a linewidth); 100 ms pulses | PRL p2, PPCF 2.3 |
| antenna | 30-module traveling-wave comb line, R = 2.27 m, 0.42 m above midplane, about 40 deg toroidal span; module pitch (derived) 5.3 cm = lambda_par/4 | PPCF 2.2 |
| n_par | 3 at the antenna (lambda_par = 21 cm, k_par = 29.9 /m, n_phi = k_par R = 67.9); core k_par = 41 /m (PRL), 30-44 /m edge/core in GENRAY | PRL p3, PPCF 2.4, Fig. 6 |
| spectrum width | not stated for the antenna. GENRAY input is a Gaussian P(n_par), mean 3, standard deviation 0.2, 10 rays (PPCF App. A) | PPCF App. A |
| coherence | **nothing stated.** The only statements: a "poorly defined n_par spectrum" if power leaves only the first modules (PPCF 2.3); after many passes the waves "will have a broadened n_par spectrum and potentially be stochasticized in phase space" (App. B); ICE shows modes 8 MHz apart and 30-60 MHz rising tones in the off periods, no mechanism (PPCF 3.2). The wave field is not measured (Doppler back-scattering is planned) | PPCF |

Conversion to k_par lines. Axisymmetry makes the toroidal mode number n_phi an integer, and k_par = n_phi/R at the resonant radius, so the lines are spaced Delta = 1/R_res: 0.44 /m at the antenna radius, 0.57 /m at R = 1.75 m (where the runaways sit), 0.63 /m at R = 1.6 m. Baseline Delta = 0.57 /m, centre k_par = 41 /m. Line amplitudes are Gaussian in the line index j (amplitude = sqrt(power), power sigma_j):

| case | sigma_j (lines) | basis | N_eff = sum a_j^2 |
|---|---|---|---|
| low | 3.4 | uniform 40 deg aperture: FWHM(n_par) = 0.35, sigma_n_par = 0.15 | 8.5 |
| nominal | 4.5 | GENRAY: sigma_n_par = 0.2 gives sigma_k = 2.0 /m = 4.5 lines | 11.3 |
| high | 7.0 | comb-line power decaying along the antenna (effective aperture half of the 40 deg) | 17.5 |

Equal-line spectra N = 1, 2, 3, 5, 10 are used for the collapse. Line phases are random and fixed (coherent comb); the particle's initial z spans a full 2 pi R period.
Frequency. The 10 MHz band corresponds to 2 pi 1e7 /s = 0.21 /m of Delta k_par at v = c, i.e. about half the line spacing, and to omega_b of 6e7 /s at b = 2e-4. I could not find the linewidth. I therefore bound it with a phase-diffusion rate Gamma_phi on every line (Lorentzian FWHM = Gamma_phi/pi): 0 (coherent), 1e6, 3e6, 1e7, 3e7, 1e8 /s; both common-mode (one source) and independent per-line.

## 2. Method
- **Full-orbit reference** (`p3lib.py`, `runS_scan.py`, `runP.py`): P2 Boris integrator, dt = T_gyro/32, cold-plasma fast-wave fields at n_e = 0.8e19 (k_perp = 76 /m at k_par = 41 /m), B0 = 1.4 T.
- **Reduced resonant model** (`red.py`, about 1000x cheaper): gyro-averaged n = 1 resonance, phase-exact second-order integrator. Per line dgamma = G_j F_j b_j cos(psi_j) dt, du_z = (k_j/omega_n) dgamma (single-line invariant), dpsi_c/dt = -delta_c, psi_j = psi_c + (k_j - k_c) z + phi_j. E_par, collisional friction, field-quadratic Landau-Lifshitz force (along p, as in JIC and P1), Langevin pitch scattering, phase noise. dt-convergence was tested (omega_b dt <= 0.04 used for the scans).
- Validation (`fig0_validation.png`): vs the full-orbit Boris on the same grid, the reduced model gives sigma_c(D_eff/D_QL = 0.5) = 1.14 vs 1.15 (N = 5), 1.08 vs 0.97 (N = 10), 1.00 vs 0.94 (Gauss 4.5). The reduced model is 20-25% lower in D_eff/D_QL in the short windows for sigma > 1.2 (N = 10, Gauss: mean ratio 0.77-0.80). I take 10% as the systematic uncertainty on sigma_c, i.e. 20% on b_thr. Full-orbit Boris was run only for N = 2, 3, 5, 10, Gauss 4.5, Gauss 3.4 (2 points) and Delta = 1.14; all other spectra, the Delta scan, collisions and noise use the reduced model.
- **Collision operator** (`val_coll.py`): Langevin pitch step with nu_D = (Z+1) gamma/(p^3 tau_c) (P1 convention, Z_eff = 1.5, nu_D = 0.038 /s at gamma = 16.4), momentum magnitude conserved (max |dp/p| 4e-14). Without a wave, <xi>(t) and <xi^2>(t) follow the Lorentz-operator solution xi0 exp(-nu t) and 1/3 + (xi0^2 - 1/3) exp(-3 nu t) to 0.5% for xi0 = -0.98 and 0 (8192 particles, nu t up to 0.4). Energy drag is the P2 friction and the synchrotron force is the JIC Landau-Lifshitz term. The P2 artefact (pitch spread from time compression) is avoided because collisions are applied in real time (stationary runs of 3e-8 to 1e-6 s, pitch spread sqrt(nu T) <= 0.1 rad even at 1e6 times the DIII-D rate), and the multiplier f only scales nu_D.

Metric (collisionless, stationary, no drive; valid because at the physical sweep A = 2.5e10 rad/s^2 the detuning moves by A T = 2.5e4 rad/s = 6e-4 omega_b during a window): an ensemble with gamma uniform over the line band (theta0 = 0.2 +- 0.01, random gyrophase and z), D_eff = slope of <dgamma^2> over the second half of a window of length T = max(min(t_fill, 300/omega_b), 10 beats), t_fill = time for QL to fill 25% of the band, D_QL = sum_j pi (G F b a_j)^2 / (window width in detuning), which is the stationary-phase golden rule smoothed over the line spacing (validated by P2: 129 random-phase lines, measured/QL = 0.98). Seeds: 4 phase realisations x 384 particles. The overlap parameter is sigma = 2(w_b,j + w_b,j+1)/Delta_delta evaluated at the central lines, Delta_delta = Delta k |u_par|/gamma = 1.67e8 /s at 0.57 /m.

## 3. N-line overlap map (task 2) -- `fig1_overlap_map.png`, `fig1b_band_fill.png`, `fig2_delta_scaling.png`
- D_eff/D_QL is below 0.02 for sigma <= 0.73, reaches 0.1 at sigma_c10 = 0.83-0.87, and 0.5 at sigma_c50 = 1.0-1.12 for every spectrum with N >= 5 (the "onset" window of 10-100 beats). It then stays at 0.6-1.8 for 1.2 < sigma < 2.2 (values above 1 are the short-window overshoot also seen in the full-orbit runs).

| spectrum | sigma_c(0.1) | sigma_c(0.5) | b_thr(0.5) at Delta = 0.57 /m |
|---|---|---|---|
| N = 5 | 0.87 | 1.12 | 1.23e-4 |
| N = 10 | 0.84 | 1.07 | 1.11e-4 |
| Gauss 3.4 | 0.85 | 1.02 | 1.06e-4 |
| **Gauss 4.5 (nominal)** | 0.85 | 1.02 | **1.04e-4** (b_thr(0.1) = 7.3e-5) |
| Gauss 7 | 0.83 | 0.97 | 0.94e-4 |

- N = 2 and N = 3 cannot be treated with D_eff/D_QL because the window is only 0.1-0.4 gamma wide and fills in 10 beats; the ratio is noisy and not monotonic (negative slopes). For them the band-fill fraction at t = 300/omega_b (P2 convention, `fig1b_band_fill.png`) is the usable measure: it rises from 0.01-0.08 at sigma <= 0.73 to 0.19 at 0.85 and 0.25-0.35 at sigma = 1, the same sigma window as N >= 5. The location of the P2 N = 2 jump (sigma_c = 0.8-1.0) is reproduced; the plateau value differs because the fill-fraction normalisation here uses the band along the invariant curve (P2 used the gamma separation of the two resonances at fixed theta).
- N = 1: nothing to overlap; a single line is trapped (stationary D_eff = 0). The sweep-passage scatter of N = 1 is the P2 result.
- Collapse: curves for N = 5, 10 and the three Gaussians fall on one curve within the 10-20% scatter of the windows. It is the quantity b/Delta^2 that matters.
- **Delta^2 test**: N = 10 lines at Delta = 0.285, 0.57, 1.14, 2.28 /m: sigma_c(0.5) = 1.20, 1.07, 1.18, 1.04, b_thr = 3.5e-5, 1.1e-4, 5.3e-4, 1.6e-3, fitted exponent 1.88 (error about 0.15, from the 10% scatter in sigma_c). Prediction 2. Consistent.
- Upper side (coherent comb). Onset-window D_eff/D_QL (Gauss 4.5, `Sr2_gauss45_hi`): 0.68 (sigma = 3.5), 0.38 (4.5), 0.02 (6), 0.015 (7). In longer windows the diffusion closes earlier: Gauss 4.5 at 100 beats (omega_b dt = 0.0125): 0.51, 1.16, 1.6, 1.42, 0.58, 0.12, -0.01 for sigma = 0.99, 1.16, 1.36, 1.59, 1.87, 2.19, 2.56; at 400 beats: 0.85, 0.91, 0.56, 0.06, -0.08 for sigma = 0.99, 1.16, 1.36, 1.59, 1.87 and zero beyond. N = 10 behaves the same (400 beats: 0.29, 0.12, 0.06 at 1.01, 1.18, 1.39). This is the non-impulsive suppression (the overlapping lines merge into a smooth, periodic force), so for a perfectly coherent comb the diffusive window is about 0.85 < sigma < 1.6-2.2, closing at smaller sigma for longer windows. Whether it closes for the physical dwell (about 1e6 beats) is **unknown**: beyond about 1e3 beats the reduced integrator is not converged in dt (<dgamma^2> at 1e4 beats for sigma = 2.2 is 0.009, 0.015, 0.043, 0.052 for dt omega_b = 0.1, 0.05, 0.025, 0.0125, numerical damping), and the 100-400 beat windows may carry a residual 10-20% damping bias. No claim is made beyond 400 beats (`fig3_window_longtime.png`, right panel).
- DIII-D range. b = 3e-5 to 5e-3 corresponds to sigma = 0.55 to 7 at Delta = 0.57: the lower edge is below threshold, the threshold sits at b = 1e-4 (a third of a decade above the lower edge) and most of the range lies above it.

## 4. Physical time scales (task 3) -- `fig6_passage_sweep.png`
- Physical numbers at b = 1e-4 (sigma = 1): omega_b = 4.2e7 /s, line beating Delta_delta = 1.7e8 /s, sweep (E = 8.5 E_c, friction, RR) A = 2.47e10 rad/s^2. R = omega_b^2/A = 7.2e4, Q = Delta_delta^2/A = 1.1e6, so the sweep moves the detuning by one line spacing in 6.8 ms (4e4 bounce periods).
- Cost. The full-orbit Boris integrator needs 4.4e9 steps per particle for a physical passage of 5 lines (about 15 core-hours for 192 particles) and 3.4e9 steps for 2 lines (about 1.5 core-hours for 64 particles). The laptop was at load 60-160 (other jobs) and the office at 85-100, so I ran full-orbit passages only at lambda = 1000 (N = 1, 2, 5). The stationary protocol of Section 3 is exact at the physical A because A T is below 1e-3 omega_b over a window. The reduced model reaches the physical sweep: **N = 2, lambda = 1 (R = 3.6e4-1.4e5, Q = 1.1e6, 1e7 steps)**.
- Integrator convergence matters here. For the passage the reduced integrator error is second order in dt: at lambda = 1, N = 2, sigma = 0.7 the rms kick is 0.049 (omega_b dt = 0.15, mean shift +0.045) and 0.017 (0.075, mean +0.011). Richardson extrapolation gives 0.0063. Runs at lambda = 10 and the omega_b dt = 0.15 lambda = 1 runs for sigma = 0.7 and 1.0 (rms 0.049 and 0.047, both with a b-independent mean shift of +0.045) are therefore biased and are not used (only the dt-halved point is plotted).

| N = 2 equal lines, rms dgamma per passage (wave run minus no-wave control) | lambda = 1000 | 100 | 1 (physical) |
|---|---|---|---|
| sigma = 0.7 | 0.0063 | 0.0064 | 0.017 (omega_b dt = 0.075); Richardson 0.0063 |
| sigma = 1.4 | 0.033 | 0.036 | see fig6 / below |
| R at sigma = 0.7 | 35 | 355 | 3.6e4 |

  At sigma = 0.7 the kick is independent of the sweep rate (0.0063 at lambda = 1000, 100 and, after Richardson extrapolation, at lambda = 1). At sigma = 1.4 it is 0.033-0.036 at lambda = 1000 and 100 (0.056 at lambda = 1 with the biased omega_b dt = 0.15; the dt-halved value is given in `data/Pr_eqN2_lam1_dt075_s14.pkl` if the run finished, see the end of this file). So for N = 2 the passage is in the adiabatic limit and the kick, bounded by island and band size, does not grow as 1/sqrt(A). The QL-normalised ratio of the passage falls as 1/A because QL grows as 1/A (QL exceeds the band-fill value L_c^2/6 by 40x at lambda = 1000 and 4e4x at lambda = 1 for N = 2, sigma = 1), so it is not a usable measure of the threshold in a passage.
- N = 5 (reduced model; rms dgamma and mean dgamma, band-fill rms L_c/sqrt(6) = 0.118):

| sigma | 0.7 | 1.2 | 1.6 | 2.2 |
|---|---|---|---|---|
| lambda = 1000 (R = 35-349) | 0.094, +0.086 | 0.090, -0.035 | 0.100, -0.056 | 0.168, -0.097 |
| lambda = 100 (R = 353-3490) | 0.057, +0.024 | 0.093, -0.038 | 0.140, -0.054 | **1.56, -1.55** |
| full-orbit Boris, lambda = 1000 | | | 0.187, -0.166 | 0.274, -0.250 |

  The passage kick rises smoothly with sigma, from about 0.06-0.09 (below threshold, from the sum of five island crossings) to about 0.1-0.2 at sigma = 1.6-2.2, without a jump at sigma = 1. **No sharp threshold is visible in the per-passage kick.** At sigma = 2.2 and lambda = 100 every particle is captured and dragged (mean dgamma = -1.55: the wave run ends 1.55 below the control), while at lambda = 1000 (R ten times smaller) it is not. Capture grows with R, i.e. towards the physical sweep. This is a **locking** effect (the Kang et al. trapping picture), not diffusion, and it sets in at b about 5e-4 (sigma about 2) for N = 5, above the stationary overlap threshold. I have not run it at lambda < 100 for N = 5 (see the lambda = 10 line at the end of this file if it finished), so I cannot say how the locking threshold behaves at the physical sweep. It is the most likely candidate for a barrier that does not rely on diffusion.
- Full-orbit versus reduced model for the sweep protocol at lambda = 1000: N = 2 agrees within 10-30% for sigma >= 1.4 and differs by 3x at sigma = 0.7 (Boris 0.02, reduced 0.006); N = 1 differs by 3-10x (Boris rms 0.02-0.25, reduced 0.009-0.03); N = 5 differs by 1.5-2x at sigma = 1.6-2.2. This **disagreement is unresolved** (candidates: non-resonant harmonics and tangential force terms in the Boris run, finite Boris dt). It affects the sweep protocol only; the stationary protocol was validated against Boris.
- Extrapolation. (i) The stationary threshold sigma_c = 1.0-1.1 corresponds to A -> 0 and is exact at the physical A for the window used. (ii) For N = 2 the passage kick is independent of lambda from 1000 to 1 within the integrator error at sigma = 0.7. (iii) For N = 5 the capture at sigma = 2.2 appears between R = 349 and R = 3490 and the dependence on R beyond is untested. (iv) Not tested: N >= 5 at lambda = 1, the realistic Gaussian spectrum with sweep, and the sweep together with collisions.

## 5. Collisions and radiation reaction (task 4) -- `fig4_collisions_linewidth.png`, `fig0_validation.png`
- Physical scales at b = 1e-4: omega_b = 4.2e7 /s; island width in pitch about 0.07 rad (theta = 0.2); nu_D = 0.039 /s; collisional detrapping rate nu_eff = nu_D/Delta theta^2 = 8 /s (nu_D/Delta xi^2 = 2e2 /s with xi = cos theta); frequency decorrelation Gamma_c = ((d delta/d theta)^2 nu_D)^(1/3) = 6e5 /s. So **nu_eff/omega_b = 2e-7 to 5e-6 and Gamma_c/omega_b = 0.014: collisional detrapping is not comparable to omega_b at DIII-D, by 5-7 orders.** Radiation reaction (4 /s in gamma) only enters through the sweep rate.
- Simulation (reduced model, Gauss 4.5, Langevin with multiplier f on nu_D): D_eff/D_QL is unchanged within errors for f <= 1e3 (at sigma = 0.7: 0.007, 0.000, 0.011, 0.04 for f = 0, 1, 1e2, 1e3; at sigma = 0.85: 0.10, 0.13, 0.10, 0.12). It is rescued towards QL at f = 1e4 (0.18 at sigma = 0.7), 1e5 (0.77) and 3e5 (1.14), where Gamma_c = 2.8e7 /s = omega_b. Thresholds: sigma_c(0.1) = 0.85, 0.81, 0.84, 0.81 for f = 0, 1, 1e2, 1e3 and < 0.7 at f = 1e4; sigma_c(0.5) = 1.04, 1.03, 1.03, 1.03, 1.02 for f = 0 to 1e4 and < 0.7 at f = 1e5 (b_thr down by at least a factor 2). **At DIII-D collisionality (f = 1) the shift is within the 5% resolution of the scan (sigma_c(0.1) 0.85 vs 0.81, sigma_c(0.5) 1.04 vs 1.03); a shift of the onset needs about 1e4 times, of the 50% level about 1e5 times, DIII-D collisionality.** The sigma = 1.5-2 curves decrease slightly (1.84 to 1.1) as collisions grow, removing the above-QL overshoot.
- Time-compression artefact check: nothing is compressed here; windows of 1e-7 to 1e-6 s keep sqrt(nu_D f T) <= 0.1 rad at f = 1e6.
- Same figure: phase noise. Common-mode and independent per-line phase noise Gamma_phi (Lorentzian FWHM = Gamma_phi/pi): at sigma = 0.7, D_eff/D_QL = 0.007, 0.11, 0.16, 0.40, 0.83 for Gamma_phi = 0, 1e6, 3e6, 1e7, 3e7 /s (common-mode; per-line noise is similar: 0.07, 0.18, 0.48, 0.70 for 1e6 to 3e7). The onset sigma_c(0.1) falls below 0.7 already at 1e6 /s (0.3 MHz FWHM). sigma_c(0.5) = 1.04, 1.01, 0.99, 0.82 and < 0.7 for 0, 1e6, 3e6, 1e7 (3 MHz FWHM) and 3e7 /s (10 MHz FWHM): b_thr(0.5) x 0.6 at 3 MHz and less than x 0.45 at 10 MHz. The upper cutoff (sigma > 2.2) is also removed (D_eff/D_QL at sigma = 2: 2.0, 1.7, 1.5, 1.6, 1.3, 1.0). **The threshold is a coherent-comb phenomenon.**

## 6. Power threshold in kW (task 5) -- `fig5_power_threshold.png`
Amplitudes: b_thr (peak line) = 1.0e-4 (central; Gauss 4.5, Delta = 0.57 /m), range 0.6e-4 to 1.6e-4 (lower: Delta = 0.44 /m, N-line and full-orbit sigma_c spread; upper: Delta = 0.63 /m, N = 5, +20% model error). b_tot = b_thr sqrt(N_eff) = 3.4e-4 (1.8e-4 to 6.7e-4). Energy density u_w = b_tot^2 B0^2/(4 mu0) (time average, magnetic part; the cold-plasma electric part is 1.3%).

| power model (P2 notes) | P_thr central (range) |
|---|---|
| single pass, v_g,par = 0.09-0.28 c, area 1-4 m^2 | 1.2-15 MW (0.3-59 MW) |
| damping-limited P = Gamma_P u_w V_p, V_p = 18 m^3, Gamma_P = 1e3 /s | 0.8 kW (0.2-3.2) |
| Gamma_P = 1e4 /s | 7.9 kW (2.2-32) |
| Gamma_P = 1e5 /s | 79 kW (22-315) |
| Gamma_P = 1e6 /s | 0.8 MW (0.2-3.2) |
| Gamma_P = 1e7 /s | 7.9 MW (2.2-32) |

Data (PPCF Sec. 3.4, Table 3): zero helicon power, no change; 201951 (non-ideal, 100/100 ms, 43 kW on-period power, 0.04 MW in Table 3): small drop at high energy; 201952 (continuous, 76 kW): strong reduction above 7 MeV; 201953 (300/300 ms, 0.07 MW): clear decrease in Fig. 16, no spectrum shown; ideal 201948 (0.13 MW, E/E_c = 8.5): complete scattering; ideal 201950 (0.12 MW, E/E_c = 10): incomplete ("presumably more would have been needed"). The paper calls the existence of a threshold unproven ("further data is needed"), reverses the v1 ordering and lists confounders (waveform differs, density +20%, coupled power excludes gap loss, fast/slow fraction unknown).

Assessment:
- **Consistent** with a threshold between 43 and 76 kW if 1.4e4 <= Gamma_P <= 3.5e5 /s over the full amplitude and spectrum band; for the central values the loss rate would have to lie between 5.4e4 and 9.6e4 /s (`data/numbers.json`).
- **Inconsistent** with a single-pass wave (P_thr >= 0.3 MW even at the lowest corner, vs P <= 0.2 MW in every shot) and with any loss rate above 4e5 /s (lifetime < 2.5 microseconds, i.e. < 60 toroidal transits).
- Spitzer-like collisional loss only (about 4e3 /s, P2) gives P_thr = 3 kW (0.9-13 kW), so every shot (>= 43 kW) would be above threshold by a factor 3-15 in power (1.8-4 in b). Then 201951 vs 201952 must be explained by something else (E/E_c, density rise, spectrum) or by the closing of the diffusive window above sigma about 2.2.
- Hence: **undetermined**. The model supplies one number that DIII-D can measure (b_thr) and one it cannot yet (Gamma_P). The ordering of the three non-ideal shots is what any threshold model predicts and does not test overlap.
- Shot 201950 (ideal, 120 kW incomplete) versus 201948 (130 kW complete): the coherent-comb model predicts a window, not a monotonic law: above sigma about 1.6-2.2 (b > 2.6-5e-4, i.e. 2.5-5 times threshold in b, 6-25 times in power at fixed Gamma_P) the diffusive transport closes again. With Gamma_P = 7e4 /s (central, from 76 kW), 130 kW is sigma = 1.3, inside the window. Not testable with the present data.

## 7. Verdict and the decisive experiment
- The overlap mechanism gives: a sharp onset at b_thr = 1.0e-4 (0.6-1.6e-4) per toroidal line, equal to b_tot = 3.4e-4 in total, scaling as (1/R_res)^2; collisions and RR irrelevant; threshold only for a coherent comb.
- It explains the DIII-D observations only if the wave-energy loss rate lies in 1.4e4-3.5e5 /s and the launch is phase-coherent to better than about 0.3 MHz. Neither is known. Overall: **undetermined, not refuted**.
- **Decisive prediction for DIII-D.** (i) Measure the wave amplitude directly (Doppler back-scattering is planned): threshold of the HXR reduction at delta B/B per toroidal line of 1.0e-4 (0.6-1.6e-4), total 3.4e-4 (1.8-6.7e-4); the power at which this occurs gives Gamma_P. (ii) Frequency or phase modulation of the source with bandwidth >= 3 MHz (within the klystron +-1.5 MHz gain band and the 10 MHz source band): the model predicts that sigma_c(0.5) falls from 1.04 to 0.82 at 3 MHz FWHM (amplitude threshold x 0.6) and below 0.7 (x < 0.45) at 10 MHz, with sub-threshold scattering at the 10% level already for 0.3 MHz, and that the high-amplitude cut-off disappears. A threshold that is unchanged by modulation would falsify the coherent-overlap picture. (iii) A power scan on a coherent CW launch should show a window of enhanced scattering from b = 1e-4 to about 2-5e-4 (P from P_thr to 4-25 P_thr at fixed Gamma_P; the upper end depends on the dwell time, see Section 3) and weaker scattering above it; no such rollover with the modulated launch.

## Limits (honest)
1. Uniform plasma, no toroidal geometry beyond line spacing 1/R, cold-plasma fields, b a free parameter, wave damping and refraction not modelled.
2. Stationary protocol only for the threshold; the sweep protocol does not show a sharp threshold, and Boris and reduced model disagree in it (Section 4).
3. The reduced model is 20-25% below the full-orbit Boris in D_eff/D_QL for sigma > 1.2 in short windows; sigma_c agrees to 10%. Collisions, noise and the Delta scan use the reduced model only (the collision/noise scans used omega_b dt = 0.1, which shifts D_eff/D_QL by up to 15% at sigma = 1.6 and underestimates it at sigma >= 3.5).
4. Windows are 10-400 line beats (a physical dwell is about 1e6 beats). Beyond 1e3 beats the reduced integrator damps and is not usable; a symplectic scheme is needed for the long-time claim about the upper cutoff.
5. N = 2, 3 have no meaningful D_eff/D_QL; N = 1 has none.
6. D_QL for the Gaussian spectra is averaged over a 5-line window around the centre line; local amplitude variation is 13% across it.
7. b_thr uses the P2 fast-wave polarisation at n_e = 0.8e19; the PRL text and Table 2 give 0.5-0.7e19 (k_perp 36-60 /m): F_rad changes by an unquantified amount.
8. Power conversion uses the P2 models with their own assumptions (peak b, time average, volume 18 m^3, magnetic energy only).

## Files
Figures: `fig0_validation.png` (collision operator, reduced vs Boris), `fig1_overlap_map.png` (main), `fig1b_band_fill.png` (N = 2, 3 fill), `fig2_delta_scaling.png`, `fig3_window_longtime.png`, `fig4_collisions_linewidth.png`, `fig5_power_threshold.png`, `fig6_passage_sweep.png`.
Scripts: `p3lib.py` (spectra, QL, Boris stationary protocol), `red.py` (reduced model), `val_red.py` (stationary reduced run + comparison), `runS_scan.py` (Boris scans), `runSr_scan.py`, `runSr_hi.py`, `runSfill.py`, `runX.py` (collisions and noise), `runP.py` (Boris passage), `runPr.py` (reduced passage), `val_coll.py`, `spec_numbers.py`, `power.py`, `analysis.py` (all figures and `data/numbers.json`).

## Addendum (runs still in progress when this note was written)
- N = 5, lambda = 10 (reduced, omega_b dt = 0.15, 96 particles): sigma = 1.2 gives rms dgamma = 0.26, mean -0.03 (vs 0.09 at lambda = 100 and 1000). Given the dt-bias found for the lambda = 1 runs (error second order in dt, growing with the number of steps) I do not interpret this as physical; sigma = 1.6 and 2.2 at lambda = 10 were not finished, so the locking behaviour for R > 3.5e3 at N = 5 is not established beyond lambda = 100.
- N = 2, sigma = 1.4, lambda = 1 with omega_b dt = 0.075 (to remove the dt-bias of the 0.056 value) and the full-orbit N = 2 passage at the physical sweep (3.4e9 steps, `data/Pb_eqN2_lam1.log`) were not finished; the background jobs were stopped at hand-off. The only physical-lambda numbers in this note are the N = 2 reduced-model ones in Section 4.
- Output sizes: `data/` is under 1 MB.
