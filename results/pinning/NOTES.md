# Pinning test: is the helicon amplitude self-limited by runaway absorption at the island-overlap threshold?

Scripts: `scripts/` (JAX_ENABLE_X64=1; numpy/scipy/jax; laptop only, 1 thread per process (2-3 processes at a time, slightly above the 2-thread allowance for about 30 minutes), laptop load 60-130 from other jobs; the office was not used because its load stayed at 47-58, above the 40 limit).
Data: `data/` (json, logs). Figures: `fig_main_b_vs_Pin.png`, `fig1_validation.png`, `fig2_Pwave_vs_b.png`, `fig3_feasibility.png`, `fig4_transition.png`.
Nothing committed or pushed. The P2/P3 code (`plasma.py`, `bw.py`, `theory.py`, `p3lib.py`, `red.py`) is imported unchanged; `scripts/redpin.py` is a copy of `Red.run` that adds wave-work and drive-work accumulators.

## Verdict
**Pinning is refuted as formulated, and refuted for a stated physical reason; one corner of parameter space stays undetermined.**

1. The runaway absorption rate in the pinning hypothesis (Step 2: 1.1e8 /s per 1e15 m^-3) is a quasilinear (QL) rate for a fixed, unflattened distribution. I re-derived it (energy balance over the resonance characteristic) and validated it against full-orbit test particles: the QL value for the P1 distribution is **9.5e6 /s per 1e15 m^-3 of runaways** (2.0e7 /s per 1e15 m^-3 of runaways inside the resonant zone), 5-12x below Step 2, and full-orbit electrons reproduce it to 0.83 (early) / 0.61 (late) at b = 3e-4 per line (sigma = 1.75).
2. That rate cannot be sustained. The wave flattens the distribution along the resonance characteristic in about 1 microsecond (D_QL = 7e3 gamma^2/s at b = 1e-4, 6e4 at 3e-4; characteristic length 0.2-0.3 in gamma). After that the absorbed power is limited by how fast fresh, low-p_perp electrons enter the zone, and by the Manley-Rowe relation: every absorbed quantum raises p_perp^2/2 by hbar*Omega_ce/(m c^2), so the energy absorbed per electron transit is only (omega/Omega_ce) Delta(p_perp^2)/2 = 0.012 x Delta(p_perp^2)/2 m c^2, at most about 1.4 m c^2 (zone geometry), and the zone is refreshed at 4-5 /s (E-field transit, measured in the driven test-particle runs). **Replenishment-limited absorption is 1.1-7.7 m c^2/s per resonant electron, i.e. 1.6-11 kW per 1e15 m^-3 resonant runaways (V = 18 m^3), nominal 7 kW.** The hypothesis needed 14 MW per 1e15 at b_thr (validated QL rate) or 68 MW (Step-2 rate).
3. Against the thermal threshold power P_th(b_thr) = Gamma_th u_w V = 35 kW (Gamma_th = 4e4 /s, the 2.5% end), 79 kW (16%), 238 kW (median 2.7e5 /s), 530 kW (84%), the absorption capacity for n_res = 1e15 m^-3 is 0.2 (Gamma_th = 4e4), 0.09 (9e4), 0.03 (median) of P_th(b_thr); for the extreme n_res = 4e15 m^-3 (all of a 0.9 MA current inside the zone) it is 0.8, 0.4, 0.12. The capacity reaches 0.3 of P_th only for Gamma_th <= 9e4 /s with n_res >= 2-3e15 m^-3, or Gamma_th = 4e4 /s with n_res >= 1.4e15 m^-3. Even then the absorbed power is a **b-independent offset** (it saturates at b = 5e-5, below b_thr), so b grows as sqrt(P_in - P_abs): the runaways act as a constant power dump that delays the amplitude rise, they do not hold b at b_thr (R_pin = 1.0-1.4 against 1.49 for no absorption; unsaturated QL hypothesis: 2.9).
4. No bistability or hysteresis: the loss function L(b) = Gamma u_w V + n V p(b) is monotonic in all replenishment-limited cases (40 combinations tested). A non-monotonic L(b) appears only when the unsaturated QL rate is combined with a closing diffusive window (sigma > 1.6-1.9), at P_in > 1 MW (outside the DIII-D range), and so does not apply.

What is not settled: (a) the physical (lambda = 1) value of the replenishment-limited power within 1.1-7.7 m c^2/s: driven runs reach lambda = 300 only, and the wave work per unit time is still rising as lambda falls (table in Sec. 2); (b) the reduced model underestimates the wave work 2.1-2.6x against full orbits, which moves the approach to the cap, not the cap; (c) the DIII-D distribution is the P1 one (B = 2 T, n = 1e18 m^-3, E/E_c = 6.5), not a DIII-D kinetic solution; the DIII-D drive, with tau_rad/tau_c = 10.7 against 0.58 in P1, has a different tail shape and a fixed point at larger pitch.

## 1. Quasilinear absorption by runaways: derivation, validation, DIII-D values (task 1)

### 1.1 Equations
Resonant electron (gamma, p_perp, p_par in m c), wave mode of peak field E0 (unit polarisation vector from the cold dispersion tensor, `plasma.py`), harmonic n (n = +1 normal Doppler, k_par v_par < 0; n = -1 anomalous):

- resonance: gamma omega - k_par c p_par = n Omega_ce (Omega_ce non-relativistic);
- resonant energy gain rate: d gamma/dt = c_n cos(Psi), c_n = (e E0/(m c)) |A_n|, A_n = < (E_hat . beta) exp(i zeta sin phi - i n phi) >_phi, zeta = k_perp p_perp c/Omega_ce (gyro average);
- random phases: <Delta gamma^2>/t = pi c_n^2 delta(omega - k_par v_par - n Omega_ce/gamma);
- characteristic (Kennel-Engelmann): d p_par/d gamma = k_par c/omega, d p_perp/d gamma = n Omega_ce/(omega p_perp) (exact consequence of the resonance condition), so Delta gamma = (omega/Omega_ce) p_perp Delta p_perp;
- QL diffusion is along the characteristic with Jacobian d^3p, hence the power given to the electrons

  P = -(m c^2/2) Int d^3p D_gg d_gamma f|_char,  D_gg = pi c_n^2 delta(.),  d_gamma f|_char = (k_par c d_pz f + n Omega_ce p_perp^-1 d_pperp f)/omega;

- amplitude growth rate gamma_k = -P/(2 W), W = time-averaged cold wave energy density = eps0/4 E^H d(omega K)/d omega E + |B|^2/(4 mu0); power damping rate of the wave Gamma_P = P/W.

Checks of the building blocks (`t_coupling.py`, `t_anom.py`):

| check | result |
|---|---|
| A_1 (gyro average) vs P2/P3 coupling G F_rad, DIII-D, gamma 16.4-17.2, theta 0.2-0.4 | identical to 4 digits |
| Pokol 2008 eq. 12 psi_n (n = 1) with the sign s of the E_y term | matches A_1 only with -i E_y J_n' (s = -1); Pokol's +i E_y is a different handedness convention |
| A_n vs a Boris test particle at the resonance, n = +1 and n = -1, theta 0.1-0.3, b = 1e-6 | rms energy change within 1-2% of A b T / sqrt2 |
| QL normalisation of <Delta gamma^2>/t = pi c^2 delta | P2 `val4_quasilinear` (existing) and Sec. 1.3 below |

### 1.2 Published number: Fulop, Pokol, Helander, Lisak, PoP 13, 062506 (2006)
Case of their Fig. 3: n_e = 5e19 m^-3, B = 2 T, deuterium, n_r/n_e = 5e-3, theta_k = 85 deg, distribution eq. 9 with c_Z = sqrt(3(Z+5)/pi) ln(Lambda) (ln Lambda = 15, c_Z = 35.9, E = 30 so alpha >> K_par), n = -1. The growth rate was computed with `qlib.gain_power`, with E_x-only coupling (the approximation behind their Im chi_11) and with the full cold polarisation. Eq. 21 is used as printed; eq. 22 as printed has k^2 v_A^2/omega_0, which is dimensionally inconsistent and does not give eq. 24; with k^2 v_A^2/omega_0^2 it reproduces eq. 24, and I use that form.

| k [1/m] | omega/omega_ci | eq. 21 [omega_ci] | E_x-only / eq. 21 | full polarisation / eq. 21 |
|---|---|---|---|---|
| 80 | 3.8 | 0.0047 | **0.94** | 0.50 |
| 100 | 4.9 | 0.0176 | **0.92** | 0.56 |
| 150 | 7.8 | 0.107 | 0.87 | 0.61 |
| 250 | 15.3 | 0.48 | 0.76 | 0.56 |
| 400 | 30.6 | 1.17 | 0.60 | 0.41 |
| 600 | 55.6 | 2.06 | 0.41 | 0.21 |

Result: with the paper's own coupling the energy-balance code reproduces the closed form to **6-8% at small k, i.e. where the closed form's assumptions hold (z = k_perp p_perp c/Omega_ce < 0.3, lambda << 1)**. The ratio falls as z grows (0.6 at z = 0.5), as expected from the lowest-order lambda expansion; I did not implement their eq. 20, which keeps the Bessel/exponential terms, so this part is untested. At the paper's optimal-mode conditions (k_par = 65 /m, k_perp = 500 /m, kv_A/omega_pi about 0.5; `fulop_eq24.py`) the code gives 9.5e7 /s (E_x-only) against 1.63e8 /s from eq. 24 (ratio 0.58; eq. 24 itself reproduces 1.3e-9 n_r/B_T exactly). I could not reproduce the absolute height of the Fig. 3 curve (peak about 1.5 omega_ci; eq. 21/22 as printed give 1.4 and 2.3 omega_ci at k = 450 and 650).
With the full cold polarisation the growth rate is 0.4-0.6x lower still: E_y and E_z reduce the n = -1 coupling in my convention (Pokol eq. 13, in its own convention, increases it by (1 + k_par/k)). The sign of that term is fixed in my code by the Boris check of A_n, not by the paper. I treat the Fulop check as a validation of the energy-balance normalisation in the small-z limit only. The DIII-D evaluation below has z = 0.3 and uses the validated A_n coupling.
Data: `data/fulop_check2.json`, `data/fulop_eq24.json`; `fig1_validation.png`.

### 1.3 Absorption by DIII-D runaways (n = +1, 476 MHz, k_par = 41 /m, k_perp = 76.6 /m, B = 1.4 T, n_e = 0.8e19 m^-3)
Distribution: P1 stationary tail (`results/P1/data/full_fp.npz`, t >= 60 s, 3e5 samples, smoothed on a (p, theta) grid), normalised to n_RE. Counter-propagating electrons, p_par < 0. `diiid_ql.py`.

| k_par [1/m] | k_perp [1/m] | Gamma_P (wave power damping rate) per 1e15 m^-3 of runaways [1/s] |
|---|---|---|
| 35.3 | 95.8 | 1.06e7 |
| 38.15 | 85.8 | 1.01e7 |
| **41.0** | 76.6 | **9.5e6** |
| 43.85 | 68.0 | 9.0e6 |
| 46.7 | 59.6 | 8.4e6 |

Sign: electrons gain energy, so the runaways damp the wave (the characteristic moves mostly in p_perp, down the f gradient, even though the resonant gamma = 16 lies on the rising side of the gamma distribution). 29% of the P1 electrons are in the comb resonance zone (gamma 14.5-20.5, theta < 0.54), 47% in the simulation zone (gamma 12.5-22.5, theta < 0.75); per electron of the zone the rate is 2.0e7 /s per 1e15. Runaways dominate the damping above n_RE = 2.8e13 m^-3 (Gamma_th = 2.7e5 /s); Step 2 gave 3e12 m^-3 for a monoenergetic resonant slice, 1.1e8 /s per 1e15 against 2.0e7 here per resonant-zone electron (ratio 5.5), a factor within the "factor 10" Step 2 quoted.

**Test against particles** (`ql_vs_particles.py`, `fullorbit_check.py`, `ql_early*.json`). Gaussian comb (sigma_j 4.5, 23 lines, N_eff 11.3, spacing 0.57 /m), P1 electrons in the zone, collisionless, no drive; slope of the mean wave work per electron, ratio to the QL prediction for the same distribution:

| b_pk | sigma | full orbit (Boris) | reduced model |
|---|---|---|---|
| 3e-4 | 1.71 | **0.83** (first 0.1 us), 0.61 (late) | 0.35 |
| 1e-3 | 3.2 | not run | 0.29 (early), 0.13 (late) |
| 1e-4 | 0.99 | 0.13 (early), 0.07 (late) | 0.02-0.05 |

The QL energy balance agrees with the full-orbit electrons to 20-40% in the stochastic regime. The reduced model is a factor 2.1-2.6 low (same finding as P3 for the passage; also directly: `fullcheck_b*.json`, wave work in 3e-5 s with drive, full/reduced = 2.6 +- 0.7 at 1e-4 and 2.1 +- 0.6 at 3e-4). At sigma = 1 the measured absorption is 7-13% of QL, not the 70% implied by D_eff/D_QL(sigma = 1) from P3, so in this energy-weighted measure the onset sits at somewhat larger sigma than the 50% point of the diffusion collapse.

### 1.4 Collisional damping: the factor of 30 (Fulop eq. 25)
`collisional_damping.py`: cold tensor with Krook electron-ion collisions (omega -> omega + i nu_ei on electrons), complex root at fixed real k; amplitude damping rate over nu_ei:

| case | Im(omega)/nu_ei | Fulop eq. 25 (1.5/tau_ei) |
|---|---|---|
| Fulop Fig. 3, k = 450 /m, 85 deg | 0.086 | 1.5 |
| same, k = 150 /m | 0.008 | 1.5 |
| quasi-parallel whistler, k = 450, 20 deg | 0.088 | 1.5 |
| DIII-D fast wave (0.8e19, 1.4 T, 2165 eV, k_par 41) | 0.0242 (power rate 0.048 nu_ei, as in Step 2) | 2.25 (Z_eff = 1.5) |

nu_ei (Spitzer, PPCF B9) = 1/tau_ei exactly for Z = 1 (Fulop's tau_ei), 2/3 for Z_eff = 1.5, so the comparison is not a unit problem. The cold whistler limit is Im(omega) = nu_ei k^2c^2/(omega_pe^2 + k^2c^2) (k = 450: 0.10), which can never exceed nu_ei; the 1.5 coefficient in eq. 25 is a constant that the cold result reaches in no regime. Fulop's value is 17x above the cold result in their own regime and 62x (amplitude basis) above it for the DIII-D fast wave, whose energy is carried by the ions (electron E x B energy fraction about 1/1800 at omega/omega_ci = 44). The "30" of Step 2 is 1.5 nu_ei against the power rate 0.05 nu_ei. Resolution: regime (energy partition) and a coefficient (1.5) taken from a kinetic reference that I did not re-derive. It does not matter here: at 2 keV the collisional rate is 4e3 /s volume averaged, 1.5 nu_ei would give 1.7e4-5e4 /s, both below the Landau term.

## 2. Test-particle measurement of the absorbed power (task 2)

### 2.1 Protocol (`pinrun.py`, `redpin.py`)
- Model: gyro-averaged n = 1 reduced resonant model (P3 `Red`, F table extended to gamma 11.5-23.5, theta 0-0.8), Gaussian comb (sigma_j = 4.5, 23 lines, spacing 0.57 /m, N_eff = 11.3), b_pk = per-line peak amplitude, sigma = Chirikov parameter of adjacent lines (sigma = 1 at b = 1.0e-4). DIII-D E = 8.5 E_c = 0.0589 V/m, friction, Landau-Lifshitz radiation reaction, Langevin pitch scattering (Z_eff = 1.5); tau_c = 0.246 s, tau_rad = 2.63 s.
- Ensemble: 256 electrons drawn from the P1 distribution restricted to gamma 12.5-22.5, theta < 0.75 (47% of P1). **Refill**: an electron that leaves that box is replaced by a fresh P1 draw (the E-field flux through the zone). Measured exit rate 3.7-5.4 /s (physical), about the E-field transit time: (22.5 - 12.5)/(30.5 /s net) = 0.33 s.
- Time compression lambda multiplies only the slow drives (E, friction, LL, pitch scattering). The wave interaction is at its physical rate. P_wave is the mean wave work per electron per simulated second; P_wave/lambda is its physical-time equivalent **only if the zone is replenishment-limited** (P_wave proportional to lambda). The wave-off control gives P_wave = 7e-5, net drive work 28.1 m c^2/s (E - E_c = 30.5 /s less friction/RR), <p_perp^2> = 5.7.
- Convergence: omega_b dt = 0.05 for all runs; the 4x smaller step test is in Sec. 2.3 (P_wave +43%). P3 found numerical damping beyond 1e3 line beats at larger sigma.
- Energy accounting (`fullorbit_check.py`): in the Boris runs gamma_final - gamma_0 = wave work + E0 work + friction + LL to 5e-8 (b = 1e-4) and 1.8e-7 (3e-4), rms over the ensemble; the reduced model's drive work falls by 14-16 m c^2/s when p_perp^2 rises by 38, which equals the LL loss a_B Delta(u_perp^2) = 0.38 x 38 = 14.4 /s (energy goes to synchrotron radiation, supplied by the E-field).

### 2.2 Results
| b_pk | sigma | lambda | P_wave [m c^2/s sim] | P_wave/lambda | drive work/lambda | <p_perp^2> | exit rate [1/s] |
|---|---|---|---|---|---|---|---|
| 1e-5 | 0.31 | 1000 | 22 | 0.022 | 27.8 | 6.4 | 5.1 |
| 3e-5 | 0.54 | 1000 | 21 | 0.021 | 27.5 | 7.0 | 5.0 |
| 5e-5 | 0.70 | 1000 | 205 | 0.21 | 24.7 | 13.4 | 4.7 |
| 7e-5 | 0.83 | 1000 | 456 | 0.46 | 22.1 | 19.2 | 4.4 |
| 1e-4 | 0.99 | 3000 | 1070 | 0.36 | 22.3 | 18.6 | 4.4 |
| 1e-4 | 0.99 | 1000 | 557 | 0.56 | 19.1 | 25.9 | 3.9 |
| 1e-4 | 0.99 | 300 | 324 | **1.08** | 16.4 | 32.1 | 3.7 |
| 1.5e-4 | 1.21 | 1000 | 1408 | 1.41 | 15.4 | 34.6 | 3.9 |
| 3e-4 | 1.71 | 3000 | 7139 | 2.38 | 14.0 | 38.1 | 4.5 |
| 3e-4 | 1.71 | 1000 | 3404 | 3.40 | 12.3 | 42.6 | 5.0 |
| 3e-4 | 1.71 | 300 | 1255 | **4.18** | 12.1 | 43.6 | 5.4 |

(All points: `data/pin_s*_b*_lam*.json`; the run labelled `pin_s2_b1e-04_lam1000.json` is b = 1.5e-4. `fig2_Pwave_vs_b.png`. Full-orbit transition scan: Sec. 2.3.)

Findings:
- **Onset**: at lambda = 1000 the absorbed power jumps by 10x between b = 3e-5 (sigma 0.54) and 5e-5 (sigma 0.70), i.e. at the same sigma = 0.7-0.85 where P3's D_eff/D_QL leaves 1e-3 (0.0056 at 0.72, 0.15 at 0.85). Below it (b <= 3e-5) the wave work is at the noise floor (about 20 m c^2/s sim, the lambda = 1000 passage is not adiabatic there, R = omega_b^2/(lambda A) = 7-20). It is not a jump to a new power law above the onset: P_wave/lambda grows only 0.2 -> 3.4 over sigma = 0.7 -> 1.7 whereas QL scales as b^2 (x 6).
- **Saturation**: P_wave/lambda rises as lambda falls (b = 1e-4: 0.36, 0.56, 1.08; 3e-4: 2.4, 3.4, 4.2), i.e. P_wave grows as lambda^0.5-0.8 and not as lambda^0 (diffusion limited) or lambda^1 (replenishment limited). The sequence converges toward the bound below; the physical (lambda = 1) value is **at least 1.1 (b = 1e-4) and 4.2 (b = 3e-4) m c^2/s per zone electron** and, by the argument below, at most 7.7.
- **Energy**: the zone electrons gain (omega/Omega_ce) Delta(p_perp^2)/2 per pumped transit. With Delta<p_perp^2> = 38, exit rate 5.4 /s, 0.0121 x 38/2 x 5.4 = 1.2 m c^2/s if the mean is used and up to 7.7 m c^2/s if the maximum (gamma 22.5, theta 0.75, p_perp^2 = 234) is reached in each transit. The measured 4.2 at lambda = 300 lies between and rising: the bound is not violated and not yet saturated.

**Bound used in the coupled model.** P_cap = nu_pass (omega/Omega_ce) Delta(p_perp^2)_max / 2 per zone electron = 5.4 x 0.0121 x 234/2 = 7.7 m c^2/s (upper); 1.1 m c^2/s (measured at b = 1e-4, lambda = 300, lower); nominal 5 (geometric middle of the b = 3e-4 measurement and the upper bound). Per 1e15 m^-3 of zone electrons in 18 m^3: 1.6, 7.4, 11 kW.

### 2.3 Transition across the overlap threshold (full orbit) and integrator convergence
`fullorbit_transition.py`, `ql_vs_particles.py`: collisionless, no drive, P1 zone electrons, Gaussian comb, full-orbit Boris (dt = T_gyro/32, 768-1536 electrons); wave-work slope divided by the QL energy-balance prediction for the same distribution (`fig4_transition.png`):

| b_pk | sigma | slope / QL, early (0.1-0.5 T) | late (0.5-1 T) |
|---|---|---|---|
| 5e-5 | 0.70 | -0.004 (zero within noise) | -0.004 |
| 7e-5 | 0.83 | -0.06 | -0.01 |
| 1e-4 | 0.99 | 0.13 | 0.07 |
| 1.4e-4 | 1.17 | 0.18 | 0.23 |
| 2e-4 | 1.40 | 0.31 | 0.29 |
| 3e-4 | 1.71 | 0.83 | 0.61 |

(T = 1.5e-6 s for 5e-5..2e-4, 4e-7 s (early)/3e-6 s for 3e-4, 1e-4: 1e-6 s.) The unsaturated absorption has a **sharp onset at sigma = 0.83-1.0**: it is zero within noise at sigma <= 0.83 and 7-13% of QL at sigma = 1, growing to 0.6-0.8 of QL by sigma = 1.7. The P3 diffusion-collapse ratio D_eff/D_QL is higher at the same sigma (0.15 at 0.85, 0.7 at 0.99), so the energy-weighted onset sits at about 0.15 larger sigma than the 50% point of D_eff, and with the 0.6x-0.8x full-orbit/QL level above it, the P3 b_thr = 1.0e-4 is the lower edge of the transition: 10% of QL at b_thr, 60-80% at 3e-4 (sigma 1.7).
The jump in P_RE(b) at sigma = 1 therefore exists in the unsaturated (first microsecond) absorption, and it is a rise from < 1% to 10% to 60% of QL over sigma = 0.83 -> 1.0 -> 1.7, not a step to the full QL value. In the replenished runs the jump is visible between sigma = 0.54 and 0.70 (Sec. 2.2) but is then compressed against the cap.

Integrator convergence: the b = 1e-4, lambda = 1000 case with omega_b dt = 0.0125 (4x smaller, Np = 128) gives P_wave = 795 against 557 at 0.05: +43%. The reduced model with omega_b dt = 0.05 thus under-counts long-time absorption by at least 30%, in addition to the 2.1-2.6x full-orbit difference. Both act on the approach to the cap, not on the cap.

## 3. Coupled wave-energy model (task 3)

dW/dt = P_in - Gamma_th W V - n_res V p(b) m c^2, W = u_w(b) = N_eff b_pk^2 B0^2/(4 mu0) (Step-2 convention, B0 = 1.475 T, N_eff = 11.3, V = 18 m^3), steady state solved for b_pk; `pin_model.py`, `scan_model.py`, `pin_metrics.py`, `hysteresis_test.py`. Closures for p(b), absorbed power per zone electron:
- `cap` (this work): p = [P_QL(b) rho(sigma)] || P_cap (harmonic combination), P_QL from Sec. 1.3 (zone electron: P_QL(b) = 9.5e6 /s x 1.12 u_w(b)/(1e15 m^-3 x 0.466)), rho(sigma) = D_eff/D_QL from the P3 full-orbit stationary data (Gaussian 4.5, capped at 1), P_cap = 5 (1.1-7.7);
- `QL`: same without the cap (the validated version of the hypothesis);
- `S2`: Step-2 rate 1.1e8 /s per 1e15 with rho(sigma).
The zone density n_res is held constant: the measured exit rate does not rise with b (5.1 -> 3.7-5.4 /s), so wave-induced depletion of the zone is below 30% in these runs.
Gamma_th: 4e4, 9e4, 2.7e5, 6e5, 1.4e6 /s (Step 2: 2.5%, 16%, median, 84%, 97.5%). P_in 10 kW-1 MW, n_res 1e11-1e16 m^-3. b_thr = 1.0e-4 per line (0.6-1.6e-4).

Main figure `fig_main_b_vs_Pin.png`: b_pk/b_thr against P_in, columns Gamma_th = 4e4, 2.7e5, 1.4e6, top row replenishment-limited absorption, bottom row unsaturated QL.

### 3.1 Results
Absorbed power at b_thr and pinning metric R_pin = P_in(1.1 b_thr)/P_in(0.9 b_thr) (1.49 with no absorption; pinning means well above 1.49). `data/pin_metrics.json`.

| closure | Gamma_th | P_th(b_thr) | n_res = 1e15 | 4e15 | 1e16 | 1e17 |
|---|---|---|---|---|---|---|
| cap 5 | 4e4 | 35 kW | f_abs 0.17, R_pin 1.39 | 0.46, 1.24 | 0.68, 1.14 | 0.95, 1.02 |
| cap 5 | 2.7e5 | 238 kW | 0.03, 1.48 | 0.11, 1.43 | 0.24, 1.36 | 0.76, 1.10 |
| cap 1.1 | 4e4 | 35 kW | 0.04, 1.47 | 0.16, 1.40 | 0.32, 1.31 | 0.82, 1.07 |
| cap 7.7 | 4e4 | 35 kW | 0.24, 1.35 | 0.56, 1.19 | 0.76, 1.10 | 0.97, 1.01 |
| QL (hypothesis) | 4e4 | 35 kW | 1.00, 2.89 | 1.00, 2.89 | 1.00, 2.89 | 1.00, 2.89 |
| QL (hypothesis) | 2.7e5 | 238 kW | 0.98, 2.85 | 1.00, 2.88 | 1.00, 2.89 | 1.00, 2.89 |

(f_abs = absorbed fraction of P_in at b_thr.)

- **Unsaturated QL reproduces the hypothesis only partly**: the amplitude is clamped at b/b_thr = 0.6-0.9 (not at 1) over P_in = 100 kW-1 MW for n_res = 1e14-1e15, because rho(sigma) rises smoothly, between sigma = 0.7 and 1.2. The plateau elasticity d ln b/d ln P_in is 0.08-0.13 against 0.5 without runaways, the pinning range being 2.9x in P_in for +-10% in b.
- **Replenishment-limited absorption gives no plateau for any n_res, Gamma_th, cap.** The absorbed power rises to its cap at b = 5e-5 (sigma = 0.7), below b_thr, and stays constant, so P_in = P_cap + Gamma u_w V and b/b_thr = sqrt((P_in - P_cap)/P_th). At the median Gamma_th and n_res = 1e15, b/b_thr is 0.63 at 100 kW against 0.65 with no runaways, and 2.04 against 2.05 at 1 MW. Only at Gamma_th = 4e4, n_res = 1e16 and cap 5 is the amplitude visibly reduced (0.87 against 1.7 at 100 kW) and that is an offset, not a plateau.
- **Where pinning fails**, in order of importance: (i) too little absorption capacity: cap/P_th = 0.03-0.2 at n_res = 1e15 and median-4e4 Gamma_th, 0.12-0.8 at 4e15 (`fig3_feasibility.png`), and a constant absorbed power is an offset anyway; (ii) replenishment limit rather than rate limit (the hypothesis did not include it); (iii) thermal damping dominates: Gamma_th above 1e5 /s (P_th >= 90 kW) cannot be compensated by < 5e15 m^-3; (iv) too few runaways: n_res < 1e13 m^-3 for the QL closure with Gamma_th = 2.7e5; (v) phase noise (P3): a source linewidth above 3 MHz moves sigma_c to 0.82 and above 10 MHz removes the threshold, which would also remove the QL clamp.
- **Bistability/hysteresis**: none in any replenishment-limited case. Non-monotonic L(b) occurs in 9 of 40 cases only for the unsaturated QL closure with a closing diffusive window, at b/b_thr = 2.6-9 and L = 1-130 MW, beyond the DIII-D coupled power (0.04-0.2 MW, klystron limit 1.2 MW). With the cap, the runaway part of L is constant (no feature) and L(b) is dominated by Gamma u_w V. A hysteretic experimental signature is therefore not predicted.

## 4. Experimental signatures, with DIII-D numbers (task 4)

b_thr,tot = 3.4e-4 (N_eff = 11.3; 3.0-3.7e-4), P_th(b_thr) = 238 kW (75-730 kW, Step 2), absorption capacity 7 kW (1.6-11) per 1e15 m^-3 of zone runaways.

| observable | plain threshold (my result) | pinning (hypothesis, QL closure, n_res 1e14-1e15) | how to tell |
|---|---|---|---|
| DBS wave amplitude b vs P_in (43 -> 76 -> 150 kW) | b proportional to sqrt(P_in - P_abs): b/b_thr = 0.43, 0.57, 0.79 (Gamma_th = 2.7e5); slope d ln b/d ln P = 0.5 (0.45-0.5 with P_abs <= 7 kW) | b/b_thr stays at 0.6-0.9, slope 0.08-0.13 over 100 kW-1 MW | slope below 0.2 over a factor 2 in P_in; requires a DBS calibration to 20% |
| ring-down after turn-off | exponential, tau = 1/Gamma_th = 3.7 us (1.5-10); runaways shorten it by < 5% (Gamma_RE,eff = n_res p/u_w = 1e4 /s per 1e15 at b_thr) | two-stage: fast (1e6-1e8 /s per 1e15, 0.01-0.1 us) while b > 0.7 b_thr, then slow thermal | time-resolved DBS or loop signal at 0.1 us resolution; also depends on the time since the previous pulse (zone refill 0.25-0.3 s) in the pinning case |
| HXR (> 7 MeV) vs P_in | onset at sigma = 0.7-0.85, i.e. b/b_thr = 0.5-0.7, P = 60-120 kW at the median Gamma_th; a 10x rise of the absorbed (and scattering) power between sigma 0.54 and 0.70 (measured, Sec. 2.2) | onset the same, then constant | HXR drop saturates in both; no distinction |
| ECE / synchrotron (extra radiation of the zone electrons) | zone p_perp^2 rises 5.7 -> 26-44 (x 4.5-7.7) above sigma = 0.8: extra radiated power 0.38 x 20-38 = 8-14 m c^2/s per electron, 15-25 kW per 1e15 resonant electrons in 18 m^3, taken from the E-field; steady state about 4x the wave power (1/(gamma omega/Omega_ce) = 4.3) | rises linearly with P_in - P_th to the full 100 kW-1 MW | extra emission linear in P_in beyond threshold is a pinning signature; saturating at < 25 kW per 1e15 is the plain result |
| threshold dependence on spacing | b_thr proportional to Delta^2 (P3 exponent 1.88 +- 0.15) | same | not discriminating |

The one discriminating measurement is the slope of b against P_in above P_th. DBS at 43 / 76 / 150 kW gives the numbers above; a ratio b(150 kW)/b(76 kW) = 1.4 (plain) against 1.0-1.1 (pinned).

## 5. What the analysis does and does not show
- Shown: the QL absorption of the DIII-D helicon by P1-shaped runaways is 9.5e6 /s per 1e15 m^-3 and agrees with full-orbit electrons (0.6-0.8); the wave work per electron is bounded by Manley-Rowe and the zone refill; absorbed power saturates at 1-8 m c^2/s per electron, 1.6-11 kW per 1e15 m^-3; therefore the amplitude is not pinned at b_thr, for the stated parameter ranges.
- Not shown: the physical-time value inside 1.1-7.7 (lambda = 1 not reached); the DIII-D runaway distribution (P1 is B = 2 T, n = 1e18, E/E_c = 6.5; with tau_rad/tau_c = 10.7 the DIII-D fixed point has u_perp^2 about 80); the zone density n_res (Step 2: 1e15 from 100 kA / 2 m^2; 4e15 is the upper bound of the current); the spatial structure of W (a core-concentrated wave raises the local b and the local absorption by up to sqrt2); whether the antenna phase noise is below 3 MHz.
- The pinning conclusion is robust to a factor 4 in the cap, 3 in the QL rate (the QL absorption is already saturated within 1 us), and 2.3 in the reduced-model wave work. A plateau would need an absorbed power that grows with b beyond b_thr, i.e. a rate-limited rather than replenishment-limited absorber, which requires the zone to be refilled 100-1e5 times faster than the E-field transit (the cap would then be 100-1e5 times higher). I found no mechanism for that. A constant offset P_cap (up to 45 kW at n_res = 4e15) can still shift the apparent power threshold for wave growth when Gamma_th is at the low end.

## 6. Files
`scripts/`: `qlib.py` (QL energy balance), `fulop_validation.py`, `fulop_check2.py`, `fulop_eq24.py`, `collisional_damping.py`, `diiid_ql.py`, `t_coupling.py`, `t_anom.py`, `redpin.py`, `pinrun.py`, `fullorbit_check.py`, `fullorbit_transition.py`, `ql_vs_particles.py`, `pin_model.py`, `scan_model.py`, `pin_metrics.py`, `hysteresis_test.py`, `make_figs.py`, `summarize.py`, run scripts `run_scan1.sh`, `run_scan2.sh`.

