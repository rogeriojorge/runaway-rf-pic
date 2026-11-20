# Step 2: wave damping and the map from coupled power to wave amplitude (DIII-D helicon, P3 follow-up)

Script: `scripts/helicon_damping.py` (numpy/scipy/matplotlib, one thread, about 1 minute; laptop only, the office was not used). Outputs: `data/step2.json` (main, core T_e >= 2 keV), `data/step2_T1500.json` (sensitivity, core T_e >= 1.5 keV), `data/mc_samples*.npz`, logs `data/run_*.log`, figures `fig1_damping.png`, `fig2_array_and_map.png` (and `_T1500`). Nothing committed or pushed.

## Verdict (short)
**Undetermined, leaning below threshold.** With the damping computed, the 43 and 76 kW shots map to wave amplitude b_tot = 1.3e-4 (6.5e-5 to 3.3e-4, 95% band) and 1.8e-4 (8.7e-5 to 4.4e-4), against the overlap threshold b_tot,thr = 3.0e-4 to 3.7e-4 (3.4e-4 with the GENRAY spectrum). The median ratio b/b_thr is 0.39 at 43 kW, 0.52 at 76 kW and 0.69 at 150 kW. The probability of being above threshold is 6% (43 kW), 17% (76 kW), 24% (150 kW) in the baseline scenario; it is 3%, 7%, 15% when slow-wave conversion at the edge is added, and 22%, 33%, 38% if the core T_e is allowed down to 1.5 keV. The band is a factor of about 3 wide (1 sigma), so a factor-2 miss in either direction cannot be excluded: P(0.5 < b/b_thr < 2) is only 0.4-0.7. DIII-D therefore most likely sits a factor 1.5-3 below the overlap threshold in b, at or above it in about one case in five. Two reasons that this is not a refutation: (i) the damping is dominated by electron Landau damping in the hot core, which depends on exp(-zeta^2) with zeta = 2.2-3.2, so it is uncertain by a factor 100 in the core and about a factor 10 after volume averaging; (ii) the edge/antenna loss term is a bounded model with ranges, not a calculation. The P3 data (small effect at 43 kW, strong effect at 76 kW) fall in the lower 11% of the predicted P_thr distribution (baseline), so the model and the shot ordering are only marginally consistent; if the shots show a sharp threshold between 43 and 76 kW, the mapping needs Gamma_P of about 5e4-1e5 /s, at the low end of what I compute (median 2.7e5 /s).

New finding that changes how the P3 mapping should be read: the **runaways themselves are the dominant sink once the interaction is stochastic** (Section 5). Their quasilinear damping rate is about 1e8 /s per 1e15 m^-3 of resonant runaways, far above every other loss. P = Gamma_P u_w V therefore holds only below the overlap threshold, with Gamma_P the non-runaway loss; above it the amplitude is pinned near the threshold and the extra power goes into the runaways. This is a model prediction, not an observation.

## 1. Damping model and dispersion check

Fast wave, f = 476 MHz, B along z, fields exp(i(k.x - wt)), k = (k_perp, 0, k_par). Cold dielectric tensor K with Stix S, D, P (Stix, Waves in Plasmas, ch. 1-2) for electrons, deuterium and carbon (Z_eff = (n_D + 36 n_C)/n_e). The dispersion relation is det[K + nn - n^2 I] = 0, which is a quadratic in n_perp^2; the smaller root is the fast wave, the larger the slow wave.

Checks (script, `data/step2.json: dispersion_check`):

| case | this work | reference |
|---|---|---|
| n_e = 0.8e19, B = 1.4 T, k_par = 41 /m, fast k_perp | 76.5 /m | 76 /m (PRL modelling, KEY_FACTS sec. 8; P2/P3 notes) |
| n_e = 0.5e19, B = 1.4 T, n_par = 3, fast k_perp | 70.7 /m | 69 /m (PPCF App. B, journal p24) |
| same, slow k_perp | 1344 /m | 1360 /m (PPCF App. B) |

Cold group velocity at the core (0.65e19, 1.45 T, k_par = 41): v_g = (0.10 perp, 0.31 par) c, |v_g| = 0.33 c (at n = 0.8e19, 1.4 T: 0.09 and 0.28 c, which is the range P3 used). The wave energy is W = eps0/4 E^H d(wK)/dw E + |B|^2/(4 mu0); W/(|B|^2/4 mu0) = 1.09-1.14, so using the magnetic energy only (P3) is a 10% approximation. Power damping rate (energy loss rate) is Gamma_P = P_abs / W, the quantity called Gamma_P in the P3 notes, with P_abs the time-averaged absorbed power density.

### 1a. Collisional damping
Spitzer-type nu_ei from PPCF eq. (B9) (journal p23, SI form in the script). The anti-Hermitian part of K from electron collisions (omega -> omega + i nu in the electron response; equals PPCF eqs. (5)-(8)) gives P_abs = (eps0 w/2) E^H [(K - K^H)/2i] E. Gamma_coll is obtained two ways, from this and from the complex root of the dispersion relation; they agree to 0.1% at every T_e (table). The result is Gamma_coll = 0.05 nu_ei for the oblique fast wave, which is in line with the parallel-whistler limit 2 nu w/w_ce = 0.024 nu_ei, and consistent with PPCF eq. (11) (Gamma proportional to nu_ei, no other T_e dependence, hence T_e^-3/2).

**Unresolved:** Fulop 2006 eq. (25) (KEY_FACTS sec. 5) takes the collisional damping as 1.5/tau_ei, about 30 times my value for a whistler. I did not trace the difference (different branch, or a transcription problem). It is irrelevant at 2 keV: even with 1.5 nu_ei the volume-averaged rate is only 1.7e4-5e4 /s for the profiles used here, below the other terms. It matters for the post-disruption cold plasma.

### 1b. Electron Landau damping and transit-time magnetic pumping (n = 0)
For Maxwellian electrons with k_perp rho_e << 1 (k_perp rho_e = 0.01 here) only the n = 0 term is resonant at zeta = w/(|k_par| v_te) = 2.2-3.2; cyclotron resonances sit at zeta ~ 200. The resonant parallel force on an electron is F = -e E_z - i k_par mu B_z, with mu = m v_perp^2/(2B), <mu> = T/B, <mu^2> = 2T^2/B^2. The absorbed power is the 1-D Landau result with e^2|E|^2 replaced by <|F|^2>:

P_abs = w n_e sqrt(pi) zeta exp(-zeta^2) <|F|^2> / (m_e k_par^2 v_te^2), v_te^2 = 2T_e/m_e,  <|F|^2> = e^2|E_z|^2 + 2 e k_par (T/B) Re(i E_z* B_z) + 2 k_par^2 (T/B)^2 |B_z|^2.

(From Im chi_zz = 2 omega_pe^2 sqrt(pi) zeta exp(-zeta^2) / (k_par^2 v_te^2), Stix ch. 11 / Landau; the TTMP part is the mu grad B force, Stix sec. 11-4.) E and B are the cold fast-wave eigenvector (E_z/E_x = 0.011). This reproduces the scaling quoted in PPCF eq. (B12) (journal p24), Gamma/w ~ beta_e zeta exp(-zeta^2). I did not check it against a full hot-plasma tensor. Corrections of order 30% are expected from relativistic mass increase and from the warm correction to E_z.

In this regime Landau (E_z) dominates TTMP by a factor of 30-100 in power. Ion damping and cyclotron damping are negligible.

Local rates, B = 1.4 T, Z_eff = 1.5, k_par = 41 /m, n_e = 0.8e19 (Gamma in /s):

| T_e [eV] | nu_ei | zeta | collisional (anti-Herm.) | collisional (complex root) | Landau + TTMP |
|---|---|---|---|---|---|
| 2 | 1.1e8 | 87 | 5.5e6 | 5.5e6 | 0 |
| 10 | 1.3e7 | 39 | 6.2e5 | 6.2e5 | 0 |
| 50 | 1.3e6 | 17 | 6.2e4 | 6.2e4 | 0 |
| 300 | 1.0e5 | 7.1 | 4.8e3 | 4.8e3 | 1e-11 |
| 1000 | 1.8e4 | 3.9 | 8.6e2 | 8.6e2 | 3.4e3 |
| 1500 | 9.9e3 | 3.2 | 4.8e2 | 4.8e2 | 3.0e5 |
| 2165 | 5.8e3 | 2.6 | 2.8e2 | 2.8e2 | 3.9e6 |
| 3000 | 3.6e3 | 2.3 | 1.8e2 | 1.8e2 | 1.8e7 |

At the runaway-plateau temperatures of a few eV to tens of eV, collisions give 2e5-5e6 /s (not negligible: one chord of 2 m at 1e8 m/s takes 2e-8 s, so absorption per chord is 0.005 at 20 eV and 0.1 at 2 eV, at this low density; the PPCF post-disruption case has 5e19 m^-3 and a longer path, and finds the wave is lost). At the DIII-D QRE core T_e of 2-3 keV collisions are negligible (about 2e2 /s locally, 4e3 /s volume averaged) and Landau damping takes over. PRL p2 says Landau damping is expected to be weak at low beta_e; at the core it is not negligible for zeta < 3, but it acts only inside the hot core.

### 1c. Volume average
Profiles n = n0[0.05 + 0.95(1 - rho^2)^a_n], T = 50 eV + (T0 - 50)(1 - rho^2)^a_T, k_par from 30 /m at the edge to k_c at the axis (PPCF Fig. 6, KEY_FACTS), a_n in 0.3-1, a_T in 1-2, wave energy density taken uniform over rho < 0.95. Central case (n0 = 0.65e19, T0 = 2165, B = 1.45, k_c = 41): <Gamma_coll> = 7e2, <Gamma_Landau> = 1.7e5, <Gamma_TTMP> = 1.3e3 /s; the core-only Landau rate is 2.5e6 /s.

## 2. Single pass or multipass
Absorption per chord through the axis (length ell = 1.2-3 m, speed |v_g|, integrating the local thermal Gamma): median 1.5e-2, 97.5th percentile 7e-2, P(>0.1) = 1% (`single_pass_chord`). The wave is absorbed by thermal electrons only a few percent per transit, so it is **multipass**; the 1/e lifetime is about 70 chords, 1-5 microseconds. This agrees with the PRL and PPCF App. B statement (many passes). The P3 note excluded single pass; here the thermal single-pass absorption is 0.2-7% at any chord. The runaway sink is different (Section 5): if resonant runaways exceed about 1e13 m^-3 the wave is absorbed in one pass.

## 3. Total Gamma_P and Q (non-runaway loss)

Thermal terms (volume averaged, Monte Carlo of 300 draws; ranges: n0 0.5-0.8e19 (PRL text, Table 2), T0 2-3 keV (PPCF "at least 2 keV", PRL 2.165 keV), Z_eff 1.2-2 (1.5 stated), B 1.40-1.70 T (Table 2 on-axis 1.46-1.71), k_c 36-44 /m, profile exponents as above). 2.5/16/50/84/97.5 percentiles:

| term | Gamma [1/s] |
|---|---|
| collisional | 4e2, 9e2, 4.3e3, 1.2e4, 1.7e4 |
| Landau | 2.7e4, 6.7e4, 2.5e5, 5.9e5, 1.2e6 |
| TTMP | 1e2, 4e2, 2.1e3, 7e3, 1.7e4 |
| thermal total | 3.6e4, 7.6e4, 2.5e5, 6.0e5, 1.2e6 |

Wall and antenna losses (the cavity picture). The wave bounces between edge reflections at the fast-wave cutoff; each edge hit near the antenna tunnels through the vacuum gap and is absorbed by the antenna/Faraday screen, which by reciprocity is the same tunnelling that couples the power in. Loss per hit p = f_ant T_gap + eps_wall + c_conv, T_gap = exp(-2 kappa d), kappa = sqrt(k_par,edge^2 - k0^2) = 28 /m (k_par,edge = 30 /m, PPCF Fig. 6), gap d = 4.5 cm (ideal) or 9 cm (non-ideal) (PRL p3; PPCF p5, p15), so T_gap = 0.08 and 0.006. Rate = p |v_g| / ell. Parameters with no source in the papers, bounded log-uniformly: f_ant (probability that a hit lands on the antenna) 2e-3 to 5e-2; eps_wall (resistive tiles, ports, probes) 1e-5 to 1e-3; ell 1.2-3 m. The antenna area and the vessel wall loss were not found in the literature I have; these three ranges are my estimates. Results: Gamma_edge median 1.1e4 /s (4e3-3e4, 16-84%) for the non-ideal gap and 5.4e4 /s (2e4-1.4e5) for the ideal gap. Scenario B adds conversion of the fast wave to the slow wave at the edge, with c_conv per hit 1e-4 to 1e-2 (slow wave Landau-damped at k_perp = 1360 /m, PPCF App. B); it raises the edge term to 7e4 and 1.4e5 /s medians. Whether conversion occurs is unknown.

Total (thermal + edge), 16-84% (median):

| scenario | Gamma_P [1/s] | Q = w/Gamma |
|---|---|---|
| A, non-ideal gap | 9e4 - 6e5 (2.7e5) | 5e3 - 3e4 (1.1e4) |
| A, ideal gap | 1.5e5 - 7e5 (3.2e5) | |
| B, non-ideal | 1.8e5 - 8e5 (4.0e5) | |
| B, ideal | 2.3e5 - 8e5 (4.8e5) | |
| 95% range, all | 4e4 - 1.4e6 | |

The P3 notes asked for 1.4e4-3.5e5 /s (5.4e4-9.6e4 for the central values) to be consistent with a threshold between 43 and 76 kW. My median is 3-5 times the central window and the 16th percentile of scenario A (9e4) just touches its upper edge.

## 4. Map from power to amplitude

P = Gamma_P u_w V_p, u_w = b_tot^2 B0^2/(4 mu0), V_p = 15-22 m^3, b_tot = b_pk sqrt(N_eff). Threshold: b_thr = 1.0e-4 per line (0.6-1.6e-4, treated as a lognormal with that range at +-1.5 sigma) times sqrt(N_eff) for N_eff in {9, 11.3, 13.5} (Section 6). Probabilities are over the Monte Carlo (300 draws). Table for the baseline scenario A; shot powers from PPCF Table 3 / PRL (KEY_FACTS sec. 8).

| shot | P | b_tot 2.5%, 50%, 97.5% | median b/b_thr | P(b > b_thr) | P(0.5 < b/b_thr < 2) |
|---|---|---|---|---|---|
| 201951 non-ideal | 43 kW | 6.5e-5, 1.3e-4, 3.3e-4 | 0.39 | 0.06 | 0.38 |
| 201953 non-ideal | 70 kW | 8.3e-5, 1.7e-4, 4.2e-4 | 0.50 | 0.15 | 0.49 |
| 201952 non-ideal | 76 kW | 8.7e-5, 1.8e-4, 4.4e-4 | 0.52 | 0.17 | 0.51 |
| 201950 ideal | 120 kW | 1.1e-4, 2.0e-4, 4.4e-4 | 0.62 | 0.17 | 0.62 |
| 201948 ideal | 130 kW | 1.1e-4, 2.1e-4, 4.6e-4 | 0.64 | 0.21 | 0.65 |
| PRL maximum | 150 kW | 1.2e-4, 2.2e-4, 5.0e-4 | 0.69 | 0.24 | 0.68 |
| GENRAY bound | 160 kW | 1.2e-4, 2.3e-4, 5.1e-4 | 0.71 | 0.27 | 0.70 |

Scenario B (conversion): P(above) 0.03, 0.07, 0.15 for 43, 76, 150 kW. Sensitivity T0 allowed from 1.5 keV (`_T1500`): 0.22, 0.33, 0.38 for the same shots (A) and 0.09, 0.15, 0.25 (B). The threshold power P_thr = Gamma_P V u_w(b_thr,tot), non-ideal gap, scenario A: median 280 kW, 16-84% 75-730 kW, 95% 24-2300 kW; the probability that P_thr falls below 43, in 43-76, in 76-150, above 150 kW is 6%, 11%, 17%, 66% (scenario B: 3%, 4%, 14%, 79%). If core T_e reaches 1.5 keV the lower tail grows (22%, 12%, 15%, 52%).

So: **below threshold in the median, within a factor of 2 of it in 40-70% of the draws, above it in 6-27%** (scenario A). The data ordering (43 kW small effect, 76 kW strong) is a weak test, since the mapping puts 43 to 76 kW in the lowest 11% of P_thr. The paper itself says the existence of the threshold is unproven (KEY_FACTS sec. 8). Also: every probability above assumes the comb is phase coherent (P3: 3 MHz or more of phase noise removes the sharp threshold) and that b_thr, calculated in a uniform slab at B = 1.4 T, applies at 1.47-1.7 T.

## 5. Damping on the runaways
Model (`gamma_RE`): quasilinear normal-Doppler (n = +1) absorption by counter-propagating runaways at gamma = 16 (8 MeV), pitch 0.2, with the lines of the comb treated as a continuum of spacing Delta = 0.57 /m (Pokol 2008 eqs. 11-16, PDF p5-6, coupling psi of eq. 12; Fulop 2006 eq. 15). Per electron, dE/dt = pi c^2 e^2 |psi|^2 eta / (E Delta |v_par|), eta = 1 - k_par v_par/w = 5.0 (the factor 1 + n_par beta, the wave field boosted into the runaway frame), derived from d<p_perp^2>/dt = 2 pi F^2 delta(detuning) and Delta E = c^2 Delta(p_perp^2)/(2 E eta) (Manley-Rowe). The cold fast-wave eigenvector gives |psi|^2/|E|^2 = 0.45.

Result: Gamma_RE = 1.1e8 /s per 1e15 m^-3 of resonant runaways (0.9-1.55e8 over gamma 12-20 and pitch 0.1-0.3). Runaway density from the current: 100 kA over about 2 m^2 gives n_r = I/(e c A) ~ 1e15 m^-3 (my estimate; the papers do not state n_r; a value 1e14-5e15 is plausible); the resonant fraction in the 5-10 MeV window is unknown, 0.1-1.
- A resonant density of only 3e12 m^-3 gives Gamma_RE = Gamma_nonRE (3e5 /s). Above that the runaways dominate the damping.
- At the overlap amplitude b_tot = 3.4e-4 (u_w = 0.048 J/m^3, W = 0.9 J in 18 m^3) a stochastic wave would deliver about 100 MW per 1e15 m^-3, i.e. 1 MW for 1e13. Available power is 0.04-0.16 MW. The power the runaways receive from the loop is I_RE V_loop with V_loop = E 2 pi R = 0.60 V: 30, 60, 120 kW for 50, 100, 200 kA, the same as the helicon power. At the threshold amplitude the quasilinear heating rate per runaway is several hundred times the field's acceleration rate (order of magnitude).
- Consequence (model prediction): below the overlap threshold the runaway absorption is small (P3 sweep kicks of 3-15 keV per passage; of order 1e3-1e4 /s), so the amplitude follows P = Gamma_nonRE u_w V up to b_thr. At b_thr the interaction turns stochastic and absorbs more than all the incoming power, which pins the amplitude at about b_thr for any larger P; P_thr = Gamma_nonRE u_w(b_thr) V is then the onset power, and the table of Section 4 applies unchanged. It also means "DIII-D is far above the threshold in power" cannot translate into a larger amplitude. If instead the runaways scatter at the QL rate without a threshold, the amplitude at 43-150 kW would be 1.7e-5 to 3e-5 for n_res = 1e15, 10 times below b_thr. This is the strongest alternative reading, and it needs the P3 trapping suppression of D_eff to be absent.
- I did not validate Gamma_RE against a published number (Fulop 2006 eq. 24, gamma_i^max = 1.3e-9 n_r/B, is for the anomalous resonance and an avalanche distribution; with 1e15 m^-3 it gives 9e5 /s in amplitude, a factor 100 below mine for the monoenergetic resonant slice, which is the expected direction since their n_r is spread over c_Z ~ 37 momentum widths). Treat Gamma_RE as an order-of-magnitude estimate, factor 10.
- The Fulop threshold n_r/n_e > Z^2 B/(20 T_eV^1.5) is 1e13 m^-3 for the QRE core, below the estimated n_r, so the runaways would also drive their own whistlers there (consistent with the reported broadband fluctuations, PPCF p12-13). Not pursued.

## 6. Line count from the 30-module array (task 4)
Module pitch 40 deg / 30 = 1.33 deg = 5.28 cm at R = 2.27 m. For n_par = 3 at the antenna, n_phi = n_par (w/c) R = 68, phase advance per module k_par d = 90.6 deg, i.e. a quarter wave (lambda_par = 21 cm = 4 modules, which matches the "traveling-wave comb-line" description and the PRL text). The line number is the toroidal mode number n_phi (axisymmetry), so the spectrum in lines does not depend on R; R only converts lines to k_par spacing, 1/R: 0.44 /m at the antenna, 0.57 /m at R = 1.75 m. Array factor |sum_m a_m exp(i m (n delta_phi - psi))|^2 over n_phi (period 270 lines; grating lobes at n_par shifts of 12 are not propagating):

| excitation | n_par peak | FWHM [lines] | equivalent Gaussian sigma | N_eff = sum P / P_max | b_tot,thr for b_pk = 1e-4 |
|---|---|---|---|---|---|
| uniform | 3.00 | 8 | 3.4 | 9.0 (= 270/30 exactly) | 3.0e-4 |
| cosine pedestal 0.6 / 0.3 | 3.00 | 9 | 3.8 | 9.2 / 9.7 | 3.0e-4 / 3.1e-4 |
| Hann (extreme taper) | 3.00 | 13 | 5.5 | 13.5 | 3.7e-4 |
| GENRAY Gaussian, sigma_n_par = 0.2 | 3 | 10.7 | 4.5 | 11.3 | 3.4e-4 |

The GENRAY sigma = 0.2 (4.5 lines) lies between a uniformly excited aperture (3.4 lines, the P3 "low" case) and a Hann taper (5.5). The aperture cannot give fewer lines than the uniform case unless the active length exceeds 40 deg; real tapering and phase errors increase it, and plasma multipass scattering can broaden it further (PPCF p25). b_tot,thr therefore moves by -12% to +8% (3.0 to 3.7e-4) around 3.4e-4, which is inside the b_thr uncertainty; I kept 3.4e-4 as the nominal and used N_eff in {9, 11.3, 13.5} in the Monte Carlo. The uniform sinc^2 spectrum has side lobes at the 5% level and sign flips, which P3 did not simulate; the threshold was computed for smooth and flat-top spectra. If the resonance sat at the antenna radius (spacing 0.44 /m), b_thr scales as Delta^2 and falls by 0.59; the runaways resonate at R ~ 1.75 m, as in P3.

## 7. What is and is not established
Established: the dispersion relation reproduces the paper numbers (76.5/70.7/1344 /m); collisional damping is negligible at QRE temperatures (4e3 /s) and exactly computed two ways; the wave is multipass (thermal absorption per chord 1.5% median); N_eff is 9 to 13.5, so b_tot,thr = 3.0-3.7e-4.

Estimated, with stated weak points:
1. Landau damping: formula from first principles, not checked against a hot-plasma tensor, exponentially sensitive to T_e (zeta). It dominates the thermal loss in the core and sets the median of Gamma_P.
2. Edge/antenna loss: three bounded parameters (f_ant, eps_wall, ell) without a source, plus an unknown conversion probability.
3. Runaway loss: order of magnitude, n_res unknown over a factor 1e3.
4. The map assumes uniform wave energy over V_p = 15-22 m^3. If the wave energy is concentrated in the core (where Landau damping makes it spatially non-uniform) the local b at the runaways is larger than the volume average; that moves the mapping toward threshold by up to sqrt(2) or more. I did not quantify it.

What would decide it: the Doppler back-scattering measurement of b at known P (the PPCF names this as planned): b_tot of 3.4e-4 at 43-76 kW means above threshold; below 1.5e-4 means below. A measured Gamma_P (wave decay after the klystron switches off, 100 ms on/off pulses, microsecond-scale resolution) would remove the largest term; my median predicts a decay time of 1/Gamma_P = 4 microseconds (1.5-10).
