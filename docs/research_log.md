# Research log: runaway scattering by launched RF spectra

Dated revisions of the research plan, in order, with the results that changed it.
Result directories referred to as results/<name> are in this repository.

## 2026-11-10: program

Inputs: docs/literature/PRL_SURVEY.md, literature/READING_LIST.md, results/codes/DREAM/NOTES.md,
results/codes/OTHER_CODES.md, papers/INDEX.md and papers/KEY_FACTS.md.

### What the cross-code runs changed
- The radiation reaction is confirmed by Smilei:
  - Classical LL matches to 2e-4 of the loss.
  - The quantum/classical loss ratio matches the `gaunt` g(chi) exactly.
  - Use Niel or cLL as the quantum reference, not Smilei MC, which loses 4-11% less energy.
- The L2 synchrotron fixed point is test-particle only:
  - DREAM's kinetic steady state peaks at gamma 17, pitch 0.106 rad and mean gamma 20, against the ODE's 13.13 and 0.136 rad.
  - The cause is pitch-angle diffusion, which the ODE omits.
  - The "no runaway above 6.2 MeV" wording must be withdrawn or restricted to the ODE.
- L2's avalanche estimate is about 2x low:
  - DREAM kinetic gives 0.053 /s at E/Ec = 6.54; Rosenbluth-Putvinski gives 0.069 /s.
  - The kinetic threshold is E/Ec of about 1.4.
- DREAM has no wave operator.
  - The RF comparison needs a quasilinear term added to a DREAM fork, or a KORC single-wave run.
- CODE and NORSE are blocked without MATLAB, or Octave in an isolated conda env.

### Thesis (from PRL_SURVEY, rank 1)
- Claim: coherent helicon or whistler drive scatters runaways at a trapping-limited rate, not the quasilinear one.
- Control then needs spectral overlap, with a Chirikov-type criterion; a single line saturates.
- This would give a mechanism for the DIII-D helicon power threshold (Choudhury 2025-26).
- Kill criterion: the PIC scattering rate follows quasilinear theory up to the experimental amplitudes.

### Work packages
- P1. Fix the baseline. Run JAX-in-Cell 0D3V with collisions and RR, with pitch-angle diffusion resolved.
  - Gate: reproduce DREAM's kinetic bump (gamma mode 17 +/- 5%) and avalanche rate (+/- 10%).
  - Correct the #77 text accordingly.
- P2. Single-wave scattering scan in JAX-in-Cell:
  - effective pitch-angle diffusion versus wave amplitude, at DIII-D parameters taken from KEY_FACTS;
  - locate the quasilinear-to-trapping crossover against the bounce frequency w_b.
- P3. Multi-line spectra: suppression versus line number and spacing; test the overlap criterion 2 w_b / dDelta of about 3.
- P4. Independent checks:
  - a DREAM fork with a Kennel-Engelmann quasilinear term (the quasilinear limit);
  - a KORC single-wave run (request inputs from ORNL);
  - Smilei for radiation reaction (done).
- P5. Use the differentiable gradients to find the minimum-power spectrum under the overlap constraint. This is the supporting method, not the claim.
- P6. Compare against the DIII-D power threshold. Needs data or digitised figures from Choudhury 2026.
- P7. Write the PRL draft and a companion PoP/JPP methods paper.

Order: P1 then P2, and P2 is the go/no-go for the thesis; P4a runs in parallel; then P3, P5, P6, P7.

### Revision 2026-11-11: novelty after reading the primary sources (papers/KEY_FACTS.md)
- Coherent-wave trapping is NOT new:
  - Kang, Yoon, Cho and Yun, JPP 92 E101 (2026), give a trapping threshold of b > 4.75e-4, and scattering about 1e6 times faster than quasilinear.
  - Correction: Kang et al., NF 64 106025 (2024), is a collisionless beam-instability PIC with no trapping; only Kang 2026 has trapping. Ghai 2025 shows non-diffusive kicks.
  - Thesis A is narrowed to what remains open:
    - (i) a multi-line overlap criterion and the required launch bandwidth;
    - (ii) collisions plus radiation reaction acting on the trapped-island dynamics, which no prior work treats self-consistently;
    - (iii) a quantitative amplitude threshold tied to the DIII-D power data.
  - The DIII-D power threshold is hedged in the published PPCF ("further data is needed"). Do not build on it as an established fact.
- Read before P2: Guo, McDevitt and Tang, PoP 25 032504 (2018) and PPCF 59 044003 (2017); Kang et al., NF 64 106025 (2024); Karney and Fisch 1986.
- Rosenbluth-Putvinski eq. 18 has a final bracket that the DREAM table omitted.
  - The corrected kinetic/RP ratio is 0.49-0.85.
  - The P1 avalanche gate is against DREAM kinetic, not RP.
- Aleynikov-Breizman is valid only for tau_rad/tau_c > 5. Our case has 0.58, so the DREAM bump at gamma 17 is outside their theory and must be explained by our own pitch-resolved runs.

### Revision 2026-11-11b (docs/literature/PRL_SURVEY.md, "Novelty verdict")
- Overlap/bandwidth law: open.
- Collisions + RR in the trapped regime: open only in the non-quasilinear regime. Guo 2017/2018 already cover the quasilinear regime.
- DIII-D amplitude threshold: open but weak.
- A PRL needs P2/P3 to give a sharp boundary in (amplitude, collisionality, line spacing) that joins Guo's quasilinear regime to Kang 2026's trapped regime. Otherwise target PoP/NF.
- Fallback: an adjoint minimum-power spectrum (Karney-Fisch 1986 extended with RR and the wave term; Guo 2018 calls this open).

### Cross-code status 2026-11-11 (results/codes/CODE_NORSE/NOTES.md)
- CODE and NORSE run under conda Octave 10.3, with local patches.
  - The CODE port reproduces stored MATLAB bump positions to within 2%.
- Avalanche at E/Ec = 6.54/20/50:
  - CODE: 0.073/0.238/0.65 /s; DREAM kinetic: 0.053/0.196/0.523 /s; RP: 0.069/0.237/0.612 /s.
  - CODE is 20-40% above DREAM; the cause is unresolved (source threshold or runaway definition).
  - The P1 gate is the DREAM-CODE band, not a single value.
- Synchrotron bump at T = 100 eV: CODE is unstable, and NORSE needs nP of about 2000. DREAM (mode gamma 17) is the only kinetic reference.
- NORSE: Spitzer current exact, J.E = 32.3 W/m3, bulk synchrotron loss 24.9 W/m3. This explains JIC's 34.7 W/m3 magnetic emission as about 25 from the bulk plus about 10 from the seed.

### P1/P2 results 2026-11-11 (results/P1, results/P2)
- **P1:**
  - Peak gamma: JIC 17.65 vs DREAM 17.0. Pitch: 0.101-0.105 vs 0.106 rad. Gate met.
  - Mean gamma is +4-5% after the p<40 cut, unexplained.
  - Pitch-noise scan: zero noise recovers the ODE's 13.16; full noise gives 17.4.
  - No knock-on source; decision documented.
  - #77 text corrections drafted in results/P1/NOTES.md section 4.
- **P2: GO, conditional.**
  - A coherent single line is trapping-limited at all candidate amplitudes: DIII-D 3e-5..5e-3, Guo, Kang.
  - Crossover b_x ~ 3.5e-9, with D_eff ~ b instead of b^2. D_eff/D_QL ~ 1e-5..1e-4 at DIII-D amplitudes, an extrapolation.
  - Two lines: the transport jump obeys Chirikov sigma = 2(w_b1+w_b2)/Delta ~ 0.8-1.0.
  - For toroidal sideband spacing 1/R0, sigma = 1 at b = 5.6e-5, inside the DIII-D power range.
- **Candidate PRL claim:** the helicon "power threshold" is the overlap of toroidal-sideband resonance islands.
  - Below it, scattering is island-limited and weak.
  - Above it, it is stochastic and quasilinear.
  - Testable: the threshold scales with spectrum spacing and goes as b_thr ~ Delta^2.
- **Next (P3):**
  - the antenna's real n_phi/k_par spectrum (Choudhury PPCF);
  - an N-line overlap and D_eff map at physical lambda, with collisions and RR (fix the discarded collision runs);
  - the threshold in kW with an uncertainty band.
  - P4a: a DREAM fork with a QL term above the threshold.

### P3 result 2026-11-11 (results/P3/NOTES.md): UNDETERMINED
- Coherent comb, stationary measure:
  - D_eff/D_QL collapses on Chirikov sigma; the 50% point is at sigma 0.97-1.12.
  - Threshold b_thr = 1.0e-4 per line (0.6-1.6e-4); b_tot = 3.4e-4 for the GENRAY Gaussian spectrum.
  - Spacing scaling: exponent 1.88 +/- 0.15, consistent with 2.
- Collisions are irrelevant at DIII-D (nu_eff/w_b ~ 1e-6).
- Phase noise of 3 MHz or more blurs the threshold, so it exists only for a coherent launch, and source coherence is unknown.
- Passage through resonance under the E-field sweep: no sharp jump.
- kW mapping: consistent with the 43/76 kW shots only if damping Gamma_P ~ 1e4-3e5 /s. Single pass is excluded.
- Falsifiable predictions:
  - (i) DBS-measured b of about 1e-4 per line at the threshold power;
  - (ii) 3-10 MHz source FM lowers the threshold by 0.45-0.6x and removes the high-b cutoff;
  - (iii) a CW power scan shows an enhanced window at b = 1-5e-4.
- Status: a solid PoP/NF paper with experimental predictions. PRL only if DIII-D data, such as a DBS b or an FM shot, confirm a prediction.
- Open:
  - the full-orbit vs reduced-model mismatch in passage (1.5-10x at lambda = 1000);
  - the capture/locking effect at sigma 2.2;
  - the dt bias at sigma 1.4.

### Step 2 result 2026-11-11 (results/step2/NOTES.md): undetermined, leaning below threshold
- Damping is dominated by Landau + TTMP (exp(-zeta^2), zeta 2.2-3.2).
  - Gamma_P median 2.7e5 /s (95% range 4e4-1.4e6). The wave is multipass, about 1.5% absorbed per pass.
- b/b_thr medians: 0.39 at 43 kW, 0.52 at 76 kW, 0.69 at 150 kW.
  - P(above threshold): 6% at 43 kW, 17% at 76 kW, 24% at 150 kW.
  - Predicted P_thr median 280 kW (75-730 kW).
- Runaways become the dominant sink above about 3e12 m^-3 resonant density.
  - Hypothesis: the amplitude is pinned near b_thr, which acts as a self-limiting threshold. This is a candidate new physics point; validate the absorption rate.
- The array factor gives N_eff 9-13.5, so b_tot,thr is 3.0-3.7e-4 (nominal 3.4e-4 kept).
- To do:
  - check Landau/TTMP with a hot-plasma tensor;
  - resolve the Fulop eq. 25 collisional-damping factor of 30;
  - validate the runaway absorption against a published number;
  - quantify core concentration of the wave energy.
- Decisive measurements: DBS b at a known power, or the wave decay time after the klystron turns off (predicted about 4 us).

### Pinning test 2026-11-11 (results/pinning/NOTES.md): REFUTED as formulated
- Validated QL absorption: 9.5e6 /s per 1e15 m^-3 at DIII-D, 5-12x below the Step 2 estimate.
  - Code check: reproduces Fulop 2006 eq. 21 to 6-8% at small k. Fulop eq. 22 has a typo (omega0 should be omega0^2).
- Full-orbit wave work turns on sharply at sigma 0.83-1, but reaches only 7-13% of QL at sigma 0.99.
- Absorption is replenishment-limited: the wave flattens f within about 1 us, and the zone refills at 4-5 /s.
  - Capacity is about 7 kW per 1e15 m^-3 (1.6-11), far below P_th(b_thr) of about 238 kW, so there is no plateau.
  - Unsaturated QL would clamp at 0.6-0.9 b_thr, but saturation removes it.
  - No hysteresis in the DIII-D range.
- Collisional damping: Fulop's 1.5/tau_ei is 17-62x the cold-tensor value. The cause is the regime and a kinetic coefficient; irrelevant at 2 keV.
- Consequence: the overlap threshold is an onset, not an attractor.
  - Keep the plain-threshold predictions: d ln b/d ln P ~ 0.5; b/b_thr 0.43/0.57/0.79 at 43/76/150 kW; ring-down 3.7 us.
  - New observable: synchrotron/ECE rise of about 15-25 kW per 1e15 m^-3 resonant electrons above the onset.
- Open: the lambda=1 value; the DIII-D-specific runaway distribution (tau_rad/tau_c 10.7); the 2.1-2.6x reduced vs full-orbit gap.

### Step 1 result 2026-11-11 (results/step1/NOTES.md)
- Reduced model fixed: it was missing the tangential phase term kappa = F b Omega/u_perp.
  - Passage now agrees with full orbit to 4-14%.
  - Separate bug: P3's N=1 window placeholder in runP.py.
- The sigma = 2.2 "capture" is withdrawn. What remains is a deterministic surfing lag of dgamma ~ -0.24, independent of sweep rate.
- **Critical:** for a coherent finite comb, diffusion is TRANSIENT at every sigma.
  - D_eff/D_QL < 0.03 after 1e3 beat periods for sigma 1-4 (N 5, 10). The motion is bounded chaos, not mixing.
  - So the P3 overlap collapse and threshold describe short dwell times (tens of beat periods, about 1-10 us) only.
  - Sustained quasilinear scattering at physical dwell needs decorrelation: source phase noise of 3 MHz or more, multipass, or the geometry.
  - The coherent-overlap thesis as a steady-state threshold is NOT supported.
- Plateau deficit explained as a time-compression artifact (lambda^0.5).
- Mean gamma vs DREAM: +0.9-2.7% after Richardson extrapolation, within DREAM's grid error. Closed.
- Open:
  - lambda = 1 convergence (needs omega_b dt <= 0.02);
  - the Gaussian-spectrum long window is partial;
  - no full-orbit check at 1e3 periods.

### Re-framed question
Scattering at DIII-D is controlled by the decorrelation of the launched spectrum (source linewidth, multipass phase scrambling, toroidal geometry), not by the amplitude overlap alone. Candidate thesis:
- runaway scattering by a launched comb requires BOTH overlap (sigma > 1) and phase decorrelation faster than a few beat periods;
- the coherent case saturates (bounded chaos);
- the threshold is then a 2D boundary in (b, Gamma_phi).
Next: map the sustained D_eff over (sigma, Gamma_phi) at long windows, and estimate Gamma_phi from multipass and the geometry.

### Step 3 result 2026-11-12 (results/step3/NOTES.md)
- Added independent random-walk phase noise (Lorentzian lines, HWHM Gamma_phi) to each line of the step-1 reduced model; N = 10, sigma 0.5-4, Gamma_phi/Delta_omega 0-3, T = 1000 beat periods, 3 seeds x 64 particles.
- Hypothesis supported qualitatively: late growth of <dg^2> relative to quasilinear is 0.00-0.06 at Gamma_phi = 0 for every sigma, about 0.05-0.4 at 0.03 Delta_omega, about 0.7 at 0.1 and at or above QL at 0.3 (rough law min(1, 7 Gamma_phi/Delta_omega); coherence time of about one beat period). Dependence on sigma above 0.5 is weak.
- Correction to step 1: in a finite line band, QL diffusion also has zero late slope, so "D_eff/D_QL < 0.03 at 1e3 periods" is not by itself non-QL. The coherent comb fills the band quickly (1-300 periods, at 1-3 times QL speed) to a plateau of 0.5-0.7 of complete mixing and stays there to 3000 periods; any Gamma_phi >= 0.01 Delta_omega goes past it.
- Weak points: Gamma_phi >> Delta limit only checked at sigma = 0.5 (about 40% error); Lorentzian wings heat particles far outside the band at Gamma_phi >= 0.3, sigma >= 1.5; per-cell scatter factor 2-3; dt/window checks have 1 seed.
- DIII-D: sigma about 0.6-0.8 (b/b_thr 0.39-0.69), Gamma_phi/Delta_omega <~ 1e-3 from source (Hz-kHz linewidth, and common-mode noise does not decorrelate lines) and plasma phase noise (my estimate). Bottom-left corner of the map: coherent and sub-overlap, no sustained scattering. Only parametric-decay broadening (8 MHz spacing, 0.3 Delta_omega, uncalibrated) or deliberate FM of 1-3 MHz could reach the sustained regime. Next: a source-FM power scan prediction, and band-filling kicks with E-field refill in the passage protocol.

## Conclusions and next steps (2026-11-13)

Conclusion. A coherent comb of toroidal helicon lines gives a sharp overlap threshold in the Chirikov parameter (b about 1e-4 per line, b_thr ~ Delta^1.9), but above it the motion is bounded chaos: diffusion is transient over 1-10 us and fills the line band to 0.5-0.7 of complete mixing, then stops. Amplitude pinning at threshold is refuted (absorption is replenishment-limited, about 7 kW per 1e15 m^-3). Sustained quasilinear scattering needs phase decorrelation Gamma_phi >~ 0.01-0.14 Delta_omega. DIII-D sits coherent and sub-overlap (sigma 0.6-0.8, Gamma_phi/Delta_omega <~ 1e-3), so the model does not explain the observed hard X-ray reduction unless parametric-decay broadening is large. The baseline (P1) matches DREAM's kinetic bump: the 6.2 MeV fixed point is ODE-only.

Limits: one 10-line comb, uniform plasma, no E-field refill or collisions in step 3, map good to a factor 1.5-2 per cell, cold-tensor damping, P1 runaway distribution instead of DIII-D's, lambda=1 passage not converged.

Predictions: (1) 1-3 MHz FM at fixed power turns sustained scattering on; (2) DBS pump linewidth and sideband amplitude, b about 1e-4 per line at threshold, dlnb/dlnP about 0.5; (3) 3.7 us ring-down after klystron turn-off; (4) ECE/synchrotron rise of 15-25 kW per 1e15 m^-3 resonant electrons above onset.

Next steps, in order: apply the P1 text corrections to PR #77; add the coherent-comb caveat to PR #79 text; band-filling kicks with E-field refill and collisions in the step 3 model; hot-plasma damping tensor and a DIII-D runaway distribution; physical lambda=1 passage with omega_b dt <= 0.02; design the FM power scan and DBS request with DIII-D contacts. Target PoP/NF; PRL only if a DIII-D measurement confirms a prediction. Results to PR mapping is in note/note.tex (Section "Helicon/whistler scattering": #77 baseline text, #79 caveat); the scattering scripts are in no PR.
