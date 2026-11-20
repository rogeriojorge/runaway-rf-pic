# L4 notes: minimum supplied RF resource (PR #79, branch rf-minimum-power, base runaway-baseline)

Head SHA 6301e6963362442e7cb34f05fdf48e92dc37b0da. Raw results: this directory's data/ (validate.json, opt/*.json, opt_robust/*.json, scen/).
Office copies: ~/work/jic/results/L4/{validate,opt,opt_robust,rf_source}. Example: examples/3_advanced/runaway_rf_minimum.py.

## 1. Reproduction of rf_source_control.py (office, jax 0.6.2, x64)
Maxwell/Poynting residuals, refinement gate (field ratios 3.8-4.2), AD vs finite-difference derivatives (<2e-11) pass; source-cost optimum
controls (0, 2.5, 0.26708, 0) match the reference to 1e-13. The field-exposure SLSQP stopped at its 60-iteration limit on jax 0.6.2
(reference: converged in 44), which stops the script's later assertions but does not touch the source-cost result.

## 2. L2 scenario: bounded negative result
Fixed point gamma 13.13 (6.2 MeV). Suppressed starts (gamma 2, 3, 5, 10) peak at 13.13, a start at gamma 20 peaks at 20.00 (< 20.57 = 10 MeV):
no RF needed over any hold. The 20 MeV seed is above 10 MeV for 6.81 s without RF. The fixed point never disappears with E (no bifurcation in the
reduced ODE); it crosses 10 MeV at E = 6.26 mV/m.

## 3. Declared study scenario (numerical choice, not a prediction)
E = 15 mV/m (E/E_c 19.6, 2.4x the 10 MeV crossing field), fixed point 58 MeV (5.6x threshold in gamma). Regrowth gamma 5 -> 10 MeV: 2.70 s (hold),
gamma 3 (xi 0.5): 3.03 s, gamma 10: 1.68 s from xi 0.99. RF window 0.3 s, horizon 3.0 s. Seed gamma U[5,25], pitch U[0.05,0.5], fraction 1e-5.
No RF: 22 % above 10 MeV at t=0, 100 % at the end (mean gamma 28.8).

## 4. QL response vs resolved windows (data/validate.json)
Counter-propagating cold R whistler, normal-Doppler resonance, QL along E + |v_ph| p_par = const. Force amplitude A checked against Boris to 0.7 %.
Metric: variance growth between half and full window, resolved / QL (N=160 particles, own phase realization each; SE 0.2-0.4 in ratio):
in-domain nine cases pitch 0.92-1.67 (mean 1.30), energy variance 1.0-2.5, mean energy shift 1.1-2.0 (resolved larger, sign agrees).
Out of domain: line spacing 3x too wide (zeta 1) 0.84 and the mean energy shift changes sign; zeta 6: 0.47; single coherent line and three lines: trapped, no diffusion.
Off-resonance band: both ~0. Window 0.6 us (about two bounce periods), onset transient removed by the increment metric.
Validity domain: noise-like band, >= 32 lines at 2 w_b/dDelta_sp = 3, particle inside the band, D_max dt <= 0.01. Fall back to the resolved push otherwise.

## 5. Source ledger
Closure of the ramp energy 1e-6, supplied energy converges to < 1e-6 relative under refinement (needs piecewise-linear forcing; a frozen forcing gave a
step-dependent rectified power). Steady power within 3 % of the closed form. Ramp supplied 5e-4 J/m3 for the chosen controls; Q- equal to it.
Circular wave: no cycle-level sloshing, so the nonrecovering port only matters on the ramps. Tail exchange is the dominant cost.

## 6. Search (240 evaluations each, 512 particles, CRN seed; D_max dt <= 0.01 enforced)
| strategy | feasible | verified Q+ (J/m3) | frac above 10 MeV (8192 seeds) | checks (seed, dt/2, nodes x2, amp x1.3, amp x0.77) Q+ |
|---|---|---|---|---|
| no RF | 0 | - | 1.00 | - |
| single band (w_c 1.51e10, width 1.5, b 1.3e-5) | 2/240 | 8.2 | 0.24 % (ub 0.35 %) | 8.2, 8.3, 8.2, 9.6, 6.9; all pass |
| 3 bands random search | 2/240 (1 rejected on seeds) | 9.0 | 0.24 % | 9.0, 9.0, 8.9, 10.6, 7.3; all pass |
| 3 bands evolution strategy (6 candidates rejected on seeds) | 107/239 | 5.1 | 1.57 % (ub 1.82 %) | 5.1, 5.2, 5.1, 6.1, 4.2; x0.77 fails (2.6 %) |
ES bands (w, width, b): (8.37e9, 0.80, 9.4e-6), (1.41e10, 0.48, 2.3e-6), (1.62e10, 1.03, 4.8e-6). Q+ split: tail exchange 16.9 W/m3, bulk 0.22 W/m3.
Uncertainty: seeds/dt/nodes +-2 %; amplitude +-30 % gives 4.2-6.1; QL tail exchange low by 1.0-1.7 gives up to ~9 J/m3. Bulk DC Ohmic work in 3.0 s ~ 1 kJ/m3.

## 7. Limits found
Constraint holds at the end of the hold only: with RF off, fraction above 10 MeV after +0.5 s: 48 % (single), 34 % (random), 73 % (ES); 100 % after +1 s
(data/opt/margin.json). Robust variant (hold +2 s = 4.70 s): 0 feasible of 720 evaluations, smallest fraction 99.8 % (data/opt_robust).
So the best-found budget is for the declared horizon only; sustained suppression is not established.
Not modeled: avalanche, bremsstrahlung, transport, antenna loading, wall-plug efficiency, mode conversion, wave heating of the bulk.
Aside: the L2 note quotes thermal electron collision rate ~5e6 /s; Braginskii gives tau_e = 23 us (4.4e4 /s) at n=1e18, 100 eV (used here); not re-checked against the L2 dt requirement.

## Left
Longer or sustained RF windows and pitch-aware energy targets; PIC-level confirmation of the whistler response (coherent trapping, collisions) and antenna coupling;
more candidates / larger N to tighten the evolution-strategy margin; kinetic (CODE/DREAM) comparison.
