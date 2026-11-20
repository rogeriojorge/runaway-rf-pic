# L2 notes: no-RF homogeneous baseline (PRELIMINARY, reduced test-particle reference; kinetic confirmation pending)

Case: B=2 T, n=1e18 m^-3 (Z=1 deuterium), T=100 eV, lnL=15, E_parallel=0.005 V/m (total mean E, held by ideal J_ext=-J_plasma),
seed 20 MeV (gamma=40, pitch 0.1 rad, fraction 1e-5).

## Scales (SI, analytic)
- Collision time tau_c = 4 pi eps0^2 m^2 c^3/(n e^4 lnL) = 2.2285 s. Critical field E_c = n e^3 lnL/(4 pi eps0^2 m c^2) = 7.649e-4 V/m, so E/E_c = 6.54.
- Synchrotron: tau_syn = 1/(tau_rad (eB/m)^2) = 1.2897 s (rate a_B = 0.7754 /s). eE/(mc) = 2.933 /s.
- Thermal electron collision time (Braginskii) tau_e = 23 us, nu ~4.4e4 /s (an earlier 5e6 /s here was a miscalculated v_th); the 100 us default horizon is ~1e-5 of the seed scale.

## Reduced test-particle ODE (collisional drag gamma^2/p^2/tau_c, pitch scattering (Z+1)gamma/p^3/tau_c, E drive, Landau-Lifshitz synchrotron)
- A synchrotron-limited fixed point exists: gamma_eq = 13.13 (K = 6.20 MeV), pitch 0.136 rad (xi=0.9908). No runaway above ~6.2 MeV at 2 T, E=5 mV/m.
- The 20 MeV seed is ABOVE that fixed point. With no RF it decays: gamma(t)=39.05 (0.1 s), 32.73 (1 s), 27.10 (2.5 s), 22.54 (5 s), 18.18 (10 s), 14.61 (20 s), 13.25 (40 s).
  Initial slope dgamma/dt = -9.5 /s; e-folding time of the gap to gamma_eq = 4.64 s.
- Regrowth from a suppressed state (time for the gap to gamma_eq to close by 50% / 63% / 90%):
  gamma0=2 (xi 0.9): 4.9 / 7.1 / 16.9 s; gamma0=3 (xi 0.5): 6.1 / 8.3 / 18.1 s; gamma0=5 (xi 0.9): 6.2 / 8.5 / 18.3 s; gamma0=10 (xi 0.99): 3.5 / 5.7 / 15.7 s.
  => no-RF regrowth timescale is O(5-8 s) (63%), O(16-18 s) to 90%; any hold shorter than ~1 s sees < 20 % of the regrowth.
## Neglected physics over the horizon (order of magnitude)
- Avalanche (Rosenbluth-Putvinski, no sync): ~0.03 /s, so <= 3 % per second, <= ~35 % over 10 s; synchrotron-limited reduces it. Not in the kernels.
- Bremsstrahlung loss at gamma=13: ~5e-3 /s of gamma vs eE/mc = 2.9 /s (0.2 %): negligible.
- Primary (Dreicer) generation: E/E_D = 1.3e-3, exp(-E_D/4E)~e^-190: zero.
- Ohmic heating of the bulk ~ 33 W/m^3 (Spitzer), i.e. T_e rises ~1.4x per second: bulk must be evolved (it is) on s horizons.

## Update 1 (kinetic confirmation, jax 0.12 CPU, example `examples/3_advanced/runaway_baseline.py`, branch runaway-baseline)
- Production kinetic run (512 bulk/64 seed/512 ion markers, dt = 0.2 us, 0.4 ms): seed <gamma> change -3.967e-3 kinetic vs -3.955e-3 reduced ODE (0.3 %).
  100 us default (quick 16 seeds): -9.83e-4 vs -9.89e-4. The reduced ODE (slope -9.5 /s at 20 MeV) is therefore the matched reference.
- CORRECTION: nu_ee ~4.4e4 /s (tau_e 23 us), not 5e6; the claimed 10x ohmic overestimate at 2 us was an argument from that wrong rate, never measured. Measured (2048 markers, 4 realizations, noisy): bulk DC power 37 W/m^3 at dt 2 us, 17+-19 at 0.5 us, 28+-33 at 0.125 us vs Spitzer ~32: no dt effect resolvable. Kernel rates verified vs exact Spitzer (0.98). dt requirement is set by slow electrons (s ~ dt nu (v_th/v)^3): ~12% low exchange at tau_e/50, ~3% at tau_e/400; 0.2 us = tau_e/115 is a convergence choice (~5-8 %), dt ~ 1-2 us is adequate for seed-only quantities.
- 64-seed collision pilots from the reference reproduced exactly (jax 0.6.2 CPU, office): raw-defect mean/SE 2.769e-4/1.713e-4 (512/64/256), 6.207e-5/9.925e-5 (2048/64/256),
  -3.131e-4/2.554e-4 (512/256/256), 8.852e-6/1.689e-4 (512/64/512); seed-work means match the documented table to 4 digits; seed-0 defect 6.810571e-4 J/m^3.
  Cold-target variance table identical (R=12499.875: 5.478 vs analytic 6.103 +/- 1.22).
  (Bitwise source-equivalence assertion fails across jax versions, 3e-12..2e-8 relative: chaotic amplification, not a physics change.)
- Moment restoration (new, `collide(restore=True)` / `Collisions(restore=True)`): 64 seeds, 512/64/256: raw defect SE 2.4e-4 J/m^3 -> restored |defect| <= 3e-13;
  seed collision work -3.504e-5 +/- 1.3e-6 (raw) vs -3.537e-5 +/- 1.3e-6 (restored): no bias. Light species is bit-identical for the same key.
- Ledger convention: E = total mean field, ideal J_ext = -J_plasma; supplied energy = DC work sum_s q E.v (bulk 2.4e-2, seed 2.4e-4*(t/0.1ms) J/m^3 per 100 us).
- Production-code ensemble (example, 64 realizations, 512/64/512, 100 us, dt 0.39 us): seed <gamma> change -9.864e-4 (SE 2.2e-6 on <gamma>) vs ODE -9.888e-4 (0.24 %);
  raw collision defect mean 2.769e-4 (rms 1.39e-3) J/m^3 -> restored rms 4.8e-13; seed collision work restored -3.480e-5 +/- 1.8e-6 vs raw -3.601e-5 +/- 1.8e-6 J/m^3.
  Ohmic bulk DC work 3.57e-3 J/m^3 per 100 us = 35.7 W/m^3 (Spitzer estimate 33 W/m^3); T_e 99.93 -> 99.96 eV; magnetic emission 3.47e-3 J/m^3.

## Update 2 (5 ms kinetic runs, dt = 0.2 us, 25000 steps, 256 bulk/32 seed/256 ion markers, 4 realizations, restored collisions; office CPU jax 0.12)
| seed start | kinetic delta<gamma> (5 ms) | reduced-ODE delta<gamma> | note |
|---|---|---|---|
| gamma 40, pitch 0.1 | -4.959e-2 (SE 1.5e-4) | -4.935e-2 | 0.5 %: seed decays toward gamma_eq |
| gamma 5, pitch 0.3 | +3.232e-3 (SE 1.1e-4) | +3.321e-3 | 2.7 % (0.8 SE): regrowth, initial slope +0.66 /s |
Ledger residual (K-K0-W_DC-W_RR) <= 8e-12 J/m^3 over 25000 steps; electron density above 10 MeV unchanged (1e13 m^-3 from the seed; 0 for the gamma 5 start).

## FOR L4 (hold choice)
- No-RF dynamics at B = 2 T, E = 5 mV/m, n = 1e18: there is no unbounded runaway. Tail energy relaxes to gamma_eq = 13.13 (6.20 MeV, pitch 0.136 rad).
- 1/e relaxation time of the gap: 4.64 s from 20 MeV (gamma 40); 7.54 s from gamma = 5.
- Regrowth (reacceleration) after the tail has been suppressed to gamma0: initial dgamma/dt = +0.66 /s at (gamma 5, pitch 0.3) [kinetic-confirmed];
  gap-closing times 50 % / 63 % / 90 %: gamma0=2: 4.9/7.1/16.9 s; 5: 6.2/8.5/18.3 s; 10: 3.5/5.7/15.7 s (reduced ODE; first three confirmed to <3 % at 5 ms).
  So an RF-off hold shorter than ~1 s sees <~15 % regrowth of the gap to gamma_eq, and a 5 ms kinetic window sees <0.1 %.
- Supplied power density with E held: bulk Ohmic J.E = 35.7 W/m^3 (Spitzer 33), seed (1e-5 fraction, 20 MeV) 2.4 W/m^3; the DC work is accumulated per species.
- Kinetic horizon: dt ~0.2 us used (tau_e/115; see CORRECTION above, 1-2 us probably adequate for the seed), ~25 ms/step/core; 5 ms = 25000 steps ~ 1 h/4 realizations on a loaded office CPU.
- Neglected, bounded: avalanche <= 3.5 %/s tail-number growth (RP, sync only lowers it); bremsstrahlung ~0.2 % of eE c at gamma 13; Dreicer zero; screening none (Z=1 fully ionized);
  reabsorption none (photons escape). Not run: CODE/DREAM (needs PETSc/HDF5 build; not installable here).

## Delivery
PR https://github.com/uwplasma/JAX-in-Cell/pull/77 (draft, base relativistic-radiative-control), head b7e4b819f4312f2ed87a2ccab1c86a44851deaef.
Gate: flake8 0; sphinx -W ok; 646 passed, 2 failed not from this branch (rf_source_control.py lacks JAX_ENABLE_X64 on the base; external-xyz NaN on jax 0.6.2 only); combined coverage 100 % statements+branches.
