## Why
L2 showed that at B=2 T, n=1e18, T=100 eV, E=5 mV/m synchrotron drag sets a fixed point at 6.2 MeV, so a minimum-RF problem for the 10 MeV threshold is degenerate there. This PR documents that as a bounded negative result and runs the full control program in one declared alternative scenario.

## Changes
- `examples/3_advanced/runaway_rf_minimum.py` (stages: scenario, validate, source, opt_single/opt_random/opt_es, verify, margin) and `docs/numerics/runaway_rf_control.md`. No production code changed.
- Scenario: the synchrotron fixed point never disappears with E; it crosses 10 MeV at 6.26 mV/m. Declared study choice E = 15 mV/m (fixed point 58 MeV, 5.6x the threshold in gamma), hold 2.70 s from the measured regrowth gamma 5 -> 10 MeV, RF window 0.3 s. Collision and radiation strengths unchanged.
- Slow RF response: counter-propagating cold whistler, quasilinear pitch/energy diffusion along the wave-frame curve, compared with a resolved relativistic Boris push on 14 windows.
- Source energy: explicit drive current J_drive = curl B/mu0 - eps0 dE/dt - J_plasma, P = -J_drive.E, ideal nonrecovering port Q+ with zero-crossing quadrature; returned and net energy reported separately.
- Search: single band, random search (3 bands) and evolution strategy (3 bands), 240 evaluations each, frozen-control checks on independent seeds, dt/2, 2x nodes, +-30 % amplitude.

## Validation
- rf_source_control.py reproduced on office (jax 0.6.2): Maxwell residuals, refinement gate, derivative checks and the source-cost optimum (controls 0, 2.5, 0.267, 0) match the reference. Its field-exposure optimizer hit the iteration limit there, which does not affect the source-cost result.
- Resolved vs quasilinear, held-out windows: pitch-variance growth ratio 0.92 to 1.67 (mean 1.3), mean energy shift 1.1 to 2.0 times larger when resolved. Coherent, sparse and dense-line cases fail and fall back to the resolved push.
- Source ledger: closure 1e-6, supplied energy changes < 1e-6 under refinement, steady power within 3 % of the closed form.
- Best verified feasible control (3 bands, evolution strategy): Q+ = 5.1 J/m3 over 0.3 s (4.2 to 6.1 over the checks; 5 to 9 if the quasilinear tail exchange is low by the measured factor), tail fraction above 10 MeV at the end of the hold 1.6 % (upper bound 1.8 % on independent seeds); no RF gives 100 %. Single band 8.2 J/m3, random search 9.0 J/m3, same budget. DC Ohmic work over the 3.0 s horizon is roughly 1 kJ/m3.
- flake8 clean, sphinx -W ok.

## Limits
- The constraint holds at the declared end of the hold only: the controls park the tail at the synchrotron fixed point of the scattered pitch (9 MeV mean) and with RF off 34 to 73 % exceed 10 MeV after another 0.5 s, 100 % after 1 s. With 2 s added to the hold no strategy finds a feasible control (smallest fraction 99.8 %), a bounded negative result for a 0.3 s window, three bands and b <= 1e-5.
- One check at 30 % lower amplitude fails for the best control (2.6 %).
- Most of Q+ is energy taken by the resonant tail, not bulk absorption (0.2 W/m3). The controls need noise-like bands, about 1e4 lines per band.
- Not modeled: avalanche, bremsstrahlung, transport, antenna loading and wall-plug efficiency, mode conversion, bulk heating by the wave. The ideal port is a lower bound.
