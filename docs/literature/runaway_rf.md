# Runaway electrons and RF/whistler control (L0 survey)

Status key: [V] retrieved, [H] reference, [K] memory, verify against source before using as a gate.

## Reference scales (computed here, SI, n=1e18 m^-3, B=2 T)
- `tau_c = 4 pi eps0^2 m^2 c^3/(n e^4 lnL)`: 3.34 s (lnL=10), 2.23 s (15), 1.67 s (20).
- Connor-Hastie `E_c = m c/(e tau_c) = n e^3 lnL/(4 pi eps0^2 m c^2)`: 5.1e-4, 7.6e-4, 1.0e-3 V/m for lnL=10,15,20.
- Dreicer `E_D = E_c m c^2/T`. Synchrotron time `tau_B = 1.29 s` at 2 T (radiation.md). (Radiation-limit energy for a given E/E_c must be computed from the CODE/DREAM balance; not asserted here.)

## Models
1. Test-particle relativistic Fokker-Planck: friction and pitch-angle scattering `nu_D`, `nu_S`, `nu_par` for full/superthermal/ultrarel. regimes (Embreus 2018; DREAM collfreq modes [V]). Full collision frequencies need partial screening for impurities (Hesslow PRL 118 255001 (2017), 10.1103/PhysRevLett.118.255001; effective critical field PPCF 60 074010, 10.1088/1361-6587/aac33e).
2. Avalanche: Rosenbluth-Putvinski (Nucl. Fusion 37 1355 (1997) [K]) growth `~ (E/E_c - 1)/(2 tau_c lnL) sqrt(pi phi/(3(Z+5)))`, `phi` an inverse-aspect-ratio trapping factor [K, verify factors]. Known to be inaccurate for modest runaway energy; Hesslow et al. 2019 (Nucl. Fusion 59 084004, fluid growth rate benchmarked against kinetic) is the better formula [search snippet confirms DREAM option]. Large-angle (knock-on) source: Embreus, Stahl, Fulop JPP 84 905840102 (2018).
3. CODE (Landreman, Stahl, Fulop, CPC 185 847 (2014), 10.1016/j.cpc.2013.12.004): 2D momentum space, E field + collisions + avalanche + synchrotron. DREAM (Hoppe, Embreus, Fulop CPC 268 108098 (2021)): `dream_avalanche` physics test compares FLUID, FLUID_HESSLOW and kinetic growth rate (run with `--plot`; numbers not retrieved offline: **L2/L4 must run it to get the table**). Examples dir: `runaway`, `runaway_partiallyIonized`, `basic`, `convergence` (chalmersplasmatheory/DREAM `examples/`) [V listing].
4. Energy-dependent critical field with kinetic instability: Liu PRL 120 265001 (2018): E_c raised and RE decay by fan instability (quasilinear).
5. RF/whistler mitigation:
   - Ghai et al., PoP 33 102501 (2026) and arXiv 2509.13271: AORSA + KORC, nondiffusive and energy-dependent pitch-angle scattering [V snippet].
   - Choudhury et al., PRL 136 025101 (2026) (arXiv 2505.19279): experiment with externally launched helicon scattering [H].
   - Guo et al.: quasilinear whistler slab, synchrotron-damping via pitch-angle scattering [snippet]. APS DPP 2025 "firewall effect" abstract: external R/whistler waves suppress acceleration [snippet].
   - Aleynikova, Komar, Spong papers: not found by searches this session; do not cite them without retrieval.
6. Quasilinear diffusion (Kennel-Engelmann): `D_pp ~ (pi e^2/ c^2) ... sum_n |E_k|^2 delta(omega - k_par v_par - n Omega/gamma)`; resonance condition for whistler `omega - k_par v_par = -Omega_ce/gamma` (anomalous Doppler/normal Doppler). Cold-plasma whistler `n^2 = 1 - omega_pe^2/(omega(omega - Omega_ce cos theta))`. Use as the analytic reference for the PIC slow-RF response; resonance location in `p_par` is the test target [K standard; verify constants].

## Proposed numerical targets (to be pinned by running reference tools)
| Target | Source | Number |
|---|---|---|
| E_c(n=1e18, lnL=15) | Connor-Hastie | 7.6e-4 V/m (computed) |
| tau_B(2 T) | classical synchrotron | 1.290 s (computed) |
| 20 MeV -> gamma 10 cooling time, B=2 T, theta=90 | coth solution | 0.0973 s (computed) |
| avalanche growth rate vs E/E_c for Z=1 | DREAM `dream_avalanche`, CODE | run; no retrieved numbers |
| whistler resonance p_par for given omega, k_par | cold dispersion + resonance | analytic, compute per run |
| RE decay with pitch-angle scattering | Liu 2018 / Ghai 2026 qualitative | no numbers retrieved |

## Cautions
Time scales (seconds) are many orders above PIC time steps; any PIC study must be rescaled (accelerated collision/synchrotron rates) and the rescaling must be preserved across tau_c/tau_B/tau_RF ratios. Fluid-level growth rates cannot validate PIC with 1D3V geometry (no trapping, phi factor).
