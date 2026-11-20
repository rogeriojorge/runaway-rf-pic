# P1 notes: JAX-in-Cell no-RF baseline against the kinetic references

Code: fresh clone of uwplasma/JAX-in-Cell, branch `quantum-recoil` (200567b), in ~/local/jic-work/P1. No library change was needed, so
there is no `p1-baseline` commit and nothing was pushed. All drivers are in `scripts/`; data (npz snapshots, logs) in `data/` (108 MB total).
JAX_ENABLE_X64=1 set before import (asserted in the driver). Laptop, 1 thread per process, at most 4 processes at once.

## Verdict
| Gate | Reference | JAX-in-Cell | Status |
|---|---|---|---|
| Synchrotron mode gamma, 17 +/- 5 % | DREAM 16.8-17.1 | 17.65 (pooled, 1.46e6 samples; runs 17.15-17.9) | met against the DREAM centre (+3.8 %); +5.1 % against its lowest grid value |
| Pitch at the mode | DREAM 0.105-0.107 rad | 0.101 (p-bin), 0.105 (gamma 16-18) | met (-4 % / -1 %) |
| Mean gamma | DREAM 19.8-20.1 (pmax 40, absorbing wall) | 21.9 untruncated; 20.9 truncated at p<40 | NOT matched: +9 % raw, +4-5 % after matching the DREAM grid cut |
| 90th percentile gamma | DREAM ~30 (pmax 40); 33.1 (pmax 60, 40 s) | 33.8; 31.6 truncated at p<40 | 33.8 vs 33.1 at pmax 60; +5 % after truncation |
| Pitch-diffusion diagnosis | | confirmed (section 3) | done |
| Avalanche +/-10 % of the DREAM-CODE band | 0.053-0.073, 0.196-0.238, 0.523-0.65 /s | none: no knock-on source | NOT reproducible by construction (section 5) |

## 1. Setup (what was run)
Production kernels only: `jaxincell._collisions.collide` (relativistic Perez rate, Nanbu angle, explicit lnL = 15) and
`jaxincell._core.landau_lifshitz` (reduced LL, B-only), the same pieces as `examples/3_advanced/runaway_baseline.py`.
Case: B = 2 T, E = 5 mV/m along B, n = 1e18 m^-3, lnL = 15 (n lnL = 1.5e19, so tau_c = 2.2285 s, tau_syn = 1.2897 s, E/Ec = 6.54), T = 100 eV, Z = 1.
`scripts/tail_steady.py`: N_t tail markers (starts: gamma 13.134 pitch 0.1361 = the ODE fixed point, or gamma 40 pitch 0.1) collide each step with
N_b frozen Maxwell-Juttner electron markers and N_b frozen deuteron markers (both pairs, Z = 1, so e-e and e-i scattering both act: (Z+1)).
Strang drift/collision/drift, midpoint-RK2 drift with LL, dt = 1e-3 s unless stated. Tail marker weight 1e-8 n/N_t, bulk and ion weight n/N_b.

**Equivalence to DREAM's operator.** Write f = f_M + f_t with n_t/n ~ 1e-8. The Landau/Boltzmann operator is C[f_M,f_M] = 0 plus the
test-particle term C[f_t; f_M] plus the field-particle term C[f_M; f_t], which is O(n_t/n) and is dropped. C[f_t; f_M] is the linear operator with
a fixed Maxwellian field population: drag, pitch scattering and the (tiny, T/mc^2 ~ 2e-4) energy diffusion. DREAM's superthermal collision operator
for the runaway grid is this linearization with fixed bulk T and n. In the kernel, a tail marker is paired once per step with a random marker of the
bulk (density n through the pair-density rule, because N_b >= N_t) and the bulk-side update is discarded, so the bulk stays at f_M exactly. Tail-tail
collisions are omitted (n_t negligible). What this is NOT: the live-bulk run of L2, where the bulk is evolved and Ohmic-heated at 33-37 W/m^3
(T_e would rise ~1.4x per second). A steady state at fixed T is only defined with a held bulk, which is what DREAM does too.

Time scales: the stationary state needs tens of seconds (mean gamma 21.4 at 50 s, 21.9 at 80 s from the fixed-point start; from gamma 40 it is
within 1.5 % by 40 s). A live bulk needs dt <~ tau_e/100 = 0.2 us, i.e. 5e8 steps for 100 s; the frozen bulk needs dt ~ 1e-3 s (the tail dynamics
are slow: scattering rate per step s ~ 1e-5), 1.5e5 steps. This is the acceleration used, and it is exact in the sense above, not an approximation of the
tail physics. A weighted live-bulk run was not attempted beyond the 5 ms of L2.

## 2. Results and comparison with DREAM
Stationary window 50-80 s (gamma-40 starts) or 60-150 s / 80-150 s (long runs), snapshots every 2 s pooled; mode of dN/dgamma over p > 3 on a
dp = 0.25 grid smoothed with sigma = 1 in p (DREAM's definition: caseb.py `moments`; pitch at the mode = arccos of the mean xi in the p bin).

| quantity | JAX-in-Cell (pooled 1.46e6) | spread over independent runs | DREAM kinetic |
|---|---|---|---|
| mode gamma | 17.65 | 17.15-17.9 | 16.8 / 17.1 (two grids), 16.97 (pmax 60, 40 s) |
| pitch at mode (rad) | 0.1012 (p-bin); 0.1055 (gamma 16-18) | 0.098-0.105 | 0.105-0.107 |
| mean gamma | 21.93 | 21.78-21.98 | 19.8 / 20.1 (pmax 40); 21.3 at 40 s (pmax 60, gamma-40 start) |
| median gamma | 20.51 | 20.3-20.6 | 19.0-19.3 (pmax 40) |
| p90 gamma | 33.8 | 33.3-33.9 | 29.9-30.2 (pmax 40); 33.1 (pmax 60, 40 s) |
| mean / median / p90 truncated at p < 40 | 20.9 / 20.1 / 31.6 | | 20.07 / 19.26 / 30.17 |
| rms pitch (rad) | 0.105 | | |
4.0 % of the tail sits above p = 40 (DREAM's b_fp_L2 grid ends at p = 40 with an absorbing wall and kept 75 % of its particles at 100 s), which
explains about two thirds of the raw mean offset. The remaining +4 to +5 % in mean, median and p90 is not explained; candidates are DREAM's upwind
numerical diffusion and 0.17-0.25 grid spacing (its own mode moved 2-3 % between grids) and the delta-function start used here (DREAM seeds are Gaussians).
Time-matched from gamma 40 (single snapshots, mean / median): 2 s 29.2 / 29.3 vs DREAM 28.0 / 27.8; 10 s 24.0 / 23.2 vs 23.2 / 22.0; 20 s 23.0 / 21.6 vs 22.1 / 20.6;
40 s 22.2 / 20.6 vs 21.3 / 19.9: a steady +3 to +4 %, including at 2 s where the early offset is the DREAM seed width (sigma_theta 0.03 loses 21 % faster).
The shapes agree (`p1_tail_vs_dream.png`, left). Both starts (fixed point and gamma 40) converge to the same stationary state: means 21.97 and 21.98 at 100-150 s.

### Convergence (stationary mean / mode; window 50-80 s, gamma-40 start)
| change | mean | median | p90 | mode | note |
|---|---|---|---|---|---|
| baseline N_t = N_b = 8192, dt 1e-3 | 21.96 | 20.57 | 33.8 | 17.9 | seed 0 |
| N = 2048 | 21.85 | 20.29 | 33.9 | 17.9 | |
| N = 32768 | 21.92 | 20.49 | 33.9 | 17.65 | |
| N_b = 32768, N_t = 8192 | 21.95 | 20.56 | 33.8 | 17.9 | trajectories correlated with baseline (same key), not independent |
| dt = 4e-3 | 21.98 | 20.55 | 33.8 | 17.15 | |
| dt = 2.5e-4 | 21.96 | 20.47 | 33.9 | 17.65 | |
| lognormal tail weights, sigma = 1 (weighted estimators) | 21.97 | 20.50 | 33.9 | 17.65 | dynamics identical to baseline bit for bit: tail weights (smaller than every background weight) never enter the update |
| seed 1 (new bulk, ions, tail, keys) | 21.78 | 20.33 | 33.6 | 17.15 | |
| seed 2 | 21.95 | 20.53 | 33.8 | 17.90 | |
The mean is converged to 0.5 % in N, dt and seed (spread 21.78-21.98); the mode, a noisy estimator of a broad peak, scatters 17.15-17.9 (+/-2 %) between
runs of 1.3e5 samples each and 17.65 pooled. Weighted-seed note: with a seed of fraction 1e-5 the weight ratio only needs to stay below 1 for the tail to
be updated at full rate; weights change the estimators, not the trajectories. In the full three-species PIC with a live bulk the unequal-weight handling
is `restore=True` (L2), whose cross-species validation is not repeated here.

## 3. Diagnosis: pitch-angle diffusion (confirmed), with one correction to the suggested test
The suggested test, "switch off collisional scattering in the tail and keep drag only, expect the ODE value", does not hold as stated. The ODE's
-(Z+1) xi gamma/(tau_c eta^3) term is the Ito drift of the same pitch diffusion. Removing the scattering removes that drift too and is not the ODE.
Done anyway (`mode=drag`: post-collision momentum rotated back to its pre-collision direction, magnitude kept): the pitch is then driven to zero by E,
synchrotron loss vanishes and the tail runs away: mean gamma 133 -> 191 at 50-80 s and still rising. So pitch scattering is what makes the
synchrotron-limited state exist at all in this regime.

The clean test is a noise-strength scan (`scripts/sde_scan.py`, 5e4 particles, Euler-Maruyama dt 1e-3): the ODE of L2 plus pitch noise
sqrt(kappa nu (1-xi^2) dt) with the drift left unchanged. kappa = 0 is exactly the L2 ODE; kappa = 1 is the Ito-consistent spherical diffusion.
| kappa | mode gamma | mean | p90 | pitch at mode (rad) |
|---|---|---|---|---|
| 0 | 13.16 (ODE 13.13) | 13.13 | 13.13 | 0.1361 |
| 0.25 | 14.16 | 15.21 | 19.7 | 0.126 |
| 0.5 | 15.41 | 17.37 | 24.5 | 0.116 |
| 1 | 17.40 | 21.71 | 33.5 | 0.102 |
| production kernel | 17.65 | 21.93 | 33.8 | 0.101 |
The kappa = 1 Langevin reduction reproduces the production kernel (mode 17.40 vs 17.65, mean 21.71 vs 21.93, p90 33.5 vs 33.8, pitch 0.102 vs 0.101),
and the observables move monotonically from the ODE to the kinetic bump as the noise is switched on (`p1_noise_scan.png`). The DREAM bump is therefore
pitch-angle diffusion at fixed mean drift; no energy diffusion, field-particle term or large-angle process is needed. The pitch-scattering strength
also fixes the answer (second kernel check, `mode=noion`, electron-electron collisions only, so nu_D halves): mode 34.6, mean 40, pitch 0.051. The state
is highly sensitive to Z_eff.

Correction to a statement in KEY_FACTS section 3: the argument that pitch spread alone should lower gamma (Aleynikov-Breizman form, 10.4) is a balance
of the pitch-averaged force and predicts the wrong sign here. The measured state has gamma raised, because the distribution is broad in gamma
(tau_rad/tau_c = 0.58, outside their tau_rad-bar > 5 range) and the particles that diffuse to small pitch lose almost nothing and accelerate. Not
derived analytically here; observed in the kernel, the SDE and DREAM.

## 4. Draft corrections for PR #77
Synchrotron-limit wording. Replace "A synchrotron-limited fixed point exists ... No runaway above ~6.2 MeV at 2 T, E = 5 mV/m" (NOTES) and the PR
"Baseline result" bullet by:

> The zero-noise test-particle ODE has a synchrotron-limited fixed point at gamma = 13.13 (6.2 MeV, pitch 0.136 rad). It is not a bound on the kinetic
> tail. With collisional pitch-angle diffusion (Z+1 = 2) the stationary distribution at B = 2 T, E = 5 mV/m, n lnL = 1.5e19 peaks at gamma = 17.7
> (pitch 0.10 rad), with mean gamma 21.9 and 90th percentile 33.8 (10.7 MeV mean kinetic energy, 16.7 MeV at the 90th percentile). The peak and the shape agree with the
> DREAM kinetic solution (mode 16.8-17.1) to 4 %, mean to 4-5 % after matching its momentum cutoff. The ODE is reached only as the diffusion is switched
> off. The 20 MeV seed relaxes toward the ODE for the first ~2 s (it is within 3 % of DREAM at 1-2.5 s and within 0.3-2.7 % of the ODE over the 5 ms windows)
> and then toward the kinetic bump, taking tens of seconds. Pitch scattering does not decay it away: with electron-electron scattering only, the mode is gamma ~ 35.

Avalanche estimate. Replace "Avalanche (Rosenbluth-Putvinski, no sync): ~0.03 /s, so <= 3 % per second, <= ~35 % over 10 s" by:

> Avalanche growth at E/Ec = 6.54 is 0.05-0.07 /s (DREAM kinetic 0.053, CODE 0.073, Rosenbluth-Putvinski eq. 18 with its final bracket 0.073; the
> earlier 0.03 /s was about 2x low). That is 5-7 % per second and a factor 1.7-2.0 over 10 s; the kinetic threshold is E/Ec ~ 1.4. It is not in the
> JAX-in-Cell kernels (no large-angle collision source), so it is not part of the comparison over the 100 us-5 ms kinetic windows (growth < 0.04 % at 5 ms).
Also change the "Limits" bullet "no CODE/DREAM run (not installable here)": DREAM and CODE were run (results/codes), and the PIC tail was checked against
DREAM at long times (this note); the live-bulk 5 ms windows are unchanged.

## 5. Avalanche: JAX-in-Cell cannot produce it; decision on a knock-on source
Collisions in `_collisions.py` are cumulative small-angle Nanbu rotations of the COM momenta of a pair; elastic and energy-conserving, with no creation of
a free electron above a threshold. There is no Moller (large-angle) event and no species creation, so the avalanche growth rate cannot be measured with this
code. The gate "+/-10 % of the DREAM-CODE band" (E/Ec = 6.54, 20, 50: 0.053-0.073, 0.196-0.238, 0.523-0.65 /s, widened to 0.048-0.080, 0.176-0.262, 0.471-0.715)
is therefore reported as not reproducible, not as failed or passed. I did not relax it.

Decision: do not implement a knock-on source for the thesis. Reasons: (i) the RF thesis is pitch scattering of existing runaways; the avalanche is a
multiplicative number source acting on a population near p_c ~ 0.4, and the RF rates P2/P3 measure per-runaway scattering, which does not depend on it;
(ii) over every window P2/P3 will use (ms to <= 1 s) avalanche growth is <= 5-7 % (0.05-0.07 /s x 1 s), small against the quasilinear-versus-trapped
difference being sought (orders of magnitude, Kang et al.); (iii) a Moller source here is not small: it needs particle creation in fixed-size JAX
arrays (the zero-weight slot convention used for wall-collected particles could host it), the RP/Embreus kinematics (secondary pitch fixed by energy-
momentum conservation), a population-control step to keep the particle count fixed under growth, and tests plus docs to meet the repo's 100 % branch-coverage
and docs gates. A driver-side Monte Carlo would not test the library. Revisit only if the paper claims suppression of avalanche multiplication by RF
scattering (RF-broadened pitch changes the avalanche rate): then the right instrument is the planned P4a DREAM fork with an RF diffusion term,
which already has the kinetic Moller operator, with JAX-in-Cell supplying only the scattering rate.

## 6. Notes for the next work packages
- A held bulk (frozen or thermostatted) is the right baseline for any run longer than ~1 s: a live bulk heats ~1.4x per second.
- Pitch scattering strength (Z+1) controls the tail strongly, so the Z_eff of the DIII-D case must be fixed before comparing RF thresholds.
- dt for tail-only runs is set by the tail dynamics, 1e-3 s here, but any wave term needs its own resolution (bounce period).
- Reproduce: `PYTHONPATH=~/local/jic-work/P1 python scripts/tail_steady.py out.npz nt=8192 dt=1e-3 T=80 snap=2 start=g40` (set XLA_FLAGS to one thread);
  `scripts/sde_scan.py`; `scripts/summarize.py` regenerates `summary.json`, `summary_sde.json` and the PNGs. Figures: `p1_tail_vs_dream.png`, `p1_noise_scan.png`.
