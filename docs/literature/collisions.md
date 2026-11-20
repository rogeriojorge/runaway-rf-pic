# Relativistic binary Coulomb collisions (L0 survey, 2026-11-09)

Status key: [V] retrieved and read this session; [H] taken from reference `research/COLLISIONS.md` (not re-derived here); [K] from memory, verify before use.

## Models
| Model | Core idea | Conservation | Weights | Note |
|---|---|---|---|---|
| Takizuka-Abe 1977 (doi 10.1016/0021-9991(77)90099-7) | random pairing, Gaussian small-angle delta | exact (equal w), nonrel. | equal only | lineage; Gaussian tan(chi/2) breaks for s>~1 |
| Nanbu 1997 (10.1103/PhysRevE.55.4642), Bobylev-Nanbu 2000 | cumulative angle, `E[cos chi]=exp(-s)` | exact per pair (equal w) | equal only | valid for any s in the angle law; physics still needs small dt |
| Sentoku-Kemp 2008 (10.1016/j.jcp.2008.03.043) | Nanbu + relativistic COM + weights | exact (equal w); weighted only on average | stochastic accept by w ratio | weighted rate was wrong (fixed by Higginson) |
| Perez 2012 (10.1063/1.4742167) | relativistic COM Nanbu, Coulomb-log from b_min=max(b0/2, hbar/2p*) | exact per pair | equal | Eq. 22 has a squared-parenthesis typo [V: Smilei docs] |
| Higginson-Holod-Link 2020 (10.1016/j.jcp.2020.109450) | unequal weights: scatter both with probability w_other/w_max | exact on average only | arbitrary | right rates; energy random-walks per step |
| Angus et al. 2025 (JCP 531, 113927; arXiv 2407.19151) [V abstract] | weighted + post-scatter velocity adjustment restoring exact momentum and energy per cycle | exact per cycle | arbitrary | abstract says relativistic is covered; full text not read; cheap to retrofit |
| Sarkimaki-Hirvijoki-Terava 2018 (10.1016/j.cpc.2017.09.025) | Beliaev-Budker test particle SDE in Maxwell-Juttner bath | none (bath absorbs) | n/a | fixed-background runaway path |
| Vay et al. 2025 (PRE 111, 025306) | energy-preserving PIC + MCC coupling | by construction (nonrel.) | | warns composition with explicit PIC can heat |
| Braams-Karney 1987 (10.1103/PhysRevLett.59.1817) | continuum relativistic FP operator | exact | n/a | independent oracle for drag/diffusion |

Search (WebSearch, 2024-2026) found no newer energy-conserving pairwise relativistic scheme than Angus 2025. No 2025 paper beyond Vay was found; absence of results is not proof.

## Core equations (all [H], kinematics checked vs WarpX source)
- `r = g1 g2 - eta1.eta2`, `M = sqrt(m1^2+m2^2+2 m1 m2 r)`, COM energies `e1* = m1(m1+m2 r)/M`.
- Strength `s12 = n_eff dt sigma_eff v_M* g1* g2*/(g1 g2)`, `sigma_eff = pi b0^2 lnL`, `b0 = |q1q2|/(2 pi eps0 mu* v_M* v_inv)`. Nonrel. limit `q1^2 q2^2 n lnL dt/(4 pi eps0^2 mu^2 v^3)`.
- Angle law (Nanbu): density `~ exp(A mu)`, `coth A - 1/A = exp(-s)`, `mu = (1/A) log(e^-A + U(e^A - e^-A))`.
- Smilei fast fit [V]: `sin^2(chi/2) = alpha U/sqrt(1-U+alpha^2 U)` for s<4 (else `1-U`), `alpha = 0.37 s - 0.005 s^2 - 0.0064 s^3`; recommends s<1.
- Self-collision pairing: random perfect matching, `n_eff = (N-1) w/V` (H); odd N leaves one particle out.

## Codes (file/function)
- Smilei: `src/Collisions/Collisions.cpp` (pairing, `s`, Coulomb log, Nanbu-fit angle), `BinaryProcesses.cpp` (pair loop). Docs: smileipic.github.io/Smilei/Understand/collisions.html [V].
- WarpX: `Source/Particles/Collision/BinaryCollision/Coulomb/UpdateMomentumPerezElastic.H`, `ElasticCollisionPerez.H`, `PairWiseCoulombCollisionFunc.H` (blast-warpx repo, listed via gh API).
- EPOCH, OSIRIS, PIConGPU: not read this session; PIConGPU/OSIRIS binary-collision docs not verified.
- DREAM (grid, not MC): `py/DREAM/Settings/CollisionHandler.py`. Options [V]: collfreq_mode FULL (all-electron kinetic, Maxwell-Juttner consistent) / SUPERTHERMAL (non-conservative, no Dreicer) / ULTRA_RELATIVISTIC (v=c); collfreq_type PARTIALLY_SCREENED / COMPLETELY / NON_SCREENED; lnlambda ENERGY_DEPENDENT / THERMAL / CONSTANT.

## Validation targets (Smilei suite [V], usable as JIC test list)
1. Beam relaxation of v_par, v_perp vs NRL rates (equal-weight and weight ratios 0.1, 1, 10).
2. Electron-ion temperature equalization vs NRL (nonrel.), isotropization of a bi-Maxwellian vs NRL.
3. Maxwellianization of a rectangular distribution (entropy/H-theorem check; relativistic end state Maxwell-Juttner `f ~ exp(-gamma/theta)`).
4. Exact conservation: momentum and total relativistic energy per pair to roundoff (equal w) -> test_oracle independent of implementation; unequal w: statistical conservation to O(1/sqrt(Npairs)) (Higginson) or exact (Angus).
5. Relativistic slowing down of a test beam in a Maxwell-Juttner bath vs Beliaev-Budker / Braams-Karney drag (no tabulated numbers retrieved; derive numerically from the Braams-Karney potentials as oracle).
6. Runaway-relevant: collisional friction of a v~c electron on cold background equals `eE_c`-scale drag, `E_c = n e^3 lnL/(4 pi eps0^2 m c^2)`; at n=1e18 m^-3: 7.6e-4 V/m (lnL=15), computed here.

## Pitfalls
- Never substitute v_M* (may exceed c) for v_inv. Guard comoving pairs before evaluating singular rates; gradients through the angle root need a custom rule.
- Collisions in 1D3V: density uses `sum(w_code)/dx` (3D density); no transverse-space effect.
- Smilei lnL uses the Debye/ b_min prescription per block; fix lnL for the validation runs.
