# Radiation reaction (L0 survey)

Status key: [V] retrieved this session (Smilei docs), [H] reference, [K] memory, verify.

## Models
| Model | Equation | Valid | Differentiable | Cost |
|---|---|---|---|---|
| Full LL (Landau-Lifshitz 1962) | Lorentz force + three RR terms: field-derivative term `~ tau_e gamma (dF/dt)`, a `tau_e` field-squared term, and the `gamma^3 (v.F)^2 v` drag term | classical, chi<~1e-2 | yes, needs dF/dt | field derivative needed |
| Reduced LL (drop dF/dt and the small field-squared term) | invariant drag `F = -P_alpha chi^2 u/(u^2 c)` with `g=1` (see next row) | equals full LL up to O(1/gamma) corrections for gamma>>1 (Vranic 2016, 10.1016/j.cpc.2016.04.002) | yes | cheap |
| High-energy drag | `-P_alpha chi^2 u/(u^2 c)` [V Smilei: `P_alpha = 2 alpha^2 m c^2/(3 tau_e)`, `tau_e=r_e/c`] | gamma>>1 | yes | cheapest |
| cLL (Smilei) | multiply by `g(chi) ~ [1+4.8(1+chi)ln(1+1.7chi)+2.44chi^2]^(-2/3)` [V] | chi<~0.1 | yes (smooth) | cheap; mean recoil only |
| Niel FP (PRE 97 043209) | `dp = F_L dt + F_rad g dt + m c^2 sqrt(R) dW u/(u^2 c)`, `R=(2/3)(alpha^2/tau_e) gamma h(chi)` [V] | chi<~1 (weak quantum) | pathwise through dW | stochastic, small dt |
| Monte-Carlo photon emission | optical depth `tau_f=-log xi`, rate `(2/3)(alpha^2/tau_e) K(chi)` [V] | any chi, LCFA | no (discrete) | tables |
`h(chi)` fits [V]: Ridgers `chi^3 (165/(48 sqrt3)) (1+(1+4.528chi)log(1+12.29chi)+4.632chi^2)^(-7/6)`; order-5 log polynomial 1e-3<chi<10. `g` exact integral form and Smilei coefficient table in Smilei docs.
Pushers: Vranic/Tamburini (Boris + RR substep, Tamburini 2010 NJP 12 123005 [K]); Elkina 2014 notes explicit LL integrators violate conservation over long runs and symmetric implicit variants do better [from search snippet]; Ripperda 2018 compares integrators. Pétri 2022 (arXiv 2207.00624) exact reduced-LL step for constant fields [snippet].

## Exact test solutions
1. Uniform B, reduced LL (mc=1 units, `u=p/mc`, `gamma=sqrt(1+u^2)`): radiated power `P = u_perp^2/tau_B` (units mc^2), drag `F = -(u_perp^2 gamma/(tau_B |u|)) u_hat`, rotation does not change `u_perp`, `u_par`. Hence pitch angle is constant under reduced LL and `d ln|u|/dt = -gamma sin^2(theta)/tau_B`. For theta=90 deg: `dgamma/dt = -(gamma^2-1)/tau_B`, `gamma(t) = coth(t/tau_B + C)`. `1/tau_B = e^4 B^2/(6 pi eps0 m^3 c^3)`. Computed here: B=2 T -> tau_B = 1.290 s; a 20 MeV electron (gamma=40.14) reaches gamma=10 at t=0.0973 s and gamma=2 at t=0.676 s. (Derived in this survey, checked for self-consistency at 90 deg; the finite-pitch ODE has no closed form beyond quadrature; integrate numerically as oracle.)
2. Plane wave: Di Piazza 2008 exact LL solution (10.1007/s11005-008-0228-9). Full-LL vs reduced-LL comparison through a pulse: published agreement to O(1/gamma) corrections (Vranic 2016).
3. Constant crossed fields: Yaremko (10.1063/1.4820131).
4. Magnetized plasma cooling -> ring/inversion: Bilbao & Silva PRL 130 165101 (2023) (10.1103/PhysRevLett.130.165101) (not re-read; geometry check needed).
5. Smilei benchmarks [V] as cross-code numbers: counter-propagating plane wave a0=270: ~80% of initial kinetic energy radiated, LL overestimates; synchrotron 2D chi=0.1 gamma=450: energy error 2-3e-9; chi=0.1 gamma=1e3 constant B: cLL, Niel, MC spectra agree.

## chi monitor (feeds L5)
`chi = (gamma/E_s) sqrt((E+vxB)^2 - (v.E)^2/c^2)`, `E_s = m^2 c^3/(hbar e) = 1.3e18 V/m` [V]. Thresholds [V]: LL classical ~1e-3 (overestimates for chi>1e-2); cLL ~1e-2; Niel/MC ~1e-1; MC mandatory chi>~1. For 20 MeV in 2 T: chi = gamma B/B_crit = 40*2/4.41e9 ~ 1.8e-8, deep classical. Classical LL is exact enough; quantum coefficients are irrelevant for the tokamak case.
Beyond LCFA: Di Piazza 2018 (10.1103/PhysRevA.98.012134), 2019 (10.1103/PhysRevA.99.022125).

## Codes
Smilei `src/Radiation/RadiationLandauLifshitz.cpp`, `RadiationCorrLandauLifshitz.cpp`, `RadiationNiel.cpp`, `RadiationMonteCarlo.cpp`, `RadiationTables.cpp` (tables of h, K, xi) [V file list]. EPOCH QED and PICSAR QED not inspected.

## Pitfalls
Reduced LL drag in the PIC pusher must act on the same u used for deposition; energy removed from particle must be accounted as radiated energy diagnostic. In explicit Boris, order (rotation then drag) matters at O(dt); test against coth solution for convergence rate.
