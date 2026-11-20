# RF source power accounting in PIC (L0 survey)

Status: mostly [K] standard EM theory; no code survey of antenna models beyond what is listed. Mark items verify.

## Closed budget (Poynting theorem, 1D3V)
`d/dt integral (eps0 E^2/2 + B^2/(2 mu0)) dx + [S_x]_boundaries = - integral J.E dx`, `S = E x B/mu0`.
With a driven external current `J_ext` (soft source) the plasma sees `E` from Ampere with `J_p + J_ext`:
- Power delivered by the source: `P_src = - integral J_ext . E dx` (sign convention: positive = source does work on field).
- Power absorbed by particles: `P_p = integral J_p . E dx = d/dt K_total + P_rad_loss(if RR active)` (RR drag removes kinetic energy: `dK/dt = J_p.E - P_rad` -> track `P_rad` separately).
- Field energy change + boundary flux = `P_src - P_p`. Residual of this identity is the primary diagnostic; target ~O(dt^2) for the explicit leapfrog, machine roundoff for an energy-conserving implicit scheme (JIC has both pushers).
Time-averaged for a monochromatic drive: `<P_src> = (1/2) Re integral J_ext* . E dx`; antenna radiation resistance `R = 2<P_src>/|I|^2`.

## Antenna models for 1D PIC
| Model | Description | Pros | Cons |
|---|---|---|---|
| Soft current sheet | `J_ext(x,t) = J0 g(x) sin(omega t) w(t)` added to Ampere | simple, power known exactly via `-J_ext.E` | reflects waves unless boundaries absorbing |
| Hard/boundary field | prescribed `E_y(x_b,t)` | simple | power = Poynting flux at that port, source impedance undefined |
| TF/SF (total/scattered) | inject plane wave with correct Poynting flux `E0^2/(2 Z0)` | clean one-way injection | needs vacuum buffer region |
| Port with matched absorber | characteristic (Silver-Mueller / Mur) BC at x ends | power = in minus out flux | absorbers reflect at low-k whistler (long wavelength) |
For a slow whistler/helicon the dispersion is dispersive and the plasma loading is part of the budget; a constant-amplitude source gives plasma-dependent power. A **power-controlled** drive (feedback on `J0` to hold `<P_src>` constant) is what an optimization over controlled parameters should hold fixed: otherwise "minimum energy" claims compare different supplied powers. Objective candidates: minimize supplied energy `integral P_src dt` to reduce RE energy below a threshold; report `E_src`, absorbed `integral P_p`, radiated, reflected/transmitted fluxes.

## Test targets
1. Vacuum antenna: `<P_src>` equals outgoing Poynting flux summed over both sides to <1e-3 for absorbing BC (dx<lambda/20).
2. Uniform cold plasma: whistler absorption zero without collisions, `P_src` then limited by reflection; with resonant electrons (Landau/cyclotron) `P_p` > 0 matching quasilinear `dK/dt`.
3. Closure residual vs dt: slope 2 (leapfrog) in a full simulation with source and RR on.

## Codes
Smilei (`LaserProfile`, antenna `Antenna` in `src/Params`/`Field`) and EPOCH laser/`bc_x` injection: not read this session. VPIC/Smilei antenna docs describe prescribed current sources; no open code found that reports a Poynting-closure budget by default (not searched exhaustively).
