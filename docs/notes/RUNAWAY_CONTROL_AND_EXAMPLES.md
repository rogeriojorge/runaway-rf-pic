> Original research specification, preserved for equations, sources and remaining work. Some API proposals and implementation-status statements are historical. Read `../START_HERE.md` first; the bundled branch and its documentation define the implemented API. Historical `evidence/` and `probes/` paths refer to the earlier full research archive, not to new production validation. Current reproduction is in `../VALIDATION.md`.

**2026-11-05 continuation:** read `../CONTINUATION_RESULTS.md` before using this
historical plan. The laser-beam and cold-plasma RF source-cost examples now
exist. The older blanket positive-cycle-average power rule is superseded:
the actuator recovery/storage model must define the cost. The tested ideal
port dumps returned work and pays positive instantaneous source work; a
physical RF port needs validated storage/loss/recovery accounting. Arbitrary
averaging bins changed rankings. Current API and branch instructions take
precedence over historical filenames and proposed interfaces.

# Local RF control of runaway electrons: physical model, examples, and gates

Prepared 4 November 2026. This note is a research and implementation specification, not a report of completed simulations. Numerical parameter sets and acceptance tolerances below are proposed starting points. The analytical reductions and timescale estimates are derived here; literature-supported claims are cited separately.

## 1. Research question and scope

Find the smallest **local RF energy input over a fixed control time**, equivalently the smallest mean local RF power over that time, that reduces a specified energetic-electron population and keeps it below a prescribed level during a subsequent observation interval. The mechanism is resonant redistribution of momentum followed by synchrotron cooling, competing with electric-field acceleration, collisions, and any significant electron sources.

The first calculation is a homogeneous or local 1D3V calculation with a fixed background magnetic field and a supplied set of RF modes. It does not calculate a magnetic equilibrium, antenna coupling, ray propagation, accessibility, global current evolution, or wall loading. Those are separate problems. Use power density in W/m³, energy density in J/m³, or a stated power/energy per unit area for a one-dimensional domain. Do not convert a local field amplitude into an antenna power in MW without a separately supplied and validated coupling model.

A finite-time reduction in energetic particles is the initial claim. “Runaway suppression” requires additional evidence about replenishment, reacceleration and the observation interval. If the accelerating field remains supercritical after the RF is switched off, indefinite suppression does not follow from a favorable terminal distribution.

### Existing results and the remaining opportunity

Externally launched helicon waves have already produced resonant pitch-angle scattering and limited the high-energy runaway population in low-density DIII-D plasmas [R1]. A 2026 follow-up investigates alignment, propagation direction and coupled power [R2]. Self-excited whistlers, pitch scattering, enhanced radiation and changes to avalanche behavior also have a substantial literature [R3, R4]. AORSA–KORC calculations already combine realistic wave fields and full-orbit runaway tracing [R5].

The proposed advance is a **validated, differentiable, constrained control calculation**: determine which local spectrum and temporal allocation of RF energy achieve a specified lasting energy reduction at the smallest found budget, quantify sensitivity to uncertain plasma and beam parameters, and show that the gain survives independent particle/kinetic verification. The search performed for this reference did not establish that this precise optimization has already been solved; it also does not establish a priority claim. Recheck focused literature before making one.

Useful questions are:

1. Does distributing RF power over several resonances reduce the energy needed to cool a broad runaway distribution compared with the best single-frequency drive?
2. As the electrons cool and leave their initial resonance, is a time-dependent allocation of power more efficient than a stationary spectrum?
3. How much of the apparent improvement survives collisions, radiation-induced pitch narrowing, bulk heating, and uncertainty in density, accelerating field and initial pitch distribution?
4. When does coherent resonant trapping invalidate a quasilinear control model, and can a controller selected with that model still be certified with phase-resolved dynamics?
5. Is the useful minimum set by resonant coverage, RF absorption by the bulk, competition with acceleration, or replenishment of the tail?

The existence of wave-driven scattering or enhanced synchrotron emission is a validation target, not a new result.

## 2. Model hierarchy and claims permitted at each level

| Level | Dynamical model | What it establishes | What it does not establish |
|---|---|---|---|
| A | One electron in prescribed uniform fields with physical Landau–Lifshitz radiation reaction | Correct force, cooling, pitch evolution and derivatives | Wave coupling or population control |
| B | An ensemble in fixed, physically consistent RF fields plus radiation; collisions added separately | Resonance, coherent trajectories, phase averaging and local particle sensitivities | RF input power, self-consistent wave response or long-time suppression |
| C | Local electromagnetic 1D3V PIC with an explicit RF source, physical radiation and validated collisions | Local wave–particle feedback, source work, bulk absorption and short-time kinetic response | A full device, antenna efficiency or millisecond cooling at an affordable step count by default |
| D | Conservative local momentum-space kinetic evolution, using shared collision/radiation kernels and a validated RF response model | Physical millisecond-to-second control horizons and optimization with sources/reservoirs | Phase-resolved collective dynamics outside the response model's validity range |
| E | Level D controls tested in independent kinetic calculations and selected Level B/C windows | Evidence that optimization survives model and discretization changes | An uninterrupted full-PIC simulation of the entire physical horizon unless one was actually run |

Implement A–C as the JAX-in-Cell radiation-reaction extension. Level D may be a small optional local kinetic driver sharing the physical coefficients. Do not silently call it PIC or advertise its result as a full PIC optimum. Reuse a suitable existing differentiable kinetic implementation if one is already in the repository; avoid creating a separate framework merely for the example.

A no-radiation RF-scattering example is useful, but it cannot demonstrate that radiation reaction has achieved net cooling. Likewise, multiplying the radiation coefficient to make cooling visible in a short PIC run is a numerical stress test. Such a run must be labeled as scaled physics and must not be used to establish the fusion result.

## 3. The particle and kinetic equations

For species s, the local phase-space equation is

\[
\partial_t f_s+v_x\partial_x f_s+
\nabla_{\boldsymbol p}\!\cdot\!\left[
\left(q_s(\boldsymbol E+\boldsymbol v\times\boldsymbol B)
+\boldsymbol F_{\rm LL,s}\right)f_s\right]
=C_s[f]+S_s-L_s.
\tag{1}
\]

The radiation term is a divergence in momentum space. Radiation contracts phase-space volume; replacing it by only \(\boldsymbol F_{\rm LL}\cdot\nabla_p f\) drops that compression. In the particle representation, evolving momenta with constant physical particle weights naturally transports the distribution with this compression.

The fields contain a fixed \(\boldsymbol B_0\), self-consistent perturbations, a clearly defined accelerating-field drive, and an RF source. The code must document whether the **total** mean parallel electric field is prescribed or evolves. Adding an imposed constant electric field to a freely evolving mean Ampère mode can screen or alter the intended drive. Either prescribe the total mean field with a corresponding external source, or evolve it with a specified current source. Record the source work in both cases. A prescribed local field is not a self-consistent tokamak circuit model.

### A local gyrotropic reduction for the slow control calculation

**Momentum adapter.** The production API stores \(\boldsymbol u_{\rm code}=\boldsymbol p/m_e=\gamma\boldsymbol v\) in m/s. This note uses the dimensionless \(\boldsymbol\eta=\boldsymbol p/(m_ec)=\boldsymbol u_{\rm code}/c\). Pass `u=c*eta_vector` to the production particle initializer and convert rates as `du_dt=c*deta_dt`. Never pass the dimensionless \(\eta\) directly as the production `u`. The distribution \(f_\eta=(m_ec)^3 f_p\) includes the Jacobian relative to the physical-momentum distribution in equation (1).

Let \(\eta=p/(m_ec)\), \(\xi=p_\parallel/p\), \(\gamma=\sqrt{1+\eta^2}\), and choose the positive parallel direction along the initial electron beam. Define \(E_{\rm acc}\ge0\) so that the physical electric field is \(-E_{\rm acc}\boldsymbol b\). Then

\[
 a_E=\frac{eE_{\rm acc}}{m_ec},\qquad
 a_B=\tau_e\left(\frac{eB_0}{m_e}\right)^2,
 \qquad \tau_e=\frac{e^2}{6\pi\epsilon_0m_ec^3}.
\tag{2}
\]

For uniform magnetic field, the electric acceleration and classical synchrotron drift are

\[
\dot \eta=a_E\xi-a_B\gamma \eta(1-\xi^2),\qquad
\dot\xi=\frac{a_E}{\eta}(1-\xi^2)
+\frac{a_B}{\gamma}\xi(1-\xi^2).
\tag{3}
\]

The magnetic radiation terms follow from

\[
 \dot \eta_\parallel=-a_B\frac{\eta_\parallel \eta_\perp^2}{\gamma},\qquad
 \dot{\boldsymbol \eta}_\perp=-a_B
 \frac{1+\eta_\perp^2}{\gamma}\boldsymbol \eta_\perp,
 \qquad \dot\gamma=-a_B\eta_\perp^2.
\tag{4}
\]

Equation (3) is a **uniform-field slow model**, not a replacement for the full Landau–Lifshitz force in a general RF field. It retains the magnetic synchrotron contribution; the accelerating-field radiation corrections are negligible only after checking the field ordering, which is extremely strong for the proposed values of E_acc/(c B_0). Include the RF contribution through either resolved coherent dynamics or a verified response operator, and check whether RF-associated radiation changes the chosen error budget. Radiation-induced pitch narrowing is retained. Near \(\eta=0\), use the proper regularity of the kinetic equation or a Cartesian treatment; do not make the coordinate singularity disappear by an arbitrary clipping rule.

For \(F(\eta,\xi)=2\pi \eta^2f_\eta(\eta,\xi)\), normalized so that \(\int F\,d\eta\,d\xi=n_e\), use a conservative equation

\[
 \partial_t F=-\partial_\eta(\dot \eta F)-\partial_\xi(\dot\xi F)
 +C_F+Q_{\rm RF,F}+S_F-L_F.
\tag{5}
\]

Coefficients taken from a Cartesian diffusion model must be transformed with the correct drift and Jacobian. A positive diffusion tensor alone is not sufficient to make an incorrectly transformed operator conservative.

## 4. Physics needed beyond radiation reaction

### 4.1 Relativistic collisions

At minimum, the long-time model needs electron–electron slowing down and energy diffusion, and electron–ion pitch-angle scattering, with a declared background distribution and ion composition. Use coefficients valid across the modeled momentum domain. A relativistic test-particle operator is appropriate only when the tail is dilute and the background is treated consistently. If the full thermal population is resolved, require its correct Maxwell–Jüttner equilibrium and the required field-particle/conservation terms or a separately validated conservative collision scheme. DREAM's documented full collision model supports a Maxwell–Jüttner equilibrium; its superthermal and ultrarelativistic options have narrower intended ranges [R6]. The separate collision work lane specifies the particle algorithm and tests.

Do not confuse numerical PIC scattering, resolved collective waves, and Coulomb collisions. They are different mechanisms. Vary grid spacing, shape order and particle number to establish that numerical heating/scattering is below the physical response being optimized.

### 4.2 Bulk heating and the background reservoir

For a dilute explicitly followed tail, specify whether the bulk is:

- a fixed thermal bath, with an explicit energy/momentum reservoir ledger; or
- an evolving bulk with particle number and temperature updated by collisional transfer, RF absorption and imposed-field work; or
- fully kinetic in the PIC calculation.

A fixed bath is a useful controlled benchmark. It is not evidence that the corresponding physical plasma can remove the deposited heat. Compare control-induced bulk heating against an uncontrolled calculation with the same accelerating field, because the imposed field can already heat the bulk strongly. If bulk temperature changes appreciably, update collision coefficients and the local RF response; a response table at the initial temperature alone is insufficient.

Use total electron density consistently: the thermal and runaway components must sum to it. A seeded electron tail is not added on top of an unchanged neutralizing electron population. For a current-neutralized short PIC benchmark, the compensating bulk drift is calculated from charge/current balance. For a current-carrying benchmark, the mean-field/current-drive closure must be stated instead.

### 4.3 Large-angle collisions and avalanche

Small-angle Fokker–Planck or binary diffusion collisions do not automatically describe knock-on runaway multiplication. If avalanche is significant, include a relativistic large-angle electron collision operator or a verified source–sink approximation. A source that only adds energetic electrons can violate particle number, momentum and energy conservation and miss reverse-avalanche slowing near threshold. Embréus, Stahl and Fülöp derive a conservative relativistic large-angle operator and discuss the near-threshold importance of the operator choice [R7].

Partition small- and large-angle collisions consistently so their overlapping momentum-transfer range is not counted twice. If new tail particles are introduced, remove the corresponding thermal electrons and include the primary electron's recoil. In a fixed-size particle representation, use an explicit weighted representation or a controlled buffer/resampling strategy; do not change particle number by an untracked numerical side effect.

For the first seeded-relaxation study, avalanche may be omitted only after a quantitative gate, for example a validated estimate \(\int_0^{T_c+T_h}\Gamma_{\rm av}(t)\,dt<0.02\), and a source-on comparison confirming negligible change to the optimization result. This 2% criterion is a proposed accuracy budget, not a universal physical threshold. Otherwise avalanche is part of the production control model.

### 4.4 Primary sources

For the warm, dilute baseline, evaluate whether Dreicer production is negligible over the control and hold times. If a fully kinetic bulk resolves it, do not also add a fluid Dreicer source. If a reduced tail model uses a source at a matching boundary, specify the boundary flux and reservoir depletion. Hot-tail production is needed for a cooling thermal background characteristic of disruption formation; it is not required merely because the followed electrons are relativistic. Reactor tritium-decay and Compton seeds are future scenario inputs when their integrated contribution matters. The first example should not acquire a collection of unvalidated sources.

The collision/radiation problem and primary/secondary generation have established kinetic benchmarks in CODE and DREAM [R8, R9]. Use matched local parameters and conventions for these comparisons, not an unmatched whole-device discharge. This reference to CODE/DREAM does not assert that they expose the exact proposed RF control operator: compare electric-field, collision, radiation and source terms directly, and use an independently implemented RF response only where a matched operator is actually available.

### 4.5 Ion composition, partial screening, and bremsstrahlung

The first control example uses fully ionized deuterium and \(Z_{\rm eff}=1\). For impurity-rich or partially ionized conditions, a single effective charge is generally insufficient: specify charge states, free/bound electron densities, partial screening and inelastic stopping. Their effect on friction, pitch scattering and avalanche must be included before drawing conclusions about disruption mitigation. The effective critical field changes when partially screened collisions and radiation are included [R10, R11].

Bremsstrahlung is not generated correctly merely by evaluating Landau–Lifshitz on the resolved PIC grid fields. The microscopic ion encounters responsible for that emission are unresolved. Include a verified stopping-power or Boltzmann-emission treatment if its energy loss competes with synchrotron and collisional losses. The initial fully ionized low-density case may neglect it only after comparing its integrated loss with the selected error budget. Do not double count a bremsstrahlung recoil term and photon-emission operator for the same process.

### 4.6 What stays out of the first study

Do not add equilibrium reconstruction, magnetic ripple optimization, antenna design, ray tracing, wall transport, impurity assimilation, or a full disruption circuit. A local fixed \(B_0\) is sufficient to establish the requested control mechanism. These omitted effects limit the claim; they do not justify postponing the local calculation.

## 5. Coherent RF dynamics and quasilinear diffusion

The local resonance is expressed with a verified polarization/sign convention as

\[
 \omega-k_\parallel v_\parallel=s\frac{\Omega_e}{\gamma},
 \qquad \Omega_e=eB_0/m_e.
\tag{6}
\]

For a counterpropagating normal-resonance benchmark with the conventions above, use \(s=1\), \(k_\parallel<0\), and verify the resonant polarization by a direct particle orbit. Do not infer the sign solely from a name such as right-handed polarization.

A supplied coherent wave must obey Faraday's law. A simple constant-amplitude mode has \(\boldsymbol B_1=\boldsymbol k\times\boldsymbol E_1/\omega\). For changing envelopes or chirps, generate fields from a vector potential, including its envelope derivatives, or drive Maxwell's equations through an explicit current. Replacing \(k\) and \(\omega\) at each time in a monochromatic formula is not generally a Maxwell-consistent chirped field. In a periodic domain the spatial wave numbers are fixed integer multiples of \(2\pi/L\); an arbitrary time-dependent wave number is not periodic.

A small fixed bank of supplied local modes is sufficient. A cold homogeneous dielectric mode may serve as an initialization/test input. This is not a proposal for a wave-accessibility or antenna study. A self-consistent PIC calculation must be allowed to reveal deviations from that input caused by finite temperature or the beam.

### Energy transfer is not generally pure pitch diffusion

For an ideal monochromatic parallel traveling wave in uniform \(B_0\), without radiation or collisions, energy and parallel momentum exchange obey the wave-frame invariant

\[
 \mathcal E-\frac{\omega}{k_\parallel}p_\parallel=\mathrm{constant}
\tag{7}
\]

under the assumptions of that prescribed traveling-wave model. This is a useful orbit check. It also shows why an arbitrary operator that changes pitch at exactly fixed energy is not automatically the correct RF model. A physical RF diffusion operator generally has energy, pitch and mixed terms. In particular, a counterpropagating wave can add kinetic energy while shifting momentum into perpendicular motion; the subsequent radiative loss must exceed that input to produce net cooling.

Use **one** representation of a given RF component in each model:

- resolve its coherent electric and magnetic fields in a particle/PIC calculation; or
- represent its phase-averaged response by a conservative RF operator.

Do not apply both the resolved wave and its quasilinear diffusion tensor to the same particles. Different explicitly separated spectral bands may use different representations, with an overlap test.

### How to obtain and validate a reduced RF response

For phase-decorrelated forcing, estimate conditional momentum increments in short resolved runs. For Cartesian momentum coordinates, use

\[
 A_i=\frac{\langle\Delta p_i\rangle}{\Delta t},\qquad
 D_{ij}=\frac{\operatorname{Cov}(\Delta p_i,\Delta p_j)}{2\Delta t},
\tag{8}
\]

with a demonstrated intermediate time window: longer than the correlation time and shorter than the evolution time of the distribution. Verify convergence with that window, particle phases, spectral realization and numerical resolution. Include the mean drift and energy transfer, not only the diffusion covariance. Transform the resulting operator correctly into \((\eta,\xi)\).

The response table must cover the distribution and amplitudes reached by the optimizer. Interpolation outside the validated region is a failed model-validity gate, not a successful optimum. Test amplitude scaling rather than assuming \(D\propto E_{\rm RF}^2\) in a trapping regime.

If coherent trapping, phase locking or narrow-band chirps prevent a Markovian response, retain phase in a reduced resonant-particle model or optimize the phase-resolved ensemble directly for the attainable horizon. Do not repair this failure by increasing a fitted diffusion coefficient.

The 2026 AORSA–KORC study reports energy-dependent diffusive, subdiffusive and superdiffusive pitch transport [R5]. This is a specific reason to test the closure window and retained phase information; it does not imply that every local RF configuration has the same behavior.

## 6. RF cost, particle energy and conservation

For an explicit current \(\boldsymbol J_{\rm drive}\) in Ampère's law, the signed source work is

\[
 P_{\rm drive}(t)=-\int\boldsymbol J_{\rm drive}\cdot\boldsymbol E\,dx.
\tag{9}
\]

The particle and field ledger is

\[
 \frac{d}{dt}(U_{\rm particles}+U_{\rm EM})
 =P_{\rm RF}+P_{\rm acc}+P_{\rm reservoir}
 -P_{\rm photon}-P_{\rm boundary}.
\tag{10}
\]

Internal conservative collisions cancel from the total ledger. A thermal bath, tail matching boundary, particle source, or imposed field does not cancel; record its exchange. For general full Landau–Lifshitz fields, use the radiation/near-field energy accounting defined in the radiation-force specification. Do not equate instantaneous mechanical work with radiated power without the required assumptions. In the uniform-magnetic synchrotron limit of equation (4), the particle loss is the emitted synchrotron power.

Separate these quantities in every result:

1. RF source work and any returned reactive energy;
2. wave energy retained in the domain at the end;
3. work on the runaway component;
4. work on the thermal component and its subsequent heat;
5. work by the accelerating field;
6. synchrotron and other photon losses;
7. fluxes to reservoirs or through boundaries.

For a nonregenerative drive, define a cycle-averaged positive supplied-power cost over a declared carrier averaging window. Counting the positive part of instantaneous oscillatory reactive work at every numerical timestep can create an artificial carrier-dependent cost. For a regenerative ideal source, report signed net work and recovered energy separately. In either case start and finish with the RF field off, or charge the residual field inventory to the objective, so an optimizer cannot obtain a misleading favorable endpoint by borrowing energy from an uncounted field reservoir.

An imposed-field particle calculation by itself does not determine input RF power. \(\int E^2dt\) is a field-exposure measure; it has not become energy merely by being minimized. Vacuum \(I=\epsilon_0cE^2/2\) is not a general helicon-power calibration in a dispersive plasma. Use the explicit-source/Poynting ledger, or a supplied validated calibration from that ledger to the reduced response.

## 7. A concrete constrained optimization

Choose a fixed control time \(T_c\) and RF-off hold time \(T_h\). Let \(\theta\) parameterize a small bank of mode amplitudes and their time envelopes, initially with fixed carrier frequencies. Start with approximately 6–20 control coefficients; add temporal/spectral degrees of freedom only after the gradients and model validity are established. Require finite bandwidth, smooth ramps, an amplitude cap, a peak cycle-averaged source-power cap, and a bulk-heating limit.

A suitable problem is

\[
 \min_\theta\ \overline P_{\rm RF}(\theta)
 =\frac1{T_c}\int_0^{T_c}P_{\rm RF,supplied}(t;\theta)\,dt,
\tag{11}
\]

subject to a target high-energy population throughout the hold interval, acceptable residual tail energy, and acceptable extra bulk heating relative to the uncontrolled baseline. Use a smooth high-energy indicator for optimization, for example

\[
 N_h(t)=\int \sigma\!\left(\frac{K-K_c}{\Delta K}\right)
 F(\eta,\xi,t)\,d\eta\,d\xi,
\tag{12}
\]

then verify the result with the unsmoothed threshold and with several smoothing widths. The final report also gives total energetic-electron energy, current, the full energy/pitch distribution, and thermalization flux. A smaller energy does not automatically mean a smaller runaway current.

A proposed first target is \(N_{K>10\,\mathrm{MeV}}\le0.1N_{r,0}\) throughout a 0.1 s hold following 0.1 s of control, starting from the 20 MeV seed below. This is a feasibility target, not a predicted achieved result. If it is infeasible under the field and heating caps, report the Pareto curve and limiting mechanism. Do not obtain feasibility by raising the radiation coefficient or deleting difficult particles.

Begin with a feasibility search, then reduce the budget. Compare the optimized result with no RF, the best scanned single frequency, a stationary multi-frequency spectrum, and a simple time schedule at identical budgets. Multistart optimization gives a best-found local budget; it does not prove a global minimum. Use independent random phases and particle samples for final verification.

For robustness, vary at least \(n_e\), \(E_{\rm acc}\), initial mean energy, energy width, pitch width and beam density. Add temperature/ion-composition uncertainty when those quantities affect the chosen model. Optimize an ensemble mean plus a specified risk measure, or impose constraints over a declared scenario set. Report the scenario distribution and held-out cases. “Robust” does not mean optimized at one nominal state.

A later feedback controller may use measured current and radiation moments, but an open-loop robust schedule is the first deliverable. Do not assume that the full distribution is directly observable in a real experiment.

## 8. Proposed physical parameter sets and timescale calculations

These values are reproducible **test proposals**. They are not reconstructions of a discharge or claims of accessible antenna conditions. Calculations use \(m_e=9.1093837139\times10^{-31}\,\mathrm{kg}\), \(e=1.602176634\times10^{-19}\,\mathrm C\), \(c=299792458\,\mathrm{m/s}\), \(\epsilon_0=8.8541878188\times10^{-12}\,\mathrm{F/m}\) and deuteron mass \(3.3435837768\times10^{-27}\,\mathrm{kg}\).

### 8.1 Local resonance benchmark near the DIII-D parameter range

The primary experiment reports 476 MHz waves and a representative low-density parameter range with approximately \(n_e=5\times10^{18}\,\mathrm{m^{-3}}\) and \(B\simeq1.4\,\mathrm T\) [R1]. Use those values as an anchor, but choose \(T_e=1\,\mathrm{keV}\), fully ionized deuterium and a homogeneous **parallel** circular mode as a simplified numerical test. Those simplifications are ours.

For the cold parallel R-mode input, include the ion term:

\[
 n_R^2=1-\frac{\omega_{pe}^2}{\omega(\omega-\Omega_e)}
       -\frac{\omega_{pi}^2}{\omega(\omega+\Omega_i)},\quad
 |k|=n_R\omega/c.
\tag{13}
\]

At 476 MHz this gives \(|k|\simeq47.216\,\mathrm{m^{-1}}\), wavelength 0.133074 m, and counterpropagating normal resonance at \(K\simeq6.8423\,\mathrm{MeV}\) for zero initial pitch. The experimental representative \(k_\parallel\) is not identical to this deliberately simplified parallel-mode value; do not label the calculation an exact experimental reproduction.

Use initially narrow single-particle energy scans around resonance, then a finite-width beam and random gyro phases. Start with radiation and collisions disabled to establish equation (6)/(7); enable them afterward. For a short full-PIC response benchmark, choose the grid from the actual Debye length, skin depth and wavelength, then converge it. For this warm test \(\lambda_D\simeq1.0513\times10^{-4}\,\mathrm m\); a one-wavelength domain with 2048 cells has \(\Delta x\simeq6.50\times10^{-5}\,\mathrm m\). A candidate \(\Delta t=10^{-13}\,\mathrm s\) satisfies the vacuum CFL bound and resolves the background gyro/plasma frequencies. It implies 100,000 steps for 10 ns. Particle count and whether that cost is acceptable are measured before an optimization run.

### 8.2 Physical local control baseline

| Quantity | Proposed value |
|---|---:|
| Background magnetic field | 2 T |
| Total free-electron density | \(10^{18}\,\mathrm{m^{-3}}\) |
| Initial bulk electron/ion temperature | 100 eV / 100 eV |
| Composition | Fully ionized deuterium, \(Z_{\rm eff}=1\) |
| Coulomb logarithm for preliminary estimates only | 15 |
| Initial runaway density fraction | \(10^{-5}\) |
| Initial runaway kinetic energy | Smooth distribution centered at 20 MeV, width approximately 2 MeV |
| Initial pitch angle | Distribution centered near 0.1 rad, with a specified finite width |
| Accelerating field magnitude | 0.005 V/m |
| Control / hold horizon | 0.1 s / 0.1 s |
| Supplied local carrier range for the first scan | 0.4–1.5 GHz |
| Initial field-amplitude search cap | \(E_{\rm RF,rms}\le10^4\,\mathrm{V/m}\), subject also to source-power/heating caps |
| First target | Reduce the population above 10 MeV to at most 10% of the initial seed during the hold |

The RF amplitude cap is an initial numerical search range, not an experimentally justified hardware specification. If no feasible result exists, map the trade-off explicitly. The tail density, thermal density and ion density must be initialized with consistent total neutrality. Use an evolving bulk or declare and account for the thermal-bath benchmark. The accelerating-field energy input is retained during the RF-off hold unless a different scenario is explicitly being tested.

Diluteness in number does not guarantee a negligible thermal-energy perturbation. These proposed values give approximately 32 J/m^3 in the initial 20 MeV tail and 24 J/m^3 in 100 eV bulk electrons, using `n_runaway*K` and `3*n_thermal*T_e/2`. This calculation is our parameter check, not an experimental measurement. The fraction deposited collisionally or by RF therefore needs an evolving bulk or an explicitly accounted external heat reservoir.

With equation (13), the approximate zero-pitch normal resonances for this baseline are:

| Frequency | \(|k|\), m⁻¹ | Resonant kinetic energy, MeV |
|---:|---:|---:|
| 0.400 GHz | 17.7712 | 22.4176 |
| 0.476 GHz | 19.8453 | 19.5991 |
| 0.600 GHz | 23.0726 | 16.3138 |
| 1.000 GHz | 32.7685 | 10.6560 |
| 1.500 GHz | 44.2003 | 7.4257 |

This table motivates testing time-dependent allocation across resonances as energy falls. It is not an optimum, and finite pitch shifts the resonances. For a periodic full-PIC combination use commensurate spatial modes rather than inserting all these approximate continuous values into one periodic box.

### 8.3 Why a physical long-time full-PIC optimization is not the default

For the 2 T, \(10^{18}\,\mathrm{m^{-3}}\) baseline,

\[
 \omega_{pe}\simeq5.64146\times10^{10}\,\mathrm{s^{-1}},\quad
 \Omega_e\simeq3.51764\times10^{11}\,\mathrm{s^{-1}},\quad
 a_B\simeq0.775394\,\mathrm{s^{-1}}.
\tag{14}
\]

The instantaneous synchrotron energy-loss time in uniform B is

\[
 t_K=\frac{K}{P_{\rm syn}}
 =\frac1{a_B(\gamma+1)\sin^2\vartheta}.
\tag{15}
\]

At 20 MeV this is about 3.145 s for \(\vartheta=0.1\), 0.0983 s for \(\vartheta=0.6\), and 0.0609 s for \(\vartheta=0.8\) radians. These are instantaneous times, not exact exponential decay constants; the distribution and pitch evolve. They illustrate why increased perpendicular momentum can change radiation losses substantially.

The classical Connor–Hastie scale for this fully ionized estimate is

\[
 E_c=\frac{n_ee^3\ln\Lambda}{4\pi\epsilon_0^2m_ec^2}
 \simeq7.64865\times10^{-4}\,\mathrm{V/m},
\tag{16}
\]

so the proposed accelerating field is approximately 6.54 times this scale. It is not the effective critical field after radiation, partial screening or wave scattering. The electric energy-gain rate near \(v_\parallel=c\) is about 1.50 MeV/s at 0.005 V/m.

At 100 eV, \(\lambda_D\simeq7.434\times10^{-5}\,\mathrm m\). Even a relatively generous explicit timestep of \(10^{-13}\)–\(10^{-12}\,\mathrm s\) would require \(10^{11}\)–\(10^{12}\) steps for 0.1 s, before counting particles or reverse-mode work. An implicit field integrator alone does not remove the need to resolve the relevant wave/gyro resonance accurately. Measure feasibility rather than assuming a GPU closes this gap.

The intended strategy is therefore short physical PIC response calculations plus a validated slow kinetic control calculation. The final physical claim must use the real \(\tau_e\). A scaled-\(\tau_e\) stress example is supplementary only.

### 8.4 Quantum effects are negligible for these fusion examples

For these uniform-field estimates, \(\chi_e\sim\gamma B_0\sin\vartheta/B_{\rm Q}\), with \(B_{\rm Q}\simeq4.414\times10^9\,\mathrm T\). At 20 MeV and 2 T it is at most approximately \(1.8\times10^{-8}\). Classical radiation reaction is the appropriate starting model. QED sampling does not solve the timescale problem and is not needed to validate the runaway-control result.

## 9. Required examples and what each must demonstrate

The compact filename list in `IMPLEMENTATION_PLAN.md`, section 5, is authoritative. The cases below are modes or clearly organized sections within those examples, not a request for additional files. Use the repository's existing example style: self-contained readable scripts, shared physical kernels, explicit settings near the top, deliberate comments/docstrings, and no duplicated solver implementation. Select a short verification configuration or a physical research configuration with ordinary settings; do not add a new command-line parser, registry or abstraction layer for these cases.

| Canonical example | Required cases and comparison | Required output/claim |
|---|---|---|
| `electron_cooling.py` | Uniform-B LL orbit at several energies/pitches; exact solution/derivatives; parallel-E/B orbit check where useful | Energy, gyrophase, pitch, convergence, correct gradients and field-work ledger |
| `beam_cooling.py` | A finite-width energy/pitch distribution with physical radiation; collision-free references | Distribution evolution, radiation energy and population moments |
| `laser_electron_collision.py` | Plane-wave or prescribed-pulse full/reference versus reduced LL | Outgoing observables, pulse sensitivities and the model-discrepancy limit |
| `radiative_magnetized_plasma.py` | Established collective cooling/inversion benchmark with a declared spatial model | Reproduction of prior physics with resolved fields, rather than a new-discovery claim |
| `relativistic_collisions.py` | Equilibrium, beam slowing down, pitch scattering and energy exchange | Independent rates, conservation or bath ledger, statistical and derivative convergence |
| `runaway_rf.py` | Fixed-mode resonance and RF-only invariant; short source-driven 1D3V PIC; response-model validation; local collision/radiation evolution; relevant source-on and RF-off-hold cases | RF energy–momentum transfer, source/bulk/tail work, numerical-scattering floor, response validity, physical local kinetics and replenishment checks |
| `runaway_rf_optimization.py` | Constrained physical slow-model optimization; equal-budget baselines; exported-control verification in independent kinetic evolution and selected resolved/PIC windows | Best-found mean local RF power, target/hold behavior, uncertainty, cost ledger, derivative checks and a precise statement of which model spans the full horizon |
| Optional `radiative_weibel.py` | Collisionless and collisional baselines followed by radiation | A clearly scoped 1D3V collective-physics extension after the required control cases |

In `runaway_rf.py`, distinguish the resolved short-time `pic_response` case from the slow `kinetic_hold` case in its settings, metadata and figure labels. The response-validation case compares resolved fields with the reduced operator on held-out distributions/amplitudes and includes a coherent-trapping failure case. The source-on case includes significant avalanche, primary-source and reservoir physics rather than relying on a source-free terminal spectrum. Reuse the same physical kernels in all cases.

In `runaway_rf_optimization.py`, verification is a required mode/section of the same deliverable. Load the selected controls without refitting them, run independent kinetic and selected PIC-window comparisons, and report whether the observed discrepancies are smaller than the claimed improvement. A complete physical-horizon reduced-model result must not be labeled as a complete physical-horizon PIC result.

A scaled-radiation robustness test, if useful, belongs in a clearly labeled configuration of an existing cooling example or a regression test. It is supplementary numerical verification and never substitutes for the physical fusion parameters. Do not create a separate file for every validation case.

Do not postpone examples until after all features are merged. Each implementation step lands with the smallest physically discriminating example and matching documentation. Tests should check failure modes and invariants, not simply repeat the implementation formula.

## 10. Acceptance gates

The following are proposed initial project gates. Tighten them when the effect being claimed is smaller; justify changes using an error budget.

1. **Force gate:** The base LL tests and units/macroparticle-weight checks pass before RF control is attempted.
2. **Collision gate:** Single-species equilibrium, drag/scattering rates, interspecies conservation or declared reservoir exchange, and relativistic tail behavior agree with the independent collision specification.
3. **RF gate:** Resonance location and the collisionless prescribed-wave invariant converge; supplied fields satisfy Maxwell/Faraday consistency; the driven PIC source ledger closes.
4. **Resolution gate:** Numerical heating/scattering is less than 10% of the physical RF response used for optimization, and both response and gradient converge under at least one meaningful refinement. If the target gain is only 10%, use a tighter budget.
5. **Reduced-model gate:** On held-out local response tests, relevant moment changes, absorbed power and tail-transfer rates agree within a proposed 10% tolerance or within a stated statistical interval that is smaller than the claimed control improvement. A coherent-trapping regime outside the fit is excluded or modeled explicitly.
6. **Physical-timescale gate:** Benchmark a representative short run, estimate full-horizon cost and memory, and choose the declared model hierarchy before a long optimization. Do not quietly change physical collision/radiation coefficients to meet a runtime budget.
7. **Gradient gate:** Compare AD directional derivatives with converged common-sample finite differences or an independent tangent/reference calculation. Repeat at more than one control and avoid interpreting a noisy zero as success. A practical initial target is 1% agreement for smooth deterministic examples; for stochastic cases use confidence intervals and variance convergence.
8. **Source gate:** Quantify avalanche, primary production and boundary/reservoir effects over both control and hold times. Neglected terms must fit within the selected error budget and a source-on cross-check.
9. **Optimization gate:** Beat the best matched-budget baseline on held-out initial conditions/phases; report all costs, including bulk heating and residual wave energy. Report a best-found solution unless a genuine global bound is available.
10. **Durability gate:** The unsmoothed high-energy population stays below the declared target throughout the RF-off hold. Also report reacceleration after that interval when relevant; do not call a finite hold indefinite stability.
11. **Independent verification gate:** Verify the reduced-model optimum with independent kinetic evolution and selected physical PIC windows. If a full uninterrupted PIC horizon was infeasible, state that plainly.
12. **Scope gate:** The conclusion is a local control result. No equilibrium, accessibility, antenna-power or device-mitigation claim is added without its own evidence.

## 11. Future quantum roadmap, after the classical milestone

Only after the classical force, collisions, RF work, physical slow model and control examples are verified should the project add quantum-corrected continuous emission, a validated stochastic photon-emission model, and eventually pair production for strong-field applications. A scalar quantum suppression factor changes mean losses but does not reproduce stochastic spectral broadening. Discrete emission events need an estimator for derivatives of expectations; naive AD through a fixed random branch is not sufficient. Those capabilities require separate analytical/statistical tests and are not part of the fusion baseline, whose \(\chi_e\) is tiny.

## References and evidence records

The companion `references/runaway_references.json` records URLs, the supported use of each reference, and tool evidence. The links below are the stable entry points.

- **R1.** Choudhury et al., *First Demonstration of Resonant Pitch-Angle Scattering of Relativistic Electrons by Externally Launched Helicon Waves*, Physical Review Letters **136**, 025101 (2026). https://doi.org/10.1103/jz8f-ck22
- **R2.** Choudhury et al., *Resonant Pitch-Angle Scattering Of Runaway-Electrons by Externally-launched Helicon Waves in the DIII-D Tokamak*, arXiv:2606.22141 (2026), with a subsequent PPCF/institutional record. https://arxiv.org/abs/2606.22141 ; https://infoscience.epfl.ch/entities/publication/1e6a28b7-4eeb-407d-aa09-bb88e7fed23d
- **R3.** Choudhury et al., *Detailed characterization of runaway electron driven whistler waves in low-density DIII-D discharges*, Physics of Plasmas **32**, 092508 (2025). https://doi.org/10.1063/5.0277589
- **R4.** Liu et al., *Role of Kinetic Instability in Runaway-Electron Avalanches and Elevated Critical Electric Fields*, Physical Review Letters **120**, 265001 (2018). https://doi.org/10.1103/PhysRevLett.120.265001
- **R5.** Ghai, Del-Castillo-Negrete, Spong and Beidler, *Non-diffusive pitch-angle scattering of runaway electrons in the presence of whistler waves*, Physics of Plasmas **33**, 102501 (2026). https://doi.org/10.1063/5.0346141 ; https://impact.ornl.gov/en/publications/non-diffusive-pitch-angle-scattering-of-runaway-electrons-in-the-/
- **R6.** DREAM documentation, collision settings and their domain of validity. https://ft.nephy.chalmers.se/dream/frontend/api/Settings/collisions.html
- **R7.** Embréus, Stahl and Fülöp, *On the relativistic large-angle electron collision operator for runaway avalanches in plasmas*, Journal of Plasma Physics **84**, 905840102 (2018). https://doi.org/10.1017/S002237781700099X
- **R8.** Landreman, Stahl and Fülöp, *Numerical calculation of the runaway electron distribution function and associated synchrotron emission*, Computer Physics Communications **185**, 847–855 (2014). https://doi.org/10.1016/j.cpc.2013.12.004
- **R9.** Hoppe, Embréus and Fülöp, *DREAM: A fluid-kinetic framework for tokamak disruption runaway electron simulations*, Computer Physics Communications **268**, 108098 (2021). https://doi.org/10.1016/j.cpc.2021.108098 ; https://github.com/chalmersplasmatheory/DREAM
- **R10.** Hesslow et al., *Effect of Partially Screened Nuclei on Fast-Electron Dynamics*, Physical Review Letters **118**, 255001 (2017). https://doi.org/10.1103/PhysRevLett.118.255001
- **R11.** Hesslow et al., *Effect of partially ionized impurities and radiation on the effective critical electric field for runaway generation*, Plasma Physics and Controlled Fusion **60**, 074010 (2018). https://doi.org/10.1088/1361-6587/aac33e
