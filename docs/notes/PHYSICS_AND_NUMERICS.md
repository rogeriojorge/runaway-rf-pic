> Original research specification, preserved for equations, sources and remaining work. Some API proposals and implementation-status statements are historical. Read `../START_HERE.md` first; the bundled branch and its documentation define the implemented API. Historical `evidence/` and `probes/` paths refer to the earlier full research archive, not to new production validation. Current reproduction is in `../VALIDATION.md`.

# Radiation, prescribed fields, and discrete energy accounting

This document fixes the physical and numerical conventions for the proposed extension. It is a design and derivation, not a claim that these features are already integrated. Collision algorithms and the runaway-control model are specified in the companion documents.

## 1. State, units, and physical scope

Use the research branch's SI state convention throughout:

\[
\mathbf u=\mathbf p/m=\gamma\mathbf v,\qquad
\gamma=\sqrt{1+u^2/c^2},\qquad \mathbf v=\mathbf u/\gamma.
\]

Here `u` has units m/s and is not momentum normalized to `mc`. When dimensionless momentum is useful in a derivation or plot, write `p_hat = u/c` explicitly. The kinetic energy is evaluated stably as

\[
K=mc^2(\gamma-1)=\frac{m u^2}{\gamma+1}.
\]

The latter form avoids subtracting nearly equal numbers for a cold population. Physical species charge and mass are separate from macroparticle weight. In the existing 1D representation, weight is particle number per represented transverse area. Thus `sum(w*K)` is J/m², density is obtained from weights divided by longitudinal cell size, and a power budget is W/m² until a transverse area has been specified.

Retain all three momentum components. A 1D2V collisionless invariant subspace can be a useful special benchmark, but a physical 3D Coulomb operator generally scatters particles out of that subspace. A 1D2V model is not a substitute for 1D3V runaway pitch-angle physics.

### Momentum initialization

Add one unambiguous particle input, preferably `Species(u=...)` to match `State.u`, mutually exclusive with `v`. Provide a small helper or example expression for a beam with prescribed Lorentz factor and direction:

\[
\mathbf u_0=c\sqrt{\gamma_0^2-1}\,\hat{\mathbf n}.
\]

Do not convert this state through a velocity close to `c`. Validate shape, finite values, and parameter domains. Do not silently clip a requested relativistic momentum. Keep legacy velocity inputs supported, but replace silent loss of requested beam energy with documented validation. Initializing from an existing restart must bypass distribution resampling. The same requirement applies to diagnostics, archives, sources and walls: carry or reduce momentum directly where energy is needed, rather than reconstructing gamma from `1-v**2/c**2`. The included actual-diagnostic probe demonstrates the resulting precision loss.

Thermal relativistic examples require a Maxwell–Jüttner distribution in momentum, not a Gaussian in velocity with a speed clip. For a population at rest, with \(\Theta=k_BT/(mc^2)\),

\[
f(\hat{\mathbf p})=\frac{n}{4\pi\Theta K_2(1/\Theta)}e^{-\gamma/\Theta},\qquad
d^3\hat p=\hat p^2 d\hat p\,d\Omega.
\]

The radial sampling density includes the Jacobian \(\hat p^2\). Check sampled moments against numerical quadrature. An inverse-CDF construction can expose a useful temperature derivative; a rejection sampler has a different derivative problem. A spatially homogeneous drifting equilibrium must be sampled with the correct laboratory-frame measure: blindly boosting independently sampled rest-frame particles without the appropriate simultaneity/measure treatment is not sufficient validation.

## 2. Classical radiation reaction

The particle equation is

\[
\dot{\mathbf u}=\frac{q}{m}(\mathbf E+\mathbf v\times\mathbf B)+\frac{\mathbf F_R}{m}.
\]

Define

\[
\tau_0=\frac{q^2}{6\pi\epsilon_0mc^3},\quad
S=|\mathbf E+\mathbf v\times\mathbf B|^2-(\mathbf v\cdot\mathbf E)^2/c^2.
\]

The proposed production starting point is derivative-free Landau–Lifshitz, retaining all algebraic terms:

\[
\mathbf F_R=\frac{\tau_0q^2}{m}\left[
(\mathbf E+\mathbf v\times\mathbf B)\times\mathbf B
 +\frac{(\mathbf v\cdot\mathbf E)\mathbf E}{c^2}
 -\frac{\gamma^2 S\mathbf v}{c^2}\right].
\]

The full LL force also includes

\[
\mathbf F_{\partial F}=\tau_0 q\gamma[D\mathbf E+\mathbf v\times D\mathbf B],\qquad
D=\partial_t+\mathbf v\cdot\nabla.
\]

These definitions distinguish reduced LL from a momentum-parallel high-energy drag. The reduced model is the production candidate; a full-LL prescribed-field reference establishes the error from discarding field derivatives. Vranic et al. compare such models and their PIC implementation; the equations above are expressed in SI rather than their CGS convention. [Vranic et al. 2016](https://doi.org/10.1016/j.cpc.2016.04.002), [full preprint](https://arxiv.org/html/1502.02432v3).

Compute coefficients without forming avoidably tiny powers of SI particle masses. For example, use `tau = (q/m)*q/(6*pi*epsilon0*c**3)` and the acceleration coefficient `tau*(q/m)**2`. Float64 is the validation reference. Any float32 path must pass physical error tests over its advertised Lorentz-factor range.

For electrons,

\[
P_{\rm L}=\frac{\tau_0q^2}{m}\gamma^2S,\qquad
\chi_e=\frac{\gamma\sqrt S}{E_S},\qquad E_S=\frac{m_e^2c^3}{e\hbar}.
\]

`P_L` is the Liénard power evaluated with the Lorentz acceleration at the order used in LL. It is not an unconditional identity for minus the instantaneous mechanical work of full LL. This distinction determines the ledger below.

Classical applicability requires small quantum recoil and the smooth-field conditions of the classical reduction. Small `chi` does not imply small accumulated cooling. Record both the distribution of `chi` and a cooling indicator such as `P_L*dt/max(K,K_scale)`. Do not advertise a universal quantum threshold independent of the desired observable and tolerance.

### Stable cancellation

At large gamma, `|E+v×B|²-(v·E)²/c²` can lose accuracy by cancellation. A useful positive decomposition is

\[
S=|\mathbf E_\perp+\mathbf v\times\mathbf B|^2+E_\parallel^2/\gamma^2,
\]

where parallel refers to the direction of `v`. Supply a regular zero-momentum branch and test its derivative; a hard nonnegative clamp is not a substitute for stable algebra. For quantum models, the square root at exactly zero force also needs a consistent limiting implementation for the complete recoil expression.

### Macroparticles and radiation separation

Evaluate recoil using physical `q,m`. Accumulate emission as `sum(w*P_L)`. A split-particle test must preserve cooling and emission at fixed physical density. A naive replacement `q -> w*q, m -> w*m` strengthens the recoil incorrectly.

The added channel represents unresolved incoherent emission. Resolved coherent electromagnetic radiation stays in the Maxwell fields. Establish the frequency/scale separation in every collective radiation example. If separation is poor, specify a consistent cutoff or a more appropriate model; do not count the same radiation twice. A recoil force alone does not supply a photon spectrum or transport model.

## 3. Prescribed fields and parameter derivatives

Use one vectorized callback contract:

```python
def prescribed_fields(x, t, parameters):
    # x: (N, 3) in m; t: scalar in s; outputs: (N, 3), V/m and T
    return E, B
```

The callable is static configuration; its numerical parameters are dynamic pytree leaves. A prescribed field is pointwise: output i depends on position i, time and the supplied parameters, rather than on unrelated particle positions in the batch. An intended ensemble-dependent field requires its own coupling contract. Never capture optimized parameters in a static closure, convert them through NumPy, or call `float` on traced values. Existing static external arrays remain valid and additive under a documented convention. Reject accidental double specification of the same drive.

For analytically defined fields, potentials offer a convenient verification path:

\[
\mathbf E=-\nabla\phi-\partial_t\mathbf A,\qquad
\mathbf B=\nabla\times\mathbf A.
\]

This enforces the homogeneous Maxwell identities. An RF or laser example should derive its E and B from one potential or check those identities analytically. A finite pulse requires the envelope derivative in E. Prescribing a varying B with an unrelated E is not an acceptable laser example.

A plane-wave example can use transverse `A(phase)` with `phase = omega*(t-x/c)`. A local RF example can use a transverse circular or elliptical potential with `phase=k*x-omega*t` and a smooth envelope. RF frequency and wavevector must be chosen from a stated local dispersion model or declared prescribed forcing; the local project does not require an antenna or equilibrium solver.

The full-LL material derivatives can be calculated by a JVP of the field callback in direction `(v,1)`, using the instantaneous trajectory tangent. This computes `partial_t+v·grad`, not a derivative through an entire trajectory. The inner field derivative treats v as its direction; outer differentiation of the LL force must still include how that instantaneous v depends on state and parameters. Do not insert `stop_gradient(v)` in the outer sensitivity calculation. Check this JVP against analytic derivatives for constant fields, a sinusoid, and a finite pulse.

Evaluate the callback at each integrator's actual stage time and position, including particle substeps. On restart, use `State.time`, not a new time origin. Test that splitting a run into two restarts gives the same pulse phase, random stream, diagnostics, and derivatives as an uninterrupted run.

## 4. Discrete energy ledger

Use the self-consistent fields alone in stored Maxwell energy:

\[
U_{\rm sc}=\sum_j\Delta x\,[\epsilon_0|E_{{\rm sc},j}|^2+|B_{{\rm sc},j}|^2/\mu_0]/2.
\]

Use the appropriate staggering and boundary convention already present in the solver. Identify its actual discrete invariant or bounded-error energy before asserting exact conservation; the continuum-looking expression alone is not a proof for a staggered update. Adding a large static prescribed B to this diagnostic would hide particle errors under irrelevant background energy.

Maintain the following cumulative signed quantities, in J/m²:

| Quantity | Definition and purpose |
| --- | --- |
| `prescribed_work` | Work on particles from prescribed E, evaluated at the same work velocity as the push |
| `rr_work` | Signed mechanical work by the RR force; often negative, but not artificially forced negative |
| `radiated_energy` | Integral of the chosen physical emissivity, nonnegative for classical emission |
| Existing wall/source ledger | Retain outgoing/incoming particle energy, externally injected particles, and boundary field exchange |
| `bath_work` when a reservoir is used | Energy transferred from a prescribed thermal bath; equal and opposite bath response if tracked |
| `drive_work` for an RF current source | Source work in the Maxwell energy equation, `-integral(J_drive dot E dx dt)`; do not add it twice as prescribed particle work |

For a closed particle collision operator, net collision work should vanish to its exact or statistical conservation tolerance. For a fixed bath, it does not vanish and must be recorded. This is essential when assessing control efficiency.

The mechanical balance is

\[
\Delta(K+U_{\rm sc})=W_{\rm prescribed}+W_{\rm drive}+W_{\rm RR}+W_{\rm bath}
 +W_{\rm particle\ source}-E_{\rm boundary,out}+\mathcal R_E.
\]

The radiation diagnostic and the mechanical RR sink are separate. Their difference contains bound-field/Schott terms, modeling terms from the chosen reduction, and quadrature errors. In suitable asymptotic finite-pulse tests, Schott boundary terms vanish; in perpendicular uniform-B cooling, the simple power balance is especially clean. Do not manufacture equality by defining physical emitted energy as `-delta_K` in every electromagnetic configuration.

At the order retained by full LL, the bound-field contribution is

\[
E_{\rm Schott}=-\tau_0(q/m)\,\mathbf p\cdot\mathbf E,
\qquad
\frac{d}{dt}(K+E_{\rm Schott})=q\mathbf v\cdot\mathbf E-P_{\rm L}+O(\tau_0^2).
\]

This uses the Lorentz acceleration in the reduction. The remaining higher-order/model terms are not all timestep error. For constant E parallel to v, mechanical LL recoil cancels although Lienard emission is positive; this is a decisive ledger test. In a pure static B the leading Schott term is zero. Record a physical/model balance residual separately from the exact numerical momentum-work residual.

Store enough metadata to identify the emissivity and force model used. Check that the diagnostic itself has correct derivatives; a detached or host-computed energy diagnostic cannot serve an optimization objective.

## 5. Explicit and implicit integration

### Explicit starting implementation

Compose a controlled second-order RR substep symmetrically with the existing Lorentz step, with stage-centered fields. Use explicit midpoint as an initial local reference and a stiff-capable alternative only when the cooling timescale requires it. Fixed substep counts make derivative verification straightforward. A negative-energy or momentum clamp must never conceal an underresolved cooling step.

For time-dependent fields, specify the nonautonomous splitting explicitly. A local RR-half/Lorentz-full/RR-half composition can freeze the external time at the common midpoint of one physical step, with compatible position stages, or use a derived extended-state split. The three operator durations must not advance the physical clock by twice the intended step. Derive and test the chosen time allocation on a nonconstant pulse.

Field/current updates must use the actual moved particle trajectories. Radiation carries no electric charge, so the charge-continuity identity must remain intact. Keep current deposition tied to the same orbit endpoints and work velocity as the field update.

Collision placement is a separate numerical issue. The 2025 analysis by Vay et al. shows how coupling individually conservative PIC and Monte Carlo collision updates can produce artificial heating. Their presented analysis is nonrelativistic; the relativistic extension must be derived and checked for this solver's staggering. Treat this as a required coupled test, not as a reason to claim exact conservation from a pair test alone. [Vay et al. 2025](https://doi.org/10.1103/PhysRevE.111.025306).

A symmetric deterministic composition can have order two. It does not automatically give weak order two for a Monte Carlo collision integrator. Measure deterministic order, stochastic weak order, and ensemble error separately.

### Energy-consistent implicit formulation

For two momentum states, define

\[
\overline{\mathbf v}=\frac{\mathbf u^{n+1}+\mathbf u^n}{\gamma^{n+1}+\gamma^n}.
\]

Direct algebra gives

\[
m\overline{\mathbf v}\cdot(\mathbf u^{n+1}-\mathbf u^n)
=mc^2(\gamma^{n+1}-\gamma^n)=\Delta K.
\]

For a local residual of the form

\[
\mathbf u^{n+1}-\mathbf u^n=\Delta t\left[
\frac qm(\overline{\mathbf E}+\overline{\mathbf v}\times\overline{\mathbf B})
+\frac{\overline{\mathbf F}_R}{m}\right],
\]

the discrete mechanical work is exactly

\[
\Delta K=q\Delta t\,\overline{\mathbf E}\cdot\overline{\mathbf v}
+\Delta t\,\overline{\mathbf F}_R\cdot\overline{\mathbf v},
\]

up to nonlinear residual error. A centered LL force evaluated at the midpoint momentum is one candidate; the complete field-particle algorithm must retain its compatible gather/deposit and electrostatic orbit-integral treatment. This derivation establishes a design constraint, not a proof that an arbitrary Boris-plus-kick implementation satisfies the coupled identity.

If the momentum residual is `r_u = delta_u-dt*rhs`, its extra work is exactly `m*vbar dot r_u`. Recording this term keeps an unconverged nonlinear solve from masquerading as a physical radiation or collision sink.

Reuse the research branch's field inversion and fixed Picard iteration first. Check particle and field residuals, radiation work, and timestep convergence together. A stable implicit Maxwell inversion does not remove the accuracy requirement to resolve cyclotron resonance, RF phase, plasma modes of interest, or nonlinear cooling.

If a converged solve becomes too expensive to differentiate by taping iterations, use implicit differentiation of its residual. For `R(z,theta)=0`, solve

\[
R_z^T\lambda=J_z^T,\qquad dJ/d\theta=J_\theta-\lambda^T R_\theta.
\]

This derivative belongs to the converged equation. An unconverged fixed-iteration forward map has a different derivative. Require a primal residual tolerance and a transposed-solve residual tolerance. Support both JVP and VJP where the public API promises both. Reuse SOLVAX only if its supported interface reduces tested code or measured cost; do not add a dependency solely for a three-component local solve.

## 6. Analytical checks and resonance accuracy

For perpendicular uniform-B motion set

\[
a=\tau_0(|q|B/m)^2,\quad s_0=\operatorname{arctanh}(1/\gamma_0),
\quad z=\frac{\gamma_0-1}{\gamma_0+1}e^{-2at}.
\]

Then

\[
\gamma(t)=\frac{1+z}{1-z},\qquad
\frac{\partial\gamma}{\partial B}=-\frac{2at}{B}(\gamma^2-1).
\]

For initial momentum along x and B along z,

\[
\varphi(t)=-\frac{qB}{ma}\log\!\left[\frac{\cosh(s_0+at)}{\cosh s_0}\right],
\quad
\mathbf u(t)=c\sqrt{\gamma^2-1}(\cos\varphi,\sin\varphi,0).
\]

These formulas are derived by integrating `dot(gamma)=-a(gamma²-1)` and `dot(phi)=-qB/(m*gamma)`. Use stable `logcosh` differences. A reusable reference must implement the B=0 and gamma0=1 limits explicitly; the supplied probe deliberately uses nonzero B and gamma0>1. Energy and phase must both converge. The supplied actual-Boris prototype demonstrates that energy derivatives can be excellent at a timestep with a poor final momentum direction.

A more stable reusable energy expression, with delta=`a*t`, is

\[
\gamma(t)=\frac{\gamma_0+\tanh\delta}{1+\gamma_0\tanh\delta},\qquad
\partial_B\gamma=-2\tau_0(q/m)^2Bt(\gamma^2-1).
\]

For the phase logarithm use `logcosh(delta)+log1p(tanh(delta)/gamma0)`, avoiding subtraction of close logcosh values; its ratio to delta tends to `1/gamma0`. Use a stable logcosh implementation, not `log(cosh(x))` when x can overflow. These improvements are recommended for the production analytical helper; the fixed-case probe's tested implementation and evidence are retained unchanged.

For an oblique orbit, `v_parallel` is constant under uniform-field LL while `p_parallel` is not. For E parallel to v in a constant uniform electric field, the algebraic LL force cancels; emitted-power accounting still requires the appropriate near-field contribution.

Use the arbitrary plane-wave full-LL solution as a second reference. Compare like equations: the derivative-free model can have a nonzero modeling discrepancy that does not vanish with numerical refinement. [Di Piazza 2008](https://arxiv.org/abs/0801.1751); [Yaremko, constant-field solution](https://doi.org/10.1063/1.4820131).

For RF waves define a signed gyrofrequency `Omega_s=q_s*B0/m_s` and verify the chosen polarization with

\[
\omega-k_\parallel v_\parallel-\ell\Omega_s/\gamma=0.
\]

The literature's positive/negative harmonic labels depend on sign conventions. Validate physical normal/anomalous resonance behavior rather than assuming a label has the intended meaning.

## 7. Quantum extension after classical acceptance

First add a quantum-corrected mean emissivity `P_q=g(chi)*P_L` in its stated high-energy approximation. Check the classical limit and table/interpolation derivatives. Then consider a drift–diffusion or deterministic kinetic description with a verified range of photon recoil. Niel et al. provide the model hierarchy and moment tests. [Niel et al. 2018](https://doi.org/10.1103/PhysRevE.97.043209).

Discrete emission needs photon rates and recoil, a capacity/overflow contract, independent random streams, and an estimator for derivatives of expectations. Fixed keys do not differentiate event probabilities: `d 1[U<p]/dp=0` almost everywhere while `d E[1[U<p]]/dp=1`. Score estimators, appropriate reparameterization, or deterministic kinetic derivatives must be validated against finite differences of expectations. [Arya et al. 2022](https://papers.nips.cc/paper_files/paper/2022/hash/43d8e5fc816c692f342493331d5e98fc-Abstract-Conference.html).

When splitting soft continuous radiation from hard discrete photons, define one cutoff and subtract the soft contribution consistently from the discrete rate. Never apply full LL losses and full photon recoil simultaneously. Validate the locally constant field approximation before using its rates. Pair creation, photon transport, absorption, and spin are additional physics, not consequences of setting a quantum flag. The RF runaway milestone is classical and must not wait for these extensions.

### Concrete quantum implementation stages

1. **Mean recoil reference.** Reuse the existing chi calculation, field stages, momentum state and emission ledger. Add a documented rate table or quadrature and its first energy moment. A scalar `g(chi)` applied to an ultrarelativistic drag is a specified approximation; it is not automatically a derivation for multiplying every full-LL term. Verify positivity, interpolation error, derivative continuity, the chi-to-zero limit and convergence of mean outgoing energy.
2. **Diffusion or deterministic kinetic reference.** For a differential photon rate `r(eta;gamma)`, form energy-loss drift `A=integral(r*delta_gamma d eta)` and diffusion `B=integral(r*delta_gamma**2 d eta)`. A Kramers–Moyal truncation gives `d gamma=-A dt+sqrt(B) dW` in its small-recoil regime. Its validity, boundary treatment and moment closure must be checked; a deterministic kinetic implementation can provide a cleaner differentiable expectation reference. Compare at least the mean, variance and skewness as recoil grows.
3. **Discrete nonlinear-Compton events.** Integrate optical depth with the total rate, sample photon fractions from the audited spectrum, and apply recoil under the selected angular approximation. Supply fixed-capacity photon/event buffers, overflow checks, independent keys, and a cutoff-independent energy ledger. Check waiting-time/spectrum statistics, field-stage consistency and emission-step convergence. Do not silently discard excess events or equate a collinear approximation with exact finite-energy four-momentum conservation in a background field.
4. **Differentiable expectations.** Compare pathwise terms plus any required event-probability estimator against a deterministic kinetic derivative and multi-seed finite differences. Event-time and event-count derivatives are part of the problem. Use score/control-variate or supported stochastic-AD methods only after demonstrating acceptable variance for the intended objective.
5. **Further physics only when required.** Add photon transport and Breit–Wheeler pair creation for a specified strong-field scenario, with charge/current/source consistency, thresholds, energy exchange and low-chi recovery. Atomic/ion bremsstrahlung, quantum synchrotron and nonlinear Compton are different processes; their overlap and regime must be explicit.

LCFA validity is photon-energy dependent: a short formation length for energetic emission does not make the soft-photon end correct. Use an independent plane-wave/strong-field-QED reference before extrapolating pulse spectra or control gradients. [Di Piazza et al. 2018](https://doi.org/10.1103/PhysRevA.98.012134). Extend `laser_electron_collision.py` with clearly separated classical/mean-quantum/stochastic modes once these gates pass. The fusion RF parameter proposal has chi of order `1e-8`, so its control conclusion does not need this quantum machinery.

For a later implementation in general prescribed fields, the [2019 follow-up](https://doi.org/10.1103/PhysRevA.99.022125), [open preprint](https://arxiv.org/abs/1811.05834), extends the infrared correction to spacetime-dependent backgrounds and gives quantitative beyond-LCFA comparisons. Start with its plane-wave checks before applying a local-rate approximation to arbitrary callbacks. This is a follow-on reference, not a quantum feature of the present branch.
