> Original research specification, preserved for equations, sources and remaining work. Some API proposals and implementation-status statements are historical. Read `../START_HERE.md` first; the bundled branch and its documentation define the implemented API. Historical `evidence/` and `probes/` paths refer to the earlier full research archive, not to new production validation. Current reproduction is in `../VALIDATION.md`.

# Relativistic collisions for a differentiable 1D3V JAX PIC code

## Decision and status

Implement **conservative elastic relativistic binary collisions with equal macro-particle weights first**. Use physical species masses and charges, cumulative Coulomb scattering in the pair center-of-momentum (COM) frame, a documented Coulomb logarithm, and explicit cell/pair normalization. The physical target is the leading-logarithmic, small-angle relativistic Coulomb collision operator in a classical, weakly coupled, nondegenerate plasma. This permits thermal relaxation and Coulomb slowing/scattering of a relativistic tail in one operator. It does not supply a complete disruption, impurity, bremsstrahlung, or avalanche model.

The recommended lineage is Takizuka–Abe random pairing, Nanbu cumulative angles, and the relativistic formulation of Pérez et al. Modern corrections matter: use the current audited rate and Coulomb-log definitions below, and Higginson's correction if unequal weights are introduced. [Takizuka–Abe 1977](https://doi.org/10.1016/0021-9991(77)90099-7), [Nanbu 1997](https://doi.org/10.1103/PhysRevE.55.4642), [Pérez et al. 2012](https://doi.org/10.1063/1.4742167), [Higginson et al. 2020](https://doi.org/10.1016/j.jcp.2020.109450).

**Included and executed:** `com_collision.py` implements elastic pair kinematics and an uncapped physical scattering strength; `verify_com_collision.py` tests those functions in JAX CPU float64. **Not implemented or validated:** cell pairing, a Coulomb angle sampler, full collision equilibrium, unequal weights, PIC coupling, or an unbiased stochastic gradient estimator. This is an executable building block and a specification for the remaining work. Do not label the supplied files a finished collision module.

A separate fixed-background relativistic Fokker–Planck/SDE path is useful for low-cost runaway test-particle studies and an independent benchmark. AMCC starts from Beliaev–Budker collisions against a specified Maxwell–Jüttner bath. The bath absorbs momentum and energy; this path is not a conservative evolving plasma. Keep it as a separately named approximation. [Särkimäki, Hirvijoki & Terävä 2018](https://doi.org/10.1016/j.cpc.2017.09.025), [author preprint](https://arxiv.org/abs/1701.05043).

## 1. Physical and software contract

Use the following input conventions throughout the collision kernel:

- The production state is `u_code = p/m = gamma*v`, in m/s, matching `Species.u` and `State.u`. This document uses the distinct dimensionless reference variable `eta = p/(m*c) = u_code/c`. Its mass shell is `gamma = sqrt(1+dot(eta,eta))`; its physical velocity is `c*eta/gamma`. At the reference-kernel boundary, divide production momenta by `c` on entry and multiply outputs by `c` on exit. Never pass SI `State.u` directly to the dimensionless reference functions.
- `m`, `q` are the physical mass and charge of a single species particle. The macro-particle multiplicity `w` is not a modified physical mass or charge.
- Number density is SI m^-3, time is seconds, charges are coulombs, and masses are kilograms in the reference rate. The equal-weight kinematics may instead use any common mass unit.
- In the production 1D3V code, a weight is particle number per represented transverse area. Hence `n = sum(w_code)/dx` is a three-dimensional density in m^-3. The equivalent volume notation below uses `V_cell = dx*A_perp` and number weights `w = w_code*A_perp`; the arbitrary area cancels. Do not divide `w_code` by `dx*A_perp` again. A simulation of planar fields does not turn Coulomb collisions into a one-dimensional cross section.
- For a cell with weights `w_i`, the physical number density is `sum(w_i)/V_cell`, subject to the finite-particle self-exclusion and pairing corrections below.
- Binning must use actual particle positions and active masks. Boundary sentinels, inactive padding and particles awaiting injection never participate or contribute to density.
- The input Coulomb logarithm and validity flags are part of the model configuration, not implicit numerical constants.

Return updated momenta, the next PRNG state, collision energy/momentum residuals, and diagnostics of effective density, scattering strength, invalid/degenerate pairs, and maximum/mean subcycle count. A production implementation should expose the selected model and normalization in the run metadata.

For continuum benchmarking, Braams and Karney's relativistic differential-potential formulation provides a connection to the Beliaev–Budker operator. This is also a useful independent route to friction/diffusion coefficients. [Braams & Karney 1987](https://doi.org/10.1103/PhysRevLett.59.1817).

## 2. Conservative pair kinematics: implementable equations

These equations follow directly from special-relativistic four-momentum conservation. They avoid a division by the squared COM boost velocity, which is problematic when that velocity vanishes.

Let

\[
\gamma_i=\sqrt{1+\eta_i^2},\qquad r=\gamma_1\gamma_2-\boldsymbol{\eta}_1\cdot\boldsymbol{\eta}_2,\qquad
M=\sqrt{m_1^2+m_2^2+2m_1m_2r}.
\]

Define mass-unit energy and momentum quantities

\[
E=m_1\gamma_1+m_2\gamma_2,\qquad
\mathbf P=m_1\boldsymbol{\eta}_1+m_2\boldsymbol{\eta}_2.
\]

The actual total energy is `c^2*E`, and actual total momentum is `c*P`. The pair COM energies divided by `c^2` are

\[
e_1^*=\frac{m_1(m_1+m_2r)}{M},\qquad
e_2^*=\frac{m_2(m_2+m_1r)}{M}.
\]

The COM momentum of particle 1 divided by `c` is

\[
\mathbf k^*=m_1\boldsymbol{\eta}_1-
\frac{m_1\gamma_1+e_1^*}{E+M}\mathbf P.
\]

Particle 2 has the opposite COM momentum. With `n = k*/|k*|`, choose a unit vector `t` perpendicular to `n`; a projected isotropic Gaussian vector supplies a uniform azimuth. Given a sampled polar scattering angle `chi`, rotate

\[
\mathbf k^{*\prime}=|\mathbf k^*|(\cos\chi\,\mathbf n+\sin\chi\,\mathbf t).
\]

Boost back without changing either COM energy:

\[
\mathbf k_1'=\mathbf k^{*\prime}+
\frac{e_1^*+\mathbf P\cdot\mathbf k^{*\prime}/(E+M)}{M}\mathbf P,
\quad
\mathbf k_2'=\mathbf P-\mathbf k_1',
\quad
\boldsymbol{\eta}_i'=\mathbf k_i'/m_i.
\]

This preserves total momentum and total relativistic energy algebraically. Equal macro weights then preserve the represented plasma's energy and momentum for each pair. Unequal macro weights require an additional statistically correct algorithm; replacing physical masses by macro masses is not that algorithm.

For almost comoving pairs, computing `r` by subtracting two large terms can lose precision. The prototype uses the equivalent expression

\[
r=1+\frac12\left(|\boldsymbol{\eta}_1-\boldsymbol{\eta}_2|^2-(\gamma_1-\gamma_2)^2\right),\qquad
\gamma_1-\gamma_2=
\frac{(\boldsymbol{\eta}_1-\boldsymbol{\eta}_2)\cdot(\boldsymbol{\eta}_1+\boldsymbol{\eta}_2)}{\gamma_1+\gamma_2}.
\]

This improves a common cancellation; it is not a proof of stability at arbitrarily large co-propagating Lorentz factors. Keep extreme-boost tests and a supported-range diagnostic. Identical proper momenta produce the identity map. A degenerate supplied tangent uses a deterministic perpendicular fallback to retain forward conservation. Gradients at these degeneracies and fallback chart boundaries are not validated.

## 3. Physical scattering strength and exact source mapping

The following definition matches the uncapped cross-section form in the current official WarpX `UpdateMomentumPerezElastic` implementation. Its source explicitly maps the rate to **Pérez Eq. (9)** when the effective cross section is `pi*b0^2*logLambda`, the angle selection to **Pérez Sec. II.C**, and the COM rotation to **Pérez Eq. (12)**. The code also documents the Coulomb-log correction to **Pérez Eq. (22)**. The full Pérez paper was not independently retrieved during this audit; the equation-number mapping is verified from the maintained implementation and its correction notes, not inferred from an inaccessible equation image.

Pinned source: [WarpX UpdateMomentumPerezElastic.H, commit 705c2de](https://github.com/BLAST-WarpX/warpx/blob/705c2deccfec137380cc940d633eb4c4665b97ed/Source/Particles/Collision/BinaryCollision/Coulomb/UpdateMomentumPerezElastic.H). Retrieved content blob: `45a778e08863a77a3cb178a08e89225f48a6812b`. Also see [official rendered source](https://warpx.readthedocs.io/en/26.06/_static/doxyhtml/_update_momentum_perez_elastic_8_h_source.html) and [correction discussion](https://github.com/BLAST-WarpX/warpx/issues/429).

Set physical COM momentum magnitude `pstar = c*|kstar|`, `gamma_i_star=e_i_star/m_i`, and

\[
\mu_R^*=\frac{m_1\gamma_1^*m_2\gamma_2^*}{m_1\gamma_1^*+m_2\gamma_2^*},\quad
v_M^*=\frac{p^*}{\mu_R^*},\quad
v_{\rm inv}=\frac{v_M^*}{1+p^{*2}/(m_1\gamma_1^*m_2\gamma_2^*c^2)}.
\]

Here `v_M_star` is the COM speed difference and may exceed `c`; `v_inv` is the invariant relative speed and cannot. They must not be substituted for one another.

Define

\[
b_0=\frac{|q_1q_2|}{2\pi\epsilon_0\mu_R^*v_M^*v_{\rm inv}},\qquad
\sigma_{\rm eff}=\pi b_0^2\ln\Lambda,
\]

\[
\boxed{s_{12}=n_{\rm eff}\Delta t\,\sigma_{\rm eff}\,v_M^*
\frac{\gamma_1^*\gamma_2^*}{\gamma_1\gamma_2}}.
\]

`s12/dt` is the angular-transport rate for this cumulative model (the small-step decay rate of the first directional moment), not a hard-event collision count. `b0` in this convention is twice the nonrelativistic 90-degree impact parameter. The symbol `sigma_eff` is an angular-transport effective area. Treating it as an unqualified hard-collision total cross section would mix models.

An independent algebraic check follows from

\[
v_M^{\rm lab}=c\frac{\sqrt{r^2-1}}{\gamma_1\gamma_2},\quad
v_{\rm inv}=c\frac{\sqrt{r^2-1}}{r},\quad
p^*=\frac{m_1m_2c\sqrt{r^2-1}}{M}.
\]

Thus the same strength can be written

\[
s_{12}=\frac{q_1^2q_2^2 n_{\rm eff}\Delta t\ln\Lambda}{4\pi\epsilon_0^2c^3}
\frac{M^2r^2}{m_1^2m_2^2\gamma_1\gamma_2(r^2-1)^{3/2}}.
\]

This second form is useful as a test oracle away from `r=1`; it is not the proposed production evaluation near that singular limit. In the nonrelativistic limit it reduces to

\[
s_{12}\longrightarrow
\frac{q_1^2q_2^2 n_{\rm eff}\Delta t\ln\Lambda}
{4\pi\epsilon_0^2\mu^2v_{\rm rel}^3},\qquad
\mu=\frac{m_1m_2}{m_1+m_2}.
\]

The supplied reference implementation passes both this limit and the independent invariant expression. The prototype function arguments named `u1,u2` are reference-local names for `eta1,eta2`, not production SI `u_code`. Its `strength_si` name means the physical masses, charges, density and time are SI; the momentum arguments remain dimensionless.

All constants and SI rate calculations currently require float64: squaring SI electron masses in float32 underflows. A performant mixed-precision version should first algebraically normalize masses and constants, then be compared with this float64 reference. Merely changing array dtypes is unsafe.

### Coulomb logarithm

Start with an explicit fixed positive `logLambda` in homogeneous benchmarks, so errors in rates and weights cannot be hidden by a screening calculation. For a later dynamic calculation, document the background frame and the impact-parameter prescription. A conventional weak-coupling construction is

\[
b_{\min}=\max(b_0/2,\hbar/(2p^*)),\qquad
\ln\Lambda\simeq\ln(b_{\max}/b_{\min}),
\]

or a specified regulated version `0.5*log(1+bmax^2/bmin^2)`. The screening length must come from the physical thermal background, including the relevant species. A drifting lab-frame velocity variance is not automatically a rest-frame temperature.

The [official Smilei documentation](https://smileipic.github.io/Smilei/Understand/collisions.html) explicitly flags an erroneous square in Pérez Eq. (22) and states that Higginson's weighting correction is included. The pinned WarpX implementation additionally explains its use of `hbar/(2*pstar)` instead of a formula involving `h`. It also uses a minimum logarithm and a maximum cross section at low temperature/high density. Those caps are model extensions; they do not establish controlled accuracy in strong coupling. For the first weakly coupled module, report out-of-regime conditions instead of silently using a cap to claim validity.

### Required production rate-limit gates

The executable `strength_si` deliberately has no zero-relative guard, low-velocity floor, automatic screening, timestep selection, or cross-section cap. Its successful nondegenerate checks do not validate those missing regimes. Before production use:

- Exactly comoving pairs must take the identity kinematic map without evaluating a singular rate. Inactive pairs must be given finite dummy inputs before evaluating differentiable kernels; masking a NaN afterward is insufficient.
- Nearly comoving pairs require a supported precision range, a stable evaluation or limiting map, and convergence tests of any threshold. A large cumulative deflection at a tiny relative momentum can still produce a tiny physical momentum change; discarding all such pairs based only on a large rate is not justified.
- The screening prescription, quantum/classical impact-parameter transition, low-log regime and any finite cross-section cap must be explicit model choices. Compare rate and equilibrium behavior with an independent implementation over the advertised thermal-to-runaway range.
- Validate scattering-strength and timestep handling through isotropization, slowing down and equilibrium. An angle saturating at an isotropic distribution does not establish accuracy of a large physical time interval.
- Check the production/reference conversion `eta=u_code/c`, the corresponding output conversion, and the area-independent density normalization directly. The included adapter check verifies only the dimensional conversion around an isolated pair, not production integration.

## 4. Angular sampling and timestep control

For a reference Nanbu implementation, define an inverse-Langevin parameter `A>0` by

\[
\coth A-1/A=e^{-s}.
\]

Then the axisymmetric distribution of `mu=cos(chi)` is proportional to `exp(A*mu)` on `[-1,1]`, and the inverse CDF is

\[
\mu=\frac{1}{A}\log\left(e^{-A}+U(e^A-e^{-A})\right),\quad U\in(0,1).
\]

This definition has `E[cos(chi)]=exp(-s)` and therefore `E[chi^2]≈2s` for a small step. This is the convention compatible with the rate and small-`s` WarpX sampler above; do not mix quantities named `s` from different descriptions without checking their angle moments. The exact first moment follows by differentiating the stated distribution's normalization. Solving for `A` exactly samples this chosen Nanbu surrogate accurately; it does not turn its finite-step distribution into the exact relativistic collision evolution or the exact spherical-diffusion heat kernel. [Nanbu 1997](https://doi.org/10.1103/PhysRevE.55.4642), [Bobylev & Nanbu 2000](https://doi.org/10.1103/PhysRevE.61.4576).

Implement the root and inverse CDF with small-`A` expansions, large-`A` log-domain evaluation, bounded iterations and residual reporting. Derive the root's sensitivity from the defining equation if an implicit derivative is needed. The singular-looking limits `s→0` and `A→0` require analytic limits. Do not differentiate through failed roots or hide the failure with clipping.

The current Smilei angle fit is a useful fast alternative after statistical verification; the older WarpX Nanbu implementation is piecewise and has a small-`s` rejection loop. An exact truncated inverse CDF can replace that loop if reproducing that branch, but this reproduces the branch approximation rather than making it the exact full cumulative distribution. Fixed-shape JAX kernels should avoid open-ended rejection. Benchmark any faster fit against the reference's angular moments and physical relaxation rates before replacing it.

As an initial engineering target, keep the measured strengths mainly below `s=0.1` using fixed subcycles, then demonstrate convergence under halving the collision interval. This threshold is a proposed conservative starting point, not a universal accuracy theorem. Large `s` saturation prevents nonsensical angles but cannot by itself resolve energy relaxation or state-dependent coefficients. Subcycle only active cells/pairs if a static implementation can do so without changing particle visitation or random-stream semantics.

## 5. Pair normalization and unequal weights

Random partner choice supplies an estimator of the full collision sum. Its effective density must account for how often each particle is visited; a correct two-body rate alone is insufficient.

For self-collisions in a cell with `N` equal-weight particles and even `N`, a disjoint random perfect matching visits each particle once. A finite-particle estimator that excludes the particle itself uses `n_eff=(N-1)*w/V`. This approaches the continuum density `N*w/V` as `N` grows. For odd `N`, one simple design leaves a uniformly random particle unpaired; the conditional strength then uses `N*w/V` for the participating pairs so the expected per-particle coefficient remains `(N-1)*w/V`. Test that rate explicitly. This is an unbiased finite-sample generator normalization in the small-step limit, not an exact finite-step relaxation law for a nonlinear cumulative-angle sampler. Other visitation schemes are possible, but their corrections cannot be swapped blindly.

For cross-species collisions with different counts, repeatedly visiting the minority species requires visit-count corrections and sequential updates of repeated partners. A vectorized batch must contain disjoint pairs; multiple unsynchronized writes to the same particle are an algorithmic error. The pinned WarpX routine takes `n_eff=max(w1,w2)*min(N1,N2)/V` as an argument for its pairing design. Smilei's `BinaryProcesses.cpp` explicitly accounts for unequal visit counts. Port one complete, source-audited pairing-and-rate design rather than combining isolated formulas from different schemes.

Higginson et al. showed that earlier arbitrary-weight rules can yield incorrect scattering frequencies and provided a correction. Standard stochastic unequal-weight updates preserve the weighted moments in expectation, not at each realization. For a candidate physical pair scatter, the common acceptance factors are `P1=min(1,w2/w1)` and `P2=min(1,w1/w2)`, with the pairing rate chosen consistently. Their meaning is straightforward: expected weighted changes have the same coefficient because `w1*P1=w2*P2=min(w1,w2)`. This algebra does not establish the entire algorithm's frequency normalization. [Higginson et al. 2020](https://doi.org/10.1016/j.jcp.2020.109450).

Angus et al. (2025) add a moment-restoring procedure after weighted scattering and state that their energy restoration supports both relativistic and nonrelativistic particles. That is a promising later stage when exact conservation and variable weights are required. It introduces extra coupling between particles, corrections, and gradient questions; it is not necessary for the first equal-weight implementation. Do not use an ad hoc global temperature rescale and label it this method. [Angus et al. 2025](https://doi.org/10.1016/j.jcp.2025.113927), [preprint](https://arxiv.org/abs/2407.19151).

## 6. JAX execution and differentiability

A practical first implementation has fixed particle capacities, fixed species membership, static subcycle bounds and explicit masks. Use a structure-of-arrays layout or another measured layout with efficient gathers. Form random pairings per cell with permutation keys, then apply `vmap` to each disjoint pair batch and `lax.scan` to subcycles. Deposit updated state once per batch. Sorting costs `O(N log N)`; bucket methods can approach `O(N+Ncell)` with careful capacity/overflow handling. Pair interactions are `O(N)`, not all-to-all `O(N^2)`.

For masked inactive or exactly comoving pairs, supply safe finite dummy inputs before evaluating the rate and its derivatives; evaluating a singular rate and masking its result afterward can still propagate NaNs in reverse mode.

Cache compilation for fixed shapes. Measure execution only after synchronization and compilation warm-up. Report pair-updates per second, binning fraction, memory, gradient memory and the device/dtype; the included prototype supplies no GPU performance claim. Use rematerialization/checkpointing when differentiating long scans, selected from an actual memory profile.

A fixed random seed permits replay and common-random-number finite differences. It does **not** make a Monte Carlo simulation deterministic in the statistical sense or guarantee a gradient of an ensemble quantity. The derivative of one sampled trajectory is only one sample of a candidate estimator. For smooth parameter-dependent collision strengths and parameter-independent uniform/Gaussian draws, pathwise differentiation is attractive. Its validity still requires an interchange of expectation and derivative and finite variance; both become delicate near degeneracies and rare tails.

Hard cell boundaries, sorting changes, Bernoulli decisions, selected particle identities, clipping and particle creation produce nonsmooth or discrete dependence on parameters. Stopping gradients through them defines a conditional/surrogate derivative. It must be named and validated. An unequal-weight acceptance probability that depends on an optimized weight generally cannot be handled by differentiating only accepted momenta.

Start with gradients in a spatially homogeneous box and a prescribed pairing schedule. Optimize a continuous parameter such as collision multiplier, density, temperature or an external field, then compare ensemble-mean AD with common-random-number finite differences over many seeds. Confirm convergence with seed count, timestep and particle count. Add spatial design variables only after a separate study of bin crossing and partner-selection effects.

Fixed timestep/subcycle schedules are preferable initially. For adaptive SDE alternatives, a rejected Brownian increment cannot simply be discarded and redrawn: selection can bias the process. Brownian-bridge treatment is required by the chosen adaptive scheme. AMCC demonstrates that careless adaptive stochastic stepping can produce substantial errors. [Särkimäki et al. 2018](https://doi.org/10.1016/j.cpc.2017.09.025).

## 7. Validation matrix and acceptance gates

The following are proposed acceptance requirements. Tolerances for statistical tests must be fixed before seeing the results and tied to uncertainty, not merely selected to pass.

| Gate | Test | Required evidence |
|---|---|---|
| Units/adapter | `eta=u_code/c` on entry, `u_code=c*eta` on exit; number-per-area density and arbitrary-area invariance | Physical gamma and pair invariants agree across conventions; test a relativistic state that would fail conspicuously if `c` were omitted |
| Kinematics | Pair energy, momentum and mass shell; same-mass and physical electron/proton ratio; zero COM boost; comoving pair; degenerate tangent | Float64 errors consistent with roundoff on the supported momentum range; no nonfinite outputs |
| Rates | Nonrelativistic `v^-3` limit; invariant formula above; species exchange; density/time/log scaling | Independent scalar calculations and agreement with the pinned official rate at specified states |
| Angles | `E[cos chi]`, second moments and azimuthal isotropy for a grid of `s` | Confidence intervals around analytic/reference moments; endpoint and branch continuity checks |
| Pairing | Counts 0,1,2,3 and odd/even populations; unequal species counts; random partner histograms | No invalid partners or double writes; correct mean interaction rate and finite-`N` correction |
| Closed ensemble | Repeated collisions, no fields or radiation | Equal-weight total energy/momentum stay at floating-point tolerance; conservation does not drift with steps |
| Equilibrium | Maxwell–Jüttner at several temperatures and a common drift | No statistically significant systematic drift in distribution/moments; test boosted equilibrium independently |
| Relaxation | Temperature anisotropy, interspecies temperature exchange, beam relaxation | Rates converge in `dt`, particle count and cell size; compare independent kinetic/reference code |
| Relativistic transport | Fast electron stopping/scattering and relativistic resistivity | Match the same physical log, charge state, background and model; compare moments with uncertainty |
| Weighted extension | Ratios 1,2,10,100 with varied counts and mixtures | Correct frequencies, expectation conservation or documented exact restoration; no assumption from equal-weight success |
| Coupling | Closed PIC plus collisions; then RR plus collisions and externally driven cases | Combined energy ledger closes; anomalous PIC–MCC heating assessed separately from isolated conservation |
| Gradients | Smooth fixed pairing, then spatial bins, then weighted sampling | Multi-seed AD/FD consistency and uncertainty; identify unestimated discrete terms |
| Performance | Thermal plus sparse runaway tail at realistic cell occupancy | Profile complete bin/pair/update/diagnostic step and reverse pass; no hidden host loop or quadratic pair enumeration |

For Maxwell–Jüttner initialization, make the phase-space measure explicit. In the rest frame the probability density with respect to `d^3eta` is

\[
f(\boldsymbol{\eta})=\frac{\exp(-\gamma/\theta)}{4\pi\theta K_2(1/\theta)},\qquad
\theta=k_BT/(mc^2).
\]

The radial histogram includes its `4*pi*eta^2` Jacobian. Testing the wrong histogram measure can make a correct collision operator appear to fail. Boosted distributions also require the appropriate time-slice/phase-space interpretation; do not assume that simply boosting a fixed list of rest-frame samples automatically supplies every desired lab-frame initialization.

The supplied executed checks cover 512 random pairs per mass ratio with dimensionless proper-momentum scales `eta` from `10^-3` to `10^2`. Worst normalized momentum error was about `2e-16` and energy error below `7e-16`; one local angle derivative agreed with centered finite differences to `3.7e-11`. The nonrelativistic rate check agreed to `3.1e-8` relative, consistent with its finite small velocity, and the invariant rate expression agreed to `4.5e-16`. These checks are narrow, reproducible building-block tests, not equilibrium or full stochastic-gradient validation. See `com_validation.json`.

## 8. Physics beyond the first module

**Runaway generation and equilibrium tails.** Coupling electric acceleration, pitch-angle scattering, slowing down and classical synchrotron reaction is a good first application. A fixed thermal bath can isolate the tail mechanism cheaply. A conservative binary population is needed to study self-consistent thermal response. A 1D3V PIC box describes local kinetic dynamics and waves; tokamak geometry, magnetic trapping, radial losses and transport require added models or a different orbit treatment.

**Avalanche.** An accumulated large deflection from many small kicks is not a discrete hard Møller collision producing a knock-on electron. Add a separately matched large-angle relativistic electron-electron operator for avalanche work, with a shared angular/energy cutoff and subtraction of the overlap from the small-angle coefficient. Conservation and reverse scattering matter near threshold. [Embréus, Stahl & Fülöp 2018](https://doi.org/10.1017/S002237781700099X).

**Impurities.** Partial screening changes fast-electron collision rates and cannot generally be represented by a single effective charge or an ordinary Coulomb logarithm. Make charge state, bound electrons and screening an explicit later extension before interpreting impurity mitigation studies. [Hesslow et al. 2017](https://doi.org/10.1103/PhysRevLett.118.255001).

**Numerical energy.** Independently energy-conserving PIC and Monte Carlo operators can generate anomalous heating when composed. Vay et al. analyze explicit PIC–MCC coupling and give an energy-preserving arrangement; the published energy-balance derivation cited here is nonrelativistic. Use its lesson to design a relativistic discrete-work audit, not to claim a relativistic conservation theorem by analogy. [Vay et al. 2025](https://doi.org/10.1103/PhysRevE.111.025306).

**Research candidates.** Useful methods questions are whether a conservative relativistic PIC/RR/collision composition remains accurate under large timestep separation; whether gradients of runaway-tail observables converge through stochastic collisions and hard bins; and whether moment restoration for unequal weights can retain useful unbiased sensitivities. These are research directions, not claims of novelty. Survey the latest literature before framing a paper.

## 9. Minimal production pseudocode

```text
collision_step(state, config, key):
    validate model regime and SI/normalized-unit contract
    active = particles present in the physical domain and enabled species
    compute cell IDs and per-cell weighted number density
    build fixed-capacity cell lists; report overflow rather than dropping particles
    split independent PRNG streams by step, subcycle, cell, species pair
    for each statically scheduled subcycle and species-pair block:
        form randomized partners using one audited visitation scheme
        compute each pair's n_eff, including finite-count/visit corrections
        partition repeated partners into disjoint update batches
        for each disjoint batch:
            compute physical COM state
            classify exact comoving and invalid pairs
            compute the prescribed Coulomb logarithm and s12
            sample cumulative polar angle and uniform azimuth
            rotate COM momenta and boost back
            apply documented weight rule (equal weights in version 1)
            scatter both updated momenta back once
        accumulate weighted energy and momentum changes
    return state, key, physical/model/numerical diagnostics
```

Do not silently insert clipping, collision multipliers or conservation repairs while tuning tests. Each changes either the discrete algorithm or the physical model and needs a named configuration and its own checks.
