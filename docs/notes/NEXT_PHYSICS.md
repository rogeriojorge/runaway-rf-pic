# Local no-RF kinetic precursor

**Continuation:** the original precursor below is preserved. New channel audits,
four 64-seed marker/timestep pilots and a cold-target variance reference are in
`CONTINUATION_RESULTS.md` and `continuation/collisions/COLLISION_DECISION.md`.
They supersede any interpretation that the single original run established a
converged collision energy budget. The production collision algorithm is unchanged.

## Status and proposed integration

[runaway_baseline.py](next_physics/runaway_baseline.py) is a standalone example proposal
using the frozen production collision and LL kernels. No production or repository
test file was modified. [validate_baseline.py](next_physics/validate_baseline.py) is its
local one-CPU reproducer. The [final component results](next_physics/validation_final_results.json)
record five component/integration gates, including the omitted-electric-emission
diagnostic, on JAX 0.12.0.1 CPU. They were rerun successfully on the frozen source
SHA256 `dae411be867fb834f12fe7ab4c2ced0486f11eeb6dcab842b6eb89ae5555c2af`.
The reproducer SHA256 is
`92e47f06e56332144a8065b9741e75e3fda3be109ab14636a6a6af23d2aee316`.
The unmodified [default example also completed](next_physics/default_results.json)
on that same source. Execution metadata is retained for the
[component checks](next_physics/validation_final_execution.json) and
[default run](next_physics/default_execution.json).

This advances the no-RF baseline. It does not certify a long-time dilute-tail
distribution, suppression, an RF response closure, or minimum source power.
The default 100 microsecond, 1088-marker case was run once and still needs
statistical and timestep convergence. Its raw collision-budget error is material
compared with the seed's physical energy changes. Keep this as a reference prototype
until those gates pass, rather than promoting it as a validated physical control
example in the PR.

A separate example is justified here: the existing `runaway_rf.py` resolves a
coherent RF phase through `Simulation`, whereas this driver removes gyrophase
from the modeled distribution. Hiding this change behind a timestep or solver
option in that RF example would invite an invalid unresolved-resonance calculation.
It needs one example file, no new production module or dependency. Keep local
validation output outside Git. The eventual physical control example should
absorb/reuse this driver rather than proliferating alternative frameworks.

## Equations and closure

For a homogeneous distribution with axial symmetry about a uniform B, the
magnetic Lorentz term is proportional to the gyrophase derivative and vanishes.
The driver evolves a three-component momentum ensemble for the remaining kinetic
equation, with u=p/m in m/s:

\[
\partial_t f_s+\nabla_u\cdot[(q_s E_\parallel\hat b/m_s+a_{\mathrm{LL},B})f_s]
=\sum_r C_{sr}[f_s,f_r].
\]

Three momentum components allow the existing rotationally invariant binary
collision kernel to represent the gyrotropic distribution. Individual marker
gyrophases are sampling coordinates, not predicted full-orbit phases. This
reduction requires a homogeneous gyrotropic initial distribution and E parallel
to B; a coherent RF field or nonuniform orbit physics cannot simply be inserted.

The independent deterministic reference uses eta=|u|/c, xi=u_parallel/|u|,
gamma=sqrt(1+eta^2), a_E=q E_parallel/(mc),
a_B=tau(qB/m)^2, tau=q^2/(6*pi*epsilon_0*m*c^3):

\[
\dot\eta=a_E\xi-a_B\gamma\eta(1-\xi^2),\qquad
\dot\xi=a_E(1-\xi^2)/\eta+a_B\xi(1-\xi^2)/\gamma.
\]

These uniform-plasma electric/synchrotron drifts are equations 2.8–2.12 of
[Hirvijoki et al. (2015)](https://doi.org/10.1017/S0022377815000513).
Their continuum collision reduction differs from this prototype's nonlinear
binary kernel; that paper is not an independent validation of the entire driver.
[CODE's original full paper](https://arxiv.org/pdf/1305.3518) is a reference for a
future matched kinetic benchmark, not evidence that this driver reproduces CODE.
Its equations 1–8 use a test-particle collision background; synchrotron emission
is calculated from the distribution afterward. Match only shared operators, or
use a verified radiation-inclusive CODE variant, rather than treating that paper
as an oracle for this evolving-bulk, radiation-reaction calculation.

The **total uniform electric field** is held at its prescribed value by an ideal
external current J_ext=-J_plasma. With homogeneous Maxwell fields this gives
epsilon_0*dE/dt=-(J_plasma+J_ext)=0. Its supplied power density is J_plasma dot E.
No independently evolved mean Ampere mode can screen the intended drive. This
defines an ideal local current source, not a tokamak circuit or antenna model.

All thermal electrons, seed electrons and deuterons evolve kinetically. There is
no bath reset, fixed-temperature reservoir or removed collision heat. The initial
thermal distributions use the Maxwell–Juttner d^3p radial measure, with antithetic
markers to remove a spurious initial bulk current. Subsequent sampled bulk-current
fluctuations remain and need ensemble convergence. Initial-seed labels identify
cohorts; the high-energy electron density includes both electron cohorts.
This is a host NumPy quadrature initializer for the example, not a new public
thermal loader. Temperature is not traced through that initializer. Field/density
derivatives verified below do not establish an initial-temperature derivative.

The production LL kernel is evaluated with E=0 for magnetic radiation. Electric
radiation requires an **absolute** omission budget. For the uniform parallel fields,
the missing quadratic power on a given momentum is exactly

\[
\Delta P_E=\tau(q/m)^2m E^2(1+\eta_\perp^2),\qquad
P_B=\tau(q/m)^2mc^2B^2\eta_\perp^2.
\]

Although |E|/(cB)=8.34e-12 here controls the algebraic force ordering, it does not
bound relative emission for nearly parallel or resting particles: P_B vanishes
while Delta P_E remains positive. The updated driver separately quadratures
Delta P_E over the actual computed ensemble. Compare that absolute energy and its
quadrature/trajectory uncertainty with the selected target error budget, not
only its ratio to magnetic emission. This is the missing power evaluated along
the approximate trajectory, not a rigorous bound on an arbitrarily changed
trajectory. The DC-only B=0 control validates momentum/work of the declared
magnetic-RR model and the omitted-power diagnostic; it does not claim zero
physical electric-field emission.
This additional emission diagnostic is not subtracted as mechanical work; general
electric-field radiation requires the consistent reaction and near-field budget.

Unresolved photons escape; radiative reabsorption is not included. Collisions
retain the existing fully ionized, weak-coupling, leading-log approximation with
an explicit Coulomb logarithm. No additional primary, hard Moller/avalanche,
bremsstrahlung or screening model has been smuggled into these kernels.

## Units, numerical method and independent budgets

Marker weights are number densities in m^-3. Multiplication by a 1 m reference
cell length converts them to the collision kernel's per-area weights; division
by that same cell length in the pairing rate recovers the prescribed physical
density. The reference length does not represent an evolving spatial mesh.
Kinetic energies and all accumulated work are J/m^3; multiply by a represented
physical length before comparison with a planar PIC J/m^2 budget.

The composition is half deterministic drift, full collision step, half drift.
The drift uses midpoint RK2. Its exact discrete energy gradient is
v_bar=(u_new+u_old)/(gamma_new+gamma_old), so DC and mechanical RR work sum to its
kinetic-energy increment. Emission is quadrature of the independently computed
positive LL magnetic power. The finite-step Nanbu collision model and species-pair
composition still need timestep convergence; symmetric outer splitting does not
establish second order for the complete stochastic operator.

The physical closed-system residual is K-K0-W_DC-W_RR. The raw collision defect
is recorded separately. Subtracting it to check bookkeeping is not a conservation
test. Equal weights should make it roundoff-small. With unequal weights it is a
random numerical error, not work by a physical heat bath or external source.

For the default density fraction 1e-5, 512 bulk and 64 seed markers have a weight
ratio about 12500. A rare accepted update of a heavy marker can make the raw
collision defect or its variance exceed the seed's physical energy transfer, and
in a long run even the seed-energy scale. Small bookkeeping residuals cannot
certify that calculation. The follow-on must establish a particle/seed/weight
budget against the control benefit; never silently rescale momenta to force closure.
At 256 steps the expected number of accepted heavy thermal-electron updates from
seed/thermal pairings is only
256*64*(w_seed/w_bulk)=1.31. This is a count derived from the actual acceptance
algorithm, not a convergence estimate. Increasing only seed markers does not
increase this count at fixed physical fraction and bulk-marker count. A conservative
distribution solver with a validated evolving thermal closure may be much more
efficient for the eventual control calculation, with this fully kinetic driver
retained as an independent small-system reference.

The gyro period and plasma period are absent from this homogeneous model, not
underresolved by it. Steps must instead resolve electric drift, magnetic cooling,
collisional relaxation and the finite-sample pairing rates. Strong cooling still
needs dt refinement; the explicit drift is not an unconditionally stable solver.

## Completed frozen-source component controls

The final one-CPU reproducer completed in 44.66 seconds, including imports and JIT.
All five gates passed with process exit zero on the frozen source above:

| Control | Result |
|---|---:|
| Exact magnetic gamma error, 32 / 64 / 128 steps | 2.162e-7 / 5.402e-8 / 1.350e-8 |
| Last magnetic error ratio | 4.0008 |
| Magnetic emission relative error, 128 steps | -1.129e-8 |
| Magnetic d gamma_final/dB, AD / analytic | -1.1583419149 / -1.1583419541 per T |
| Exact DC momentum / work relative error | 2.84e-15 / 1.47e-14 |
| DC-only omitted-electric-emission diagnostic / analytic relative error | 2.22e-16 |
| Independent p–pitch ODE relative error, 32 / 64 / 128 steps | 2.264e-5 / 5.624e-6 / 1.402e-6 |
| Last independent ODE error ratio | 4.0125 |
| Deterministic AD / central-difference relative difference | -3.41e-11 |
| Raw equal-weight collision energy / momentum error | 2.22e-16 / 3.92e-17 |
| Cold-target density derivative, sampled / analytic | -0.0048290 / -0.0048892 |
| Derivative sampling standard error | 0.00010084 |
| Common-seed derivative / finite-difference relative difference | 1.04e-10 |

The cold-target observable is the mean normalized parallel momentum u_parallel/c.
Its reported derivative is with respect to a dimensionless density multiplier
at scale=1, with each population's density equal to scale*1e20 m^-3. Divide that
derivative by 1e20 m^-3 to obtain the derivative with respect to SI number density.
The cold-target rate uses an independent invariant formula and the analytic first
Nanbu angular moment. Its mean and derivative agree within the predeclared five
standard-error gates. These fixed, parameter-independent pairing/acceptance cases
do not prove unbiased derivatives of all event-dependent stochastic expectations.

## Completed default run and its statistical limit

The unmodified default uses 512 thermal electrons, 64 seed electrons and 512
deuterons, 256 steps over 1e-4 s (dt=3.90625e-7 s), and collision seed zero. Thermal
initialization uses the documented fixed antithetic samples. This is one
realization; no independent particle/seed/step refinement or held-out kinetic
comparison was performed for it.

The following quantities are in J/m^3, with columns identifying initial cohorts:

| Quantity | Thermal electrons | Initial seed | Deuterons |
|---|---:|---:|---:|
| Initial kinetic energy | 24.0155484 | 31.9297125 | 24.0099180 |
| Final kinetic energy | 24.0764458 | 31.9289136 | 23.9696955 |
| DC work | 0.0223897246 | 0.0002388859 | 0.0000060888 |
| Mechanical magnetic RR work | -0.0024281535 | -0.0010115392 | -5.192e-14 |
| Magnetic emission | 0.0024281535 | 0.0010115392 | 5.192e-14 |
| Realized collision energy change | 0.0409358390 | -0.0000262307 | -0.0402285511 |
| Omitted electric emission along the trajectory | 4.41625e-22 | 7.47577e-26 | 3.27676e-29 |

The raw total collision defect is **6.81057e-4 J/m^3**. It is about **26 times the
seed's collision-energy transfer**, and **85% of the seed's net kinetic-energy
change**. These comparisons assess the total physical budget; they are not direct
error bounds on the individual seed trajectories. The bookkeeping residual is
only 2.19e-14 J/m^3, demonstrating that bookkeeping closure alone does not resolve
the statistical conservation problem. Normalizing instead to the initial seed
energy gives 2.13e-5 and would conceal its importance for these small transfers.

The actual default bulk/seed weight ratio is 12499.875. The expected number of
accepted heavy thermal-electron updates from seed/bulk pairings is 1.310733;
individual acceptance counts were not instrumented. This expectation is not a
convergence result or a claim that this pairing channel accounts for all the
observed error. The raw total collision momentum defect is approximately
(9.892e-10, -4.458e-10, -1.545e-9) kg/(m^2 s).

The omitted electric-emission diagnostic totals 4.41700e-22 J/m^3, versus
0.00343969 J/m^3 of magnetic emission. It is negligible in this specific absolute
budget; it does not justify a universal relative-emission bound near zero pitch.
The electron density above 10 MeV remains 1e13 m^-3. No suppression, long-time
prediction or minimum-power result is claimed.

## Next gates before control

1. Run a particle/step/seed refinement matrix around the default, reporting raw
   collision defects relative to seed energy and to each relevant physical energy
   transfer. Resolve rare heavy-marker events; a handful of seeds is insufficient
   when the variance estimate has not stabilized. Decide whether a conservative
   momentum-space distribution solver is more efficient for the physical tail.
   A concrete local pilot is bulk/ion counts 512, 2048 and 8192 at 64 seed markers,
   then 64 versus 256 seed markers at a selected bulk count, followed by 256, 512
   and 1024 steps at fixed 100 microseconds. Keep all physical densities and the
   seed fraction fixed. Start with 64 independent collision seeds in bounded
   batches, then increase them until uncertainty and rare-event variance stabilize;
   64 is a pilot size, not a pass criterion. Choose tolerances before inspecting
   results, smaller than the intended control benefit. Different timesteps do not
   have coupled stochastic paths here, so compare statistical distributions and
   uncertainties rather than demanding trajectory-by-trajectory agreement.
2. Match the no-RF model to independent kinetics under the same collision,
   radiation, source and bulk assumptions. Verify distribution moments and tail
   thresholds, not only the algebraic energy ledger. Retain the current source
   during any subsequent hold.
3. Bound omitted source, screening, bremsstrahlung and photon-reabsorption effects
   over the actual control/hold interval. Include needed terms and reservoir
   transfers without double counting primary generation already resolved by a
   sufficiently sampled kinetic bulk.
4. Add only a separately validated slow RF response with coupled pitch/energy
   transfer and a local source-work or mode-energy calibration. Verify coherent
   trapping and held-out distributions/phases. This needs no equilibrium or
   antenna-accessibility study. Then optimize a fixed-horizon supplied-energy
   objective with a DC-driven, source-active RF-off suppression hold.

## Reproduce locally

From a checkout containing the frozen production extension, set PYTHONPATH to
that checkout and run `python /absolute/path/validate_baseline.py`. Its default
output is adjacent to the external reproducer, never inside Git. On a local
machine the explicit CPU-affinity call can be adapted if sched_setaffinity is
unavailable. For the example itself:

```bash
JAX_ENABLE_X64=1 MPLBACKEND=Agg python /absolute/path/runaway_baseline.py
```

No source or test mutation is needed. Both final runs used one CPU (affinity 8)
and a shared-memory guard that would stop only the probe at 7.4 GiB total cgroup
use. No guard triggered. The component worker's individual RSS is **unavailable**:
its initial monitor used a PID from a different namespace than the mounted /proc.
Do not treat its discarded RSS number as a measurement. Its valid shared-cgroup
peak was 6.748 GiB, including unrelated workloads.

The default worker used its own /proc/self PID to fix that monitor. It completed
in 16.28 seconds, with peak child RSS 0.772 GiB and shared-cgroup peak 6.633 GiB;
the additional child-RSS guard was 1.3 GiB. These elapsed times include imports,
compilation, execution and plotting, and are not warm-kernel benchmarks. The
workers ran sequentially.
