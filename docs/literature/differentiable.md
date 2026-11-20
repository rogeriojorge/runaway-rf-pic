# Differentiable PIC and kinetic codes; gradients through Monte Carlo (L0 survey)

Status: [V] = retrieved this session via search, [K] memory.

## Existing differentiable plasma codes
- JAX-in-Cell (arXiv 2512.12160; docs `user_guide/differentiation.html`) [V]: whole time loop is a JAX function; reverse mode works only for the explicit integrator (implicit uses `lax.while_loop`, reverse fails); cell index and wall absorption are non-smooth; finite-particle output is a noisy function of inputs.
- Carvalho et al., "Learning collision operators from plasma phase space data using differentiable simulators", arXiv 2601.10885 / JPP 2026 [V]: differentiable Fokker-Planck solver in JAX fitted to PIC data; PIC itself has no collision module.
- Other (not re-verified): Julia and JAX differentiable PIC (e.g. Dyson-type, Vlasov solvers) exist [K]; no one found with differentiable *binary Coulomb collisions in a PIC loop*. Searches for "differentiable DSMC/MCC" returned nothing relevant; absence is a gap, not a proof.

## Gradient estimators for stochastic collisions
| Estimator | Applies | Bias | Variance | Note |
|---|---|---|---|---|
| Pathwise (reparameterization) | continuous noise: Nanbu angle `mu = F^-1(U; A(s))`, Niel `dW` | unbiased for continuous kernels | low | differentiate `A(s)` via implicit root derivative; fixed pairing permutation (non-differentiable index, treat as fixed) |
| Score function (REINFORCE) | discrete events: pair selection, photon emission, accept/reject (Higginson) | unbiased | high; needs baselines | Arya et al. NeurIPS 2022 (stochastic AD with discrete randomness) gives a better estimator than naive branchwise AD |
| Common random numbers + FD | any | O(h) | low per sample pair | cheapest validation oracle: compare AD vs CRN-FD of an expectation |
| Ensemble mean gradient | all | 0 | 1/N | average over keys; the target quantity is dE[J]/dtheta |
Recommendation: collisions (Nanbu angle) and Niel RR are pathwise-differentiable; keep pairing indices stop-gradient; Monte Carlo photon emission is excluded from AD paths (L5 mean-recoil g(chi) is deterministic and smooth). Handle `s -> 0` and `A -> 0` with analytic limits before differentiating.

## Memory and long horizons
- `jax.checkpoint` / `jax.remat` per step or per chunk with `lax.scan`; binomial (Revolve) checkpointing is not native to JAX, chunked remat gives sqrt(T) memory [K]. JAX docs: docs.jax.dev gradient-checkpointing.
- Gradient horizon: chaotic particle dynamics make long-horizon gradients explode; reference results (JIC sheath program) already show horizon limits. Use windowed (truncated) gradients and report the AD/FD ratio versus horizon.
- Implicit solves: use `custom_vjp` with the implicit function theorem (or `jax.lax.custom_root`) instead of differentiating `while_loop`.
- Sharding: ensembles over keys first (embarrassingly parallel); deposition needs `psum`.

## Tests
1. AD vs CRN-FD of `E[T_perp]` after a collision relaxation step w.r.t. `lnL` (should agree within sampling error).
2. Gradient of the exact `coth` cooling solution w.r.t. B against the analytic `d gamma/d B`.
3. Quantify noise floor: variance of per-sample gradient vs N particles.
