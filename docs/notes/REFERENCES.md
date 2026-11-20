> Original research specification, preserved for equations, sources and remaining work. Some API proposals and implementation-status statements are historical. Read `../START_HERE.md` first; the bundled branch and its documentation define the implemented API. Historical `evidence/` and `probes/` paths refer to the earlier full research archive, not to new production validation. Current reproduction is in `../VALIDATION.md`.

# Primary references and source map

Prepared 4 November 2026. The documents distinguish published results, direct algebraic derivations, proposed design choices and measurements executed for this reference. URLs below are primary papers, author/institutional records, official source or official documentation. They are not a claim that every cited full text was downloaded or independently reproduced.

## 1. Code and numerical starting point

| Source | Use in this reference |
| --- | --- |
| [UWPlasma/JAX-in-Cell PR 42](https://github.com/uwplasma/JAX-in-Cell/pull/42), head `ac605d4f20c6dc3ee46cd566d286c5a6c791a808` | Parent research API and intended stacked-PR base; refresh its status before implementation |
| [Pinned `_core.py`](https://github.com/uwplasma/JAX-in-Cell/blob/ac605d4f20c6dc3ee46cd566d286c5a6c791a808/jaxincell/_core.py) and [pinned `_simulation.py`](https://github.com/uwplasma/JAX-in-Cell/blob/ac605d4f20c6dc3ee46cd566d286c5a6c791a808/jaxincell/_simulation.py) | Actual relativistic pusher, explicit/implicit stepping, carried momentum, source/restart paths |
| [Pinned diagnostics](https://github.com/uwplasma/JAX-in-Cell/blob/ac605d4f20c6dc3ee46cd566d286c5a6c791a808/jaxincell/_diagnostics.py) and [gradient tests](https://github.com/uwplasma/JAX-in-Cell/blob/ac605d4f20c6dc3ee46cd566d286c5a6c791a808/tests/test_gradients.py) | High-gamma diagnostic audit and distinction between discrete, expected and continuum sensitivities |
| [Pinned project configuration](https://github.com/uwplasma/JAX-in-Cell/blob/ac605d4f20c6dc3ee46cd566d286c5a6c791a808/pyproject.toml) | Existing dependencies and 100% line/branch coverage gate |

The code evidence is a verified 60-file source subset. The complete source-provenance JSON is included; the full repository is not vendored. The accompanying source-reuse report pins SOLVAX at `59b8c317357fd78582f3e06be6202972445eac68` and ESSOS at `ef95d92ad599282a40e358ce752db8faf3a7e555`. Those projects were inspected narrowly, not fully profiled or validated here.

## 2. Classical radiation and analytical benchmarks

| Reference | Required use |
| --- | --- |
| M. Vranic et al., *Classical radiation reaction in particle-in-cell simulations*, CPC 204, 141–151 (2016). [DOI](https://doi.org/10.1016/j.cpc.2016.04.002), [author preprint](https://arxiv.org/abs/1502.02432) | Distinguish full LL, derivative-free LL and high-energy drag; PIC implementation comparisons |
| A. Di Piazza, *Exact solution of the Landau–Lifshitz equation in a plane wave*, Lett. Math. Phys. 83, 305–313 (2008). [DOI](https://doi.org/10.1007/s11005-008-0228-9), [preprint](https://arxiv.org/abs/0801.1751) | Exact full-LL plane-wave reference; compare like models and asymptotic observables |
| Y. Yaremko, constant-field LL solution. [DOI](https://doi.org/10.1063/1.4820131), [preprint](https://arxiv.org/abs/1412.1661) | Independent constant electromagnetic-field reference; own uniform-B derivation is also included |
| P. J. Bilbao and L. O. Silva, *Radiation Reaction Cooling as a Source of Anisotropic Momentum Distributions with Inverted Populations*, PRL 130, 165101 (2023). [DOI](https://doi.org/10.1103/PhysRevLett.130.165101) | Magnetized-plasma cooling, ring formation and inversion benchmark; validate its geometry and parameters before adding instability claims |
| [Smilei radiation-loss documentation](https://smileipic.github.io/Smilei/Understand/radiation_loss.html) | Maintained comparison of continuous and stochastic radiation models and diagnostics |

The cooling and gyrophase equations in this package are derived directly from the stated uniform-B LL equations. The measured probe uses the actual JAX-in-Cell Boris kernel. None of these references substitutes for validating the new coupled PIC energy/current update.

## 3. Relativistic collisions

| Reference | Decision or limit it informs |
| --- | --- |
| T. Takizuka and H. Abe, *A binary collision model for plasma simulation with a particle code*, JCP 25, 205–219 (1977). [DOI](https://doi.org/10.1016/0021-9991(77)90099-7) | Random binary-collision lineage and nonrelativistic overlap |
| K. Nanbu, *Theory of cumulative small-angle collisions in plasmas*, PRE 55, 4642–4652 (1997). [DOI](https://doi.org/10.1103/PhysRevE.55.4642) | Cumulative-angle reference and moment conventions |
| A. V. Bobylev and K. Nanbu, collision-algorithm theory, PRE 61, 4576–4586 (2000). [DOI](https://doi.org/10.1103/PhysRevE.61.4576) | Small-step convergence framework; no arbitrary finite-step exactness claim |
| Y. Sentoku and A. J. Kemp, *Numerical methods for particle simulations at extreme densities and temperatures: Weighted particles, relativistic collisions and reduced currents*, JCP 227, 6846–6861 (2008). [DOI](https://doi.org/10.1016/j.jcp.2008.03.043) | Earlier relativistic/weighted design; not an excuse to ignore later weighting corrections |
| F. Perez et al., *Improved modeling of relativistic collisions and collisional ionization in particle-in-cell codes*, Phys. Plasmas 19, 083104 (2012). [DOI](https://doi.org/10.1063/1.4742167) | Selected relativistic COM/rate lineage |
| [Official pinned WarpX implementation](https://github.com/BLAST-WarpX/warpx/blob/705c2deccfec137380cc940d633eb4c4665b97ed/Source/Particles/Collision/BinaryCollision/Coulomb/UpdateMomentumPerezElastic.H), blob `45a778e08863a77a3cb178a08e89225f48a6812b`; [rendered source](https://warpx.readthedocs.io/en/26.06/_static/doxyhtml/_update_momentum_perez_elastic_8_h_source.html) | Audited uncapped SI scattering strength, COM conventions and mapping to Perez Eq. 9, Eq. 12, Sec. II.C and Eq. 22 correction |
| [Official Smilei collision documentation](https://smileipic.github.io/Smilei/Understand/collisions.html) | Perez Coulomb-log typo, corrected arbitrary-weight implementation and current angular fit |
| D. P. Higginson, I. Holod and A. Link, *A corrected method for Coulomb scattering in arbitrarily weighted particle-in-cell plasma simulations*, JCP 413, 109450 (2020). [DOI](https://doi.org/10.1016/j.jcp.2020.109450), [institutional record](https://www.osti.gov/pages/biblio/1623345) | Correct unequal-weight rates; ordinary weighted scattering conserves in expectation |
| J. R. Angus et al., *Moment-preserving Monte-Carlo Coulomb collision method for particle codes*, JCP 531, 113927 (2025). [DOI](https://doi.org/10.1016/j.jcp.2025.113927), [author preprint](https://arxiv.org/abs/2407.19151) | Later weighted moment-restoration candidate, including relativistic energy restoration; not implemented in this reference |
| B. J. Braams and C. F. F. Karney, *Differential form of the collision integral for a relativistic plasma*, PRL 59, 1817 (1987). [DOI](https://doi.org/10.1103/PhysRevLett.59.1817) | Continuum relativistic collision reference |
| K. Sarkimaki, E. Hirvijoki and J. Terava, *Adaptive time-stepping Monte Carlo integration of Coulomb collisions*, CPC 222, 374–383 (2018). [DOI](https://doi.org/10.1016/j.cpc.2017.09.025), [preprint](https://arxiv.org/abs/1701.05043) | Beliaev–Budker test-particle dynamics in a Maxwell–Juttner bath; adaptive stochastic stepping requires correct noise treatment |
| J.-L. Vay et al., *Energy-preserving coupling of explicit particle-in-cell with Monte Carlo collisions*, PRE 111, 025306 (2025). [DOI](https://doi.org/10.1103/PhysRevE.111.025306) | Composition can heat despite isolated conservation; cited energy-balance analysis is nonrelativistic and must not be relabeled a proof for this solver |

**Retrieval limit:** the original Perez Eq. 9 PDF was not obtained. The rate specification is checked against the pinned official implementation and an independently derived invariant form/nonrelativistic limit. The reference does not claim visual inspection of that original equation. The full weighted Angus algorithm has not been ported or tested. These limits remain explicit in `COLLISIONS.md`.

## 4. Runaway physics and RF control

| Reference | Role |
| --- | --- |
| H. Choudhury et al., externally launched helicon scattering, PRL 136, 025101 (2026). [DOI](https://doi.org/10.1103/jz8f-ck22), [primary preprint](https://arxiv.org/html/2505.19279v1) | Existing experimental mechanism and approximate local parameter anchor; this reference does not claim its discovery |
| H. Choudhury et al., 2026 helicon follow-up. [Preprint](https://arxiv.org/abs/2606.22141), [PPCF DOI](https://doi.org/10.1088/1361-6587/aeb629), [institutional record](https://infoscience.epfl.ch/entities/publication/1e6a28b7-4eeb-407d-aa09-bb88e7fed23d) | Existing alignment, propagation and power investigations; those studies are outside this user's requested local scope |
| Y. Ghai et al., *Non-diffusive pitch-angle scattering of runaway electrons in the presence of whistler waves*, Phys. Plasmas 33, 102501 (2026). [DOI](https://doi.org/10.1063/5.0346141), [ORNL record](https://impact.ornl.gov/en/publications/non-diffusive-pitch-angle-scattering-of-runaway-electrons-in-the-/) | Energy-dependent departures from ordinary diffusion in AORSA–KORC calculations; validate a reduced RF closure instead of assuming it |
| C. Liu et al., *Role of Kinetic Instability in Runaway-Electron Avalanches and Elevated Critical Electric Fields*, PRL 120, 265001 (2018). [DOI](https://doi.org/10.1103/PhysRevLett.120.265001) | Wave/avalanche coupling is part of the existing literature |
| O. Embreus, A. Stahl and T. Fulop, *On the relativistic large-angle electron collision operator for runaway avalanches in plasmas*, JPP 84, 905840102 (2018). [DOI](https://doi.org/10.1017/S002237781700099X), [preprint](https://arxiv.org/abs/1708.08779) | Conservative hard collisions, primary recoil, reverse avalanche, and matched small/large-angle treatment |
| M. Landreman, A. Stahl and T. Fulop, *Numerical calculation of the runaway electron distribution function and associated synchrotron emission*, CPC 185, 847–855 (2014). [DOI](https://doi.org/10.1016/j.cpc.2013.12.004) | CODE as a matched local collision/electric/radiation benchmark |
| M. Hoppe, O. Embreus and T. Fulop, *DREAM: A fluid-kinetic framework for tokamak disruption runaway electron simulations*, CPC 268, 108098 (2021). [DOI](https://doi.org/10.1016/j.cpc.2021.108098), [official collision settings](https://ft.nephy.chalmers.se/dream/frontend/api/Settings/collisions.html) | Independent kinetic/reference terms and their full/superthermal/ultrarelativistic validity ranges |
| L. Hesslow et al., *Effect of Partially Screened Nuclei on Fast-Electron Dynamics*, PRL 118, 255001 (2017). [DOI](https://doi.org/10.1103/PhysRevLett.118.255001) | Charge state and bound-electron screening require explicit physics in impurity-rich cases |
| L. Hesslow et al., effective critical field with partially ionized impurities/radiation, PPCF 60, 074010 (2018). [DOI](https://doi.org/10.1088/1361-6587/aac33e) | Physical loss/source thresholds differ from the elementary fully ionized Connor–Hastie scale |

The proposed B=2 T, density `1e18 m^-3`, 20 MeV seeded scenario and its control target are our numerical design inputs, not a reconstruction of a discharge or an experimentally certified antenna operating point. Cold parallel dispersion is used only for a supplied homogeneous test mode. No broad novelty or global optimum claim has been established.

## 5. JAX differentiation, performance, and quantum follow-on

| Reference | Use |
| --- | --- |
| [JAX checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html) | Recompute selected forward intermediates to reduce reverse memory; the supplied CPU experiment tests this directly |
| [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and [profiling](https://docs.jax.dev/en/latest/profiling.html) | Synchronization, warm execution, compilation and memory measurement |
| [JAX Pallas](https://docs.jax.dev/en/latest/pallas/) | Optional custom accelerator kernels; check current backend and derivative support rather than assuming it |
| [JAX sharding and autodiff](https://docs.jax.dev/en/latest/301/sharding-ad.html), [shard_map](https://docs.jax.dev/en/latest/201/shard-map.html) | Independent ensemble first; PIC deposition and collision-partner communication need an explicit design |
| G. Arya et al., *Automatic Differentiation of Programs with Discrete Randomness*, NeurIPS (2022). [Paper](https://papers.nips.cc/paper_files/paper/2022/hash/43d8e5fc816c692f342493331d5e98fc-Abstract-Conference.html) | Discrete random choices need an estimator beyond naive branchwise AD; no claim that this method is already integrated with JAX-in-Cell |
| F. Niel et al., PRE 97, 043209 (2018). [DOI](https://doi.org/10.1103/PhysRevE.97.043209) | Continuous, diffusion and stochastic radiation descriptions and moment-level comparisons |
| A. Di Piazza et al., *Implementing nonlinear Compton scattering beyond the local-constant-field approximation*, PRA 98, 012134 (2018). [DOI](https://doi.org/10.1103/PhysRevA.98.012134), [preprint](https://arxiv.org/abs/1708.08276) | Photon-energy-dependent formation-length/LCFA limits, especially soft emission |
| A. Di Piazza et al., *Improved local-constant-field approximation for strong-field QED codes*, PRA 99, 022125 (2019). [DOI](https://doi.org/10.1103/PhysRevA.99.022125), [preprint](https://arxiv.org/abs/1811.05834) | Improved infrared emission treatment for general spacetime-dependent backgrounds; beyond-LCFA comparisons for emission and pair production |

The final local implementation should refresh software documentation and recent physics literature before deciding backend support or framing a new research claim. Physical equations and published comparisons must still be matched to the actual implemented model.
