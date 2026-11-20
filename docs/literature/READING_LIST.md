# Reading list: papers to download as PDF (2026-11-10)

Priority: MUST = needed to write or defend a PRL on the theses in PRL_SURVEY.md; SHOULD = needed for methods, comparison or related work; OPT = background or only if the topic is pursued.
OA = open access (publisher or arXiv free full text). "arXiv-OA" means the arXiv version is free, journal version paywalled. Verification: [V] metadata confirmed in a page retrieved this session; [A] abstract-level confirmation only; [U] from memory, not verified, check DOI before citing. Books and reports without DOI are marked "none".
Naming hint for the PDF folder: `<FirstAuthor><Year>_<keyword>.pdf`.

## A. Runaway electrons: foundations and kinetic theory

1. MUST. J. W. Connor and R. J. Hastie, "Relativistic limitations on runaway electrons," Nucl. Fusion 15, 415 (1975). DOI 10.1088/0029-5515/15/3/007. [U] Defines E_c; baseline for every E/E_c in the program. Not OA.
2. SHOULD. H. Dreicer, "Electron and ion runaway in a fully ionized gas I," Phys. Rev. 115, 238 (1959). DOI 10.1103/PhysRev.115.238. [U] Primary generation; E_D. Not OA. (Part II: Phys. Rev. 117, 329 (1960), DOI 10.1103/PhysRev.117.329 [U].)
3. MUST. M. N. Rosenbluth and S. V. Putvinski, "Theory for avalanche of runaway electrons in tokamaks," Nucl. Fusion 37, 1355 (1997). DOI 10.1088/0029-5515/37/10/I03. [U] Avalanche growth rate; verify the factors flagged in earlier L0 notes. Not OA.
4. MUST. P. Aleynikov and B. N. Breizman, "Theory of two threshold fields for relativistic runaway electrons," Phys. Rev. Lett. 114, 155001 (2015). DOI 10.1103/PhysRevLett.114.155001. [V] Synchrotron-limited energy and thresholds; checks the "no bifurcation" claim in the reduced ODE. arXiv-OA (accepted manuscript on APS).
5. MUST. M. Landreman, A. Stahl, T. Fulop, "Numerical calculation of the runaway electron distribution function and associated synchrotron emission," Comput. Phys. Commun. 185, 847 (2014). DOI 10.1016/j.cpc.2013.12.004; arXiv 1305.3518. [V] CODE; comparison target. OA (arXiv).
6. MUST. M. Hoppe, O. Embreus, T. Fulop, "DREAM: A fluid-kinetic framework for tokamak disruption runaway electron simulations," Comput. Phys. Commun. 268, 108098 (2021). DOI 10.1016/j.cpc.2021.108098; arXiv 2103.16457 (likely). [V title, DOI by CPC pattern: confirm] DREAM; the primary code to compare against. OA (arXiv).
7. MUST. L. Hesslow, O. Embreus, M. Hoppe, T. DuBois, G. Papp, M. Rahm, T. Fulop, "Generalized collision operator for fast electrons interacting with partially ionized impurities," J. Plasma Phys. 84, 905840605 (2018). DOI [U]. [A] Collision operator DREAM uses. Journal not OA; arXiv version exists [U id].
8. MUST. L. Hesslow, O. Embreus, A. Stahl, T. DuBois, G. Papp, S. Newton, T. Fulop, "Effect of partially screened nuclei on fast-electron dynamics," Phys. Rev. Lett. 118, 255001 (2017). DOI 10.1103/PhysRevLett.118.255001. [U] Needed if impurities enter; shows RE physics PRL bar. Not OA.
9. SHOULD. L. Hesslow, O. Embreus, G. J. Wilkie, G. Papp, T. Fulop, "Effect of partially ionized impurities and radiation on the effective critical electric field for runaway generation," Plasma Phys. Control. Fusion 60, 074010 (2018). DOI 10.1088/1361-6587/aac33e; arXiv 1802.00717. [V] Effective E_c with synchrotron and bremsstrahlung. OA (arXiv).
10. MUST. L. Hesslow, O. Embreus, M. Vallhagen, T. Fulop, "Influence of massive material injection on avalanche runaway generation during tokamak disruptions," Nucl. Fusion 59, 084004 (2019). DOI 10.1088/1741-4326/ab26c2; arXiv 1904.00602. [V] Reference avalanche growth formula (fluid). OA (arXiv).
11. MUST. O. Embreus, A. Stahl, T. Fulop, "On the relativistic large-angle electron collision operator for runaway avalanches in plasmas," J. Plasma Phys. 84, 905840102 (2018). DOI 10.1017/S002237781700099X. [V] Knock-on source; the avalanche to include in the PIC collision module. Not OA (check arXiv).
12. SHOULD. A. Stahl, E. Hirvijoki, J. Decker, O. Embreus, T. Fulop, "Effective critical electric field for runaway-electron generation," Phys. Rev. Lett. 114, 115002 (2015). DOI 10.1103/PhysRevLett.114.115002. [U] Kinetic E_c above Connor-Hastie. Not OA.
13. SHOULD. A. Stahl, O. Embreus, G. Papp, M. Landreman, T. Fulop, "Kinetic modelling of runaway electrons in dynamic scenarios," Nucl. Fusion 56, 112009 (2016). DOI 10.1088/0029-5515/56/11/112009. [U] CODE/NORSE-era dynamics. Not OA.
14. SHOULD. A. Stahl, M. Landreman, O. Embreus, T. Fulop, "NORSE: a solver for the relativistic non-linear Fokker-Planck equation for electrons in a homogeneous plasma," Comput. Phys. Commun. 212, 269 (2017). DOI [U]; arXiv 1608.02742. [V arXiv; authors per memory] Nonlinear relativistic collision solver; independent check of the PIC bulk heating. OA (arXiv).
15. SHOULD. J. Decker and Y. Peysson, "DKE: a fast numerical solver for the 3D relativistic bounce-averaged electron drift kinetic equation," report EUR-CEA-FC-1736 (2004). DOI none. [U] LUKE reference.
16. MUST. E. Hirvijoki, I. Pusztai, J. Decker, O. Embreus, A. Stahl, T. Fulop, "Radiation reaction induced non-monotonic features in runaway electron distributions," J. Plasma Phys. 81, 475810502 (2015). DOI 10.1017/S0022377815000513; arXiv 1502.03333. [V] Bump from RR plus collisions; prior art for Thesis C. OA (arXiv).
17. SHOULD. J. Decker, E. Hirvijoki, O. Embreus, Y. Peysson, A. Stahl, I. Pusztai, T. Fulop, "Numerical characterization of bump formation in the runaway electron tail," Plasma Phys. Control. Fusion 58, 025016 (2016). DOI 10.1088/0741-3335/58/2/025016 [U]; arXiv 1503.03881 [V title "Bump formation in the runaway electron tail"]. Companion to 16. OA (arXiv).
18. SHOULD. E. Hirvijoki, O. Embreus, A. Stahl, T. Fulop, "Effect of bremsstrahlung radiation emission on distributions of runaway electrons in magnetized plasmas," J. Plasma Phys. 82 (2016). DOI [U]; arXiv 1511.03917. [V arXiv] Bremsstrahlung loss neglected in the L2 notes; check size. OA (arXiv).

## B. RF, whistler and helicon interaction with runaways

24. MUST. T. Fulop, G. Pokol, P. Helander, M. Lisak, "Destabilization of magnetosonic-whistler waves by a relativistic runaway beam," Phys. Plasmas 13, 062506 (2006). DOI 10.1063/1.2208327. [V] Required by the task; foundation for whistler-runaway resonance. Not OA.
25. SHOULD. G. Pokol, T. Fulop, M. Lisak, "Quasi-linear analysis of whistler waves in tokamak plasmas," Plasma Phys. Control. Fusion 50, 045003 (2008). DOI 10.1088/0741-3335/50/4/045003. [U] QL framework. Not OA.
26. SHOULD. A. Komar, G. Pokol, T. Fulop, "Electromagnetic waves destabilized by runaway electrons in near-critical electric fields," Phys. Plasmas 20, 012117 (2013). DOI 10.1063/1.4789859 [U]; arXiv 1301.1250 [V]. Near-critical regime of the program's study case. OA (arXiv). Related EXEL follow-up (Phys. Plasmas 21, 102503 (2014) [A]) is OPT.
27. MUST. C. Liu, E. Hirvijoki, G.-Y. Fu et al., "Role of kinetic instability in runaway-electron avalanches and elevated critical electric fields," Phys. Rev. Lett. 120, 265001 (2018). DOI 10.1103/PhysRevLett.120.265001; arXiv 1801.01827. [V] Self-consistent whistler instability plus avalanche. OA (arXiv).
28. SHOULD. C. Liu, D. P. Brennan, A. Bhattacharjee, A. H. Boozer et al., "The effects of kinetic instabilities on the electron cyclotron emission from runaway electrons," arXiv 1803.09897 (2018). [V arXiv; author list U] ECE vs pitch; Thesis C diagnostics. OA.
29. MUST. D. A. Spong, W. W. Heidbrink, C. Paz-Soldan et al., "First direct observation of runaway-electron-driven whistler waves in tokamaks," Phys. Rev. Lett. 120, 155002 (2018). DOI 10.1103/PhysRevLett.120.155002. [V] The DIII-D whistler observation (this is the likely "DIII-D whistler" paper the task refers to). Accepted manuscript free via OSTI 1494016.
30. MUST. H. Choudhury, A. Battey, C. Paz-Soldan et al., "First demonstration of resonant pitch-angle scattering of relativistic electrons by externally launched helicon waves," arXiv 2505.19279; published Phys. Rev. Lett. 136, 025101 (2026) [volume/page from the follow-up preprint; DOI U]. [V arXiv] The key experimental result. OA (arXiv).
31. MUST. H. Choudhury, J. Lestz, C. Paz-Soldan et al., "Resonant pitch-angle scattering of runaway electrons by externally launched helicon waves in the DIII-D tokamak," arXiv 2606.22141 (2026). [V abstract] Power threshold, n=-1 anomalous waves growing the population, collisional damping. OA.
32. MUST. Y. Ghai, D. del-Castillo-Negrete, D. A. Spong, M. T. Beidler, "Runaway electron interactions with whistler waves in tokamak plasmas: energy-dependent transport scaling," arXiv 2509.13271 (2025); published Phys. Plasmas 33, 102501 (2026) per earlier L0 notes [U]. [V arXiv] Non-diffusive scattering in KORC-AORSA; the claim Thesis A tests. OA (arXiv).
33. SHOULD. H. Choudhury et al., "Firewall effect on electron acceleration by external whistler waves," APS DPP 2025 abstract UO09.1 (meetings-archive.aps.org/dpp/2025/uo09/1). [A] DOI none. Related DIII-D framing; read abstract and ask authors for the preprint.
34. SHOULD. W. W. Heidbrink et al., "The effects of radio-frequency wave heating on quiescent runaway-electron plasmas in DIII-D," APS DPP 2023 abstract BP11.97. [A] DOI none. Earlier null result for helicon on QRE; context.
35. SHOULD. C. F. F. Karney and N. J. Fisch, "Numerical studies of current generation by radio-frequency traveling waves," Phys. Fluids 22, 1817 (1979). DOI 10.1063/1.862804. [U] Relativistic QL RF-FP method with collisions; precedent for Thesis A numerics. Not OA.
36. SHOULD. C. F. F. Karney, "Stochastic ion heating by a lower hybrid wave," Phys. Fluids 21, 1584 (1978). DOI 10.1063/1.862394. [U] Resonance overlap and stochastic threshold, the classical analog. Not OA.
37. SHOULD. N. J. Fisch, "Theory of current drive in plasmas," Rev. Mod. Phys. 59, 175 (1987). DOI 10.1103/RevModPhys.59.175. [U] RF-FP formalism, efficiency and power accounting; relevant to any power-cost claim. Not OA.
38. OPT. Y. Zhang et al., "Generation and suppression of runaway electrons in EAST lower hybrid current drive experiments," Phys. Scr. 84, 015501 (2011) (ADS 2011PhyS...84a5501Z). DOI [U]. [A] EAST data point requested; E-field reduction, not resonant scattering. OPT.
39. OPT. Z. Zhou and C. Liu, "DREAM quasilinear diffusion for helicon waves," slides, ft.nephy.chalmers.se/do/downloadconfpresentation.php?id=232. [A] DOI none. Only public DREAM-with-wave work found; ask authors for a paper.
40. OPT. Y. Lee, K. Park, T. Park, G. Yun, Y.-S. Na, J.-K. Park, "Non-thermal electron cyclotron emission during runaway plateau in tokamak disruptions from a highly anisotropic dielectric tensor," arXiv 2604.26034 (2026). [V abstract] ECE modeling; Thesis C. OA.
41. OPT. P. Aleynikov et al., EC-wave runaway control in post-disruption plasmas, IAEA FEC 2024 paper 13363 (conferences.iaea.org/event/392/papers/36133). [A] DOI none. n_parallel and mode conversion constraint; check authorship.

## C. Radiation reaction: theory, reviews, algorithms

42. MUST. L. D. Landau and E. M. Lifshitz, The Classical Theory of Fields, 4th ed. (Pergamon, 1975), section 76. DOI none. Defines LL force; the LL, reduced-LL and invariant-drag equations in the code. Not OA.
43. MUST. A. Di Piazza, "Exact solution of the Landau-Lifshitz equation in a plane wave," Lett. Math. Phys. 83, 305 (2008). DOI 10.1007/s11005-008-0228-9. [V in L0 notes] Analytic check for full LL through a pulse. Not OA.
44. MUST. M. Vranic, J. L. Martins, R. A. Fonseca, L. O. Silva, "Classical radiation reaction in particle-in-cell simulations," Comput. Phys. Commun. 204, 141 (2016). DOI 10.1016/j.cpc.2016.04.002; arXiv 1502.02432 [U id]. [A] Reduced LL vs full LL to O(1/gamma); pusher recipe. arXiv-OA.
45. MUST. F. Niel, C. Riconda, F. Amiranoff, R. Duclous, M. Grech, "From quantum to classical modeling of radiation reaction: a focus on stochasticity effects," Phys. Rev. E 97, 043209 (2018). DOI 10.1103/PhysRevE.97.043209. [A] g(chi), h(chi) fits; Fokker-Planck RR. arXiv-OA.
46. MUST. A. Di Piazza, C. Muller, K. Z. Hatsagortsyan, C. H. Keitel, "Extremely high-intensity laser interactions with fundamental quantum systems," Rev. Mod. Phys. 84, 1177 (2012). DOI 10.1103/RevModPhys.84.1177; arXiv 1111.3886. [V] Classic review (the "Piazza" review in the task). arXiv-OA.
47. MUST. A. Gonoskov, T. G. Blackburn, M. Marklund, S. S. Bulanov, "Charged particle motion and radiation in strong electromagnetic fields," Rev. Mod. Phys. 94, 045001 (2022). DOI 10.1103/RevModPhys.94.045001; arXiv 2107.02161. [V] Task-required review. arXiv-OA.
48. MUST. T. G. Blackburn, "Radiation reaction in electron-beam interactions with high-intensity lasers," Rev. Mod. Plasma Phys. 4, 5 (2020). DOI 10.1007/s41614-020-0042-0; arXiv 1910.13377. [V] Task-required review. OA (Springer open access, arXiv).
49. SHOULD. M. Tamburini, F. Pegoraro, A. Di Piazza, C. H. Keitel, A. Macchi, "Radiation reaction effects on radiation pressure acceleration," New J. Phys. 12, 123005 (2010). DOI 10.1088/1367-2630/12/12/123005. [U] Boris plus RR substep. OA.
50. SHOULD. C. P. Ridgers, J. G. Kirk, R. Duclous et al., "Modelling gamma-ray photon emission and pair production in high-intensity laser-matter interactions," J. Comput. Phys. 260, 273 (2014). DOI 10.1016/j.jcp.2013.12.007. [U] MC emission, h(chi) fit. Not OA.
51. SHOULD. A. Di Piazza, M. Tamburini, S. Meuren, C. H. Keitel, "Implementing nonlinear Compton scattering beyond the local constant field approximation," Phys. Rev. A 98, 012134 (2018). DOI 10.1103/PhysRevA.98.012134; and "Improved local-constant-field approximation for strong-field QED codes," Phys. Rev. A 99, 022125 (2019), DOI 10.1103/PhysRevA.99.022125. [DOIs from L0 notes; titles U] LCFA limits for the chi monitor. arXiv-OA.
52. OPT. I. V. Sokolov, N. M. Naumova, J. A. Nees, G. A. Mourou, "Pair creation in QED-strong pulsed laser fields interacting with electron beams," Phys. Rev. Lett. 105, 195005 (2010). DOI 10.1103/PhysRevLett.105.195005. [U] Task-required Sokolov reference; the exact intended Sokolov paper may differ (his radiation-friction simulations are Phys. Plasmas 18, 093109 (2011), DOI 10.1063/1.3626001 [U]).
53. OPT. A. Gonoskov, S. Bastrakov, E. Efimenko et al., "Extended particle-in-cell schemes for physics in ultrastrong laser fields," Phys. Rev. E 92, 023305 (2015). DOI 10.1103/PhysRevE.92.023305. [U] PIC radiation algorithms (PICADOR).
54. OPT. R. Ripperda, M. Bacchini, O. Porth et al., "A comprehensive comparison of relativistic particle integrators," Astrophys. J. Suppl. 235, 21 (2018). DOI 10.3847/1538-4365/aab114. [U] Integrator choice; relevant for implicit RR pushers.
55. OPT. P. Petri, "Radiation reaction in constant fields, exact reduced-LL step," arXiv 2207.00624 (2022). [A, title U] Exact step for constant fields; test target for the reduced-LL pusher. OA.

## D. Radiation reaction in plasma and experiments

56. MUST. P. J. Bilbao and L. O. Silva, "Radiation reaction cooling as a source of anisotropic momentum distributions with inverted populations," Phys. Rev. Lett. 130, 165101 (2023). DOI 10.1103/PhysRevLett.130.165101; arXiv 2212.12271. [V] Ring formation; already benchmarked in L3. arXiv-OA.
57. MUST. P. J. Bilbao, R. J. Ewart, F. Assuncao, T. Silva, L. O. Silva, "Ring momentum distributions as a general feature of Vlasov dynamics in the synchrotron dominated regime," Phys. Plasmas 31, 052112 (2024). DOI [U]; arXiv 2404.11586. [V] Generalization with collisions. arXiv-OA.
58. MUST. J. M. Cole et al., "Experimental evidence of radiation reaction in the collision of a high-intensity laser pulse with a laser-wakefield accelerated electron beam," Phys. Rev. X 8, 011020 (2018). DOI 10.1103/PhysRevX.8.011020; arXiv 1707.06821. [V] Task-required. OA.
59. MUST. K. Poder et al., "Experimental signatures of the quantum nature of radiation reaction in the field of an ultraintense laser," Phys. Rev. X 8, 031004 (2018). DOI 10.1103/PhysRevX.8.031004; arXiv 1709.01861. [V] Task-required. OA.
60. SHOULD. C. N. Wistisen, A. Di Piazza, H. V. Knudsen, U. I. Uggerhoj, "Experimental evidence of quantum radiation reaction in aligned crystals," Nat. Commun. 9, 795 (2018). DOI 10.1038/s41467-018-03165-4; arXiv 1704.01080. [V] OA.
61. SHOULD. M. Mirzaie et al., "All-optical nonlinear Compton scattering performed with a multi-petawatt laser," Nat. Photon. 18, 1212 (2024). DOI 10.1038/s41566-024-01550-8. [V] Most recent chi of order 0.5 laser-beam experiment. Not OA.
62. SHOULD. L. Comisso and L. Sironi, "Pitch-angle anisotropy controls particle acceleration and cooling in radiative relativistic plasma turbulence," Phys. Rev. Lett. 127, 255102 (2021). DOI 10.1103/PhysRevLett.127.255102 [U on DOI; title A]. Collective RR in astrophysics; context for ring work. arXiv-OA.
63. OPT. A. Gonoskov, A. Bashinov, I. Gonoskov et al., "Anomalous radiative trapping in laser fields of extreme intensity," Phys. Rev. Lett. 113, 014801 (2014). DOI 10.1103/PhysRevLett.113.014801. [U] Collective RR trapping.
64. OPT. P. J. Bilbao et al., kinetic instabilities from RR in neutron-star magnetospheres, arXiv 2603.25949 (2026). [A] Latest ring follow-up; read only if pursuing Thesis C. OA.
65. OPT. B. Kettle et al., "Experimental constraints on radiation reaction using a laser-wakefield accelerator," New J. Phys. 23, 115008 (2021) [title, authors, DOI all U; not found by search]. Recent constraint; verify before download.

## E. Collision algorithms

66. MUST. T. Takizuka and H. Abe, "A binary collision model for plasma simulation with a particle code," J. Comput. Phys. 25, 205 (1977). DOI 10.1016/0021-9991(77)90099-7. [U] Task-required. Not OA.
67. MUST. K. Nanbu, "Theory of cumulative small-angle collisions in plasmas," Phys. Rev. E 55, 4642 (1997). DOI 10.1103/PhysRevE.55.4642. [V] Task-required; the scattering-angle distribution used. Not OA.
68. MUST. F. Perez, L. Gremillet, A. Decoster, M. Drouin, E. Lefebvre, "Improved modeling of relativistic collisions and collisional ionization in particle-in-cell codes," Phys. Plasmas 19, 083104 (2012). DOI 10.1063/1.4742167. [V title via WarpX docs, DOI U] Task-required; relativistic COM kernel. Not OA.
69. MUST. D. P. Higginson, I. Holod, A. Link, "A corrected method for Coulomb scattering in arbitrarily weighted particle-in-cell plasma simulations," J. Comput. Phys. 413, 109450 (2020). DOI 10.1016/j.jcp.2020.109450. [A on title; volume U] Unequal weights; directly relevant to the 1e-5 seed weighting. Not OA.
70. MUST. J. R. Angus, Y. Fu, V. Geyko, D. Grote, D. Larson, "Moment-preserving Monte-Carlo Coulomb collision method for particle codes," arXiv 2407.19151 (2024; journal version 2025 per L0 notes [U]). [V] Exact restoration, relativistic; compare with the program's restoration. OA (arXiv).
71. SHOULD. D. P. Higginson et al., J. Comput. Phys. 349 (2017), extension of the Nanbu method to include large-angle collisions [A on content; exact title, pages and DOI U]. Needed if the avalanche Moller source is implemented in the PIC.
72. SHOULD. Y. Sentoku and A. J. Kemp, "Numerical methods for particle simulations at extreme densities and temperatures: weighted particles, relativistic collisions and reduced currents," J. Comput. Phys. 227, 6846 (2008). DOI 10.1016/j.jcp.2008.03.043. [U] Relativistic collisions with weights. Not OA.
73. SHOULD. K. Nanbu and S. Yonemura, "Weighted particles in Coulomb collision simulations based on the theory of a cumulative scattering angle," J. Comput. Phys. 145, 639 (1998). DOI 10.1006/jcph.1998.6049. [U] Weighted Nanbu. Not OA.
74. SHOULD. "Numerical heating in particle-in-cell simulations with Monte Carlo binary collisions," arXiv 2101.05798 (2021). [V title via search; authors U] Statistical heating; relevant to the 5-8% bias at coarse dt. OA.
75. SHOULD. "On the accuracy of the binary-collision algorithm in particle-in-cell simulations of magnetically confined fusion plasmas," arXiv 2104.09100 (2021). [V title; authors U] Tokamak-relevant accuracy of Nanbu-type collisions; check for runaway tails. OA.
76. SHOULD. C. F. Kennel and F. Engelmann, "Velocity space diffusion from weak plasma turbulence in a magnetic field," Phys. Fluids 9, 2377 (1966). DOI 10.1063/1.1761629. [U] Task-required; QL diffusion coefficient that the PIC response is checked against. Not OA.
77. OPT. S. T. Beliaev and G. I. Budker, "The relativistic kinetic equation," Sov. Phys. Dokl. 1, 218 (1956). DOI none; and C. J. Braams and C. F. F. Karney, "Conductivity of a relativistic plasma," Phys. Fluids B 1, 1355 (1989), DOI 10.1063/1.858964 [U]. Relativistic drag oracle.
78. OPT. J. Derouillat et al., "Smilei: a collaborative, open-source, multi-purpose particle-in-cell code for plasma simulation," Comput. Phys. Commun. 222, 351 (2018). DOI 10.1016/j.cpc.2017.09.024; arXiv 1702.05143 [U]. [A] Smilei cross-code (collisions, LL, Niel, MC). arXiv-OA.

## F. Differentiable and adjoint kinetic simulation, optimal control

79. MUST. J. Ma, R. Jorge, Y. Lu, V. Tran, A. Woolford (JAX-in-Cell), "JAX-in-Cell: a differentiable particle-in-cell code for plasma physics applications," arXiv 2512.12160 (2025). [V] The program's own code paper; cite for the method. OA.
80. MUST. A. Joglekar and A. G. R. Thomas, "Unsupervised discovery of nonlinear plasma physics using differentiable kinetic simulations," J. Plasma Phys. 88 (2022), arXiv 2206.01637. DOI 10.1017/S0022377822000939 [V from URL]. Prior differentiable Vlasov-Fokker-Planck with control. OA.
81. SHOULD. M. Carvalho et al., "Learning collision operators from plasma phase space data using differentiable simulators," arXiv 2601.10885 (2026; JPP 2026 per L0 notes). [A] Closest recent work on differentiable collisions. OA.
82. SHOULD. M. Hoppe, I. Pusztai, T. Fulop et al., "Bayesian optimization of massive material injection for disruption mitigation in tokamaks," J. Plasma Phys. 89 (2023). DOI 10.1017/S0022377823000193 [from Cambridge URL; authors U]. [A] Existing optimization of runaway mitigation with DREAM; the benchmark to beat. Not OA unless free.
83. SHOULD. "Fluid and kinetic studies of tokamak disruptions using Bayesian optimization," J. Plasma Phys. (2024). DOI 10.1017/S0022377824000606 [from URL; authors U]. [A] Fluid vs kinetic runaway optimization differences.
84. SHOULD. "A physics-constrained deep learning treatment of runaway electron dynamics," arXiv 2412.12980 (2024). [A] Adjoint plus PINN for runaway density. OA.
85. SHOULD. Adjoint Fokker-Planck method for runaway probability and slowing-down time, OSTI 22493801. [A, title and authors not retrieved; DOI none known] Earliest adjoint runaway paper found; identify citation before use.
86. SHOULD. Sherwood Theory Conference 2025 abstract proposing adjoint exploration of disruption mitigation (sherwoodtheory.org/sw2025/uploads/29/abstract.pdf). [A, PDF not read] DOI none. Check immediately: it may overlap Thesis D.
87. SHOULD. G. Arya, M. Schauer, F. Schafer, C. Rackauckas, "Automatic differentiation of programs with discrete randomness," NeurIPS 2022, arXiv 2210.08572. [U] Gradient estimators for accept/reject and pair selection. OA.
88. OPT. L. Metz, C. D. Freeman, S. S. Schoenholz, T. Kachman, "Gradients are not all you need," arXiv 2111.05803 (2021). [U] Chaotic gradient blow-up; argues for the horizon limit seen in L1/L4. OA.
89. OPT. SFINCS-JAX differentiability notes (sfincs-jax.readthedocs.io/en/stable/differentiability.html). [V] Warning on gradients through singular Fokker-Planck systems. DOI none. OA.

## G. Codes (repositories to clone, not papers)

- DREAM: github.com/chalmersplasmatheory/DREAM [V]. MIT. Needed for the CODE/DREAM comparison tables the program lacks.
- KORC: github.com/ORNL-Fusion/KORC [U]. Full-orbit RE with LL RR; ask ORNL (Ghai, del-Castillo-Negrete) for AORSA-free single-wave inputs.
- Smilei: github.com/SmileiPIC/Smilei [U]; docs retrieved earlier. LL, cLL, Niel, MC radiation, Perez collisions.
- EPOCH: github.com/Warwick-Plasma/epoch [U]; Arber et al., PPCF 57, 113001 (2015), DOI 10.1088/0741-3335/57/11/113001 [U]. OPT. QED module.
- WarpX: github.com/BLAST-WarpX/warpx [U]; collisions source `UpdateMomentumPerezElastic.H` [V doc page]. OPT.
- CODE and NORSE: Matlab, listed at ft.nephy.chalmers.se/~hoppe/retools [V]; repo URLs not verified. LUKE: CEA, no public repository found.
- Differentiable: JAX-in-Cell (own); Carvalho 2026 repo not located.

## Counts

MUST: 34 (entries 1, 3, 4, 5, 6, 7, 8, 10, 11, 16, 24, 27, 29, 30, 31, 32, 42, 43, 44, 45, 46, 47, 48, 56, 57, 58, 59, 66, 67, 68, 69, 70, 79, 80)
SHOULD: 35 (entries 2, 9, 12, 13, 14, 15, 17, 18, 25, 26, 28, 33, 34, 35, 36, 37, 49, 50, 51, 60, 61, 62, 71, 72, 73, 74, 75, 76, 81, 82, 83, 84, 85, 86, 87)
OPT: 15 (entries 38, 39, 40, 41, 52, 53, 54, 55, 63, 64, 65, 77, 78, 88, 89)
Total numbered entries: 84 (number 19 unused).

Entries I could not verify at all (DOI or authors from memory; confirm in Crossref before download): 2, 3, 8, 12, 13, 15, 20, 21, 22, 23, 25, 35, 36, 37, 38, 49, 50, 52, 53, 54, 63, 65, 66, 71, 72, 73, 77, 87, 88. Entries where I only confirmed an abstract and lack the DOI: 7, 33, 34, 39, 41, 85, 86.
Not found and not listed: the "Spark 2019 DIII-D whistler" paper (possibly Spong 2018, entry 29, or Heidbrink et al. 2019 PPCF [U]); "Gammon" RR review (possibly Blackburn 2020 or Gonoskov 2022); DREAM/CODE published avalanche-rate tables.
