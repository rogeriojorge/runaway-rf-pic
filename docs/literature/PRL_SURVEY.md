# PRL survey: what from the JAX-in-Cell radiation / runaway-control work could be a Physical Review Letter (2026-11-10)

Method: web search and fetch (about 30 queries), plus the program's own notes (PLAN.md, literature/*.md, results/L1-L6, reference/NEXT_PHYSICS.md, note.tex). Verification tags used below and in READING_LIST.md:
[V] metadata (title, authors, DOI or arXiv id) confirmed in a retrieved page this session; [A] only the abstract was read; [U] cited from memory, not verified; do not quote without checking.
Limits of this survey: search returns abstracts and landing pages, not full texts. Nothing here was checked against a PDF. The Sherwood 2025 adjoint abstract (sherwoodtheory.org/sw2025/uploads/29) was fetched as a compressed PDF and not read. The user's list names "Spark 2019 (DIII-D whistler)" and "Gammon/Piazza RR reviews": I could not identify either; the nearest verified items are Spong et al. PRL 120 155002 (2018) and Di Piazza et al. RMP 84 1177 (2012). Treat the names as a typo until the user says otherwise.

## 1. Bottom line

None of the program's current results is a PRL as it stands. The three things that are real are (a) a differentiable relativistic-collision PIC with reduced/full/quantum Landau-Lifshitz (LL) radiation reaction, (b) a validated map of when a quasilinear (QL) whistler response to a PIC-resolved wave holds, and (c) a bounded negative result on sustained suppression (5.1 J/m3 over 0.3 s, regrowth afterwards). A PRL needs a general statement plus a number a referee outside the field can read. The best candidate, ranked below, is a criterion for when coherent (narrowband) helicon launch controls runaways diffusively and when it only traps them. It is the one place where a particle-resolved code gives something DREAM/CODE (QL operators) structurally cannot, and where a very recent DIII-D PRL (Choudhury 2026) and a KORC study (Ghai 2025) leave an open, quantitative question. Even that needs a DIII-D data link or a KORC cross-check to clear the PRL bar.

The quantum piece (g(chi)) does not reach PRL: for 20 MeV in 2 T chi is about 2e-8, and the strong-field PRL literature (Poder, Cole, Bilbao-Silva) already covers chi of order 0.1 to 10. The collision and differentiability pieces are code papers (CPC, JCP, JPP), not letters.

## 2. State of the art (what exists)

### 2.1 RF / whistler runaway control
- Theory of runaway-driven whistlers and EXEL waves: Fulop et al. PoP 13 062506 (2006) [V]; Komar, Pokol, Fulop, arXiv 1301.1250 (2013) [V] (near-critical fields); Pokol/Fulop quasilinear follow-ups [U]. Liu, Hirvijoki, Fu et al. PRL 120 265001 (2018), arXiv 1801.01827 [V]: self-consistent whistler instability plus avalanche, raises the effective critical field; suggests thermal-background control can suppress avalanche.
- First direct observation: Spong et al. PRL 120 155002 (2018) [V]: 100-200 MHz whistlers excited by multi-MeV runaways in DIII-D, limit-cycle amplitude, interpreted as wave-induced pitch-angle scattering.
- External launch (the state of the art for control): Choudhury et al., arXiv 2505.19279 (2025), PRL 136 025101 (2026) [V for arXiv, PRL number from the later preprint]: DIII-D, low-density Ohmic, normal-Doppler resonance near 8 MeV; hard-X-ray growth stops, runaways above the resonance energy fall below noise even with E still high enough for exponential growth; synchrotron and ECE emission rise (pitch-angle scattering). Follow-up arXiv 2606.22141 (June 2026) [V, abstract]: ideal vs misaligned antenna; n=1 (normal) scattering limits growth, evidence of a power threshold in the non-ideal case; anomalous Doppler (n=-1) waves increase the runaway population; rising tones 30-60 MHz in helicon-off periods; collisional damping and the outboard vacuum gap block wave penetration, complicating use on post-disruption beams.
- Kinetic simulation with externally supplied waves: Ghai, del-Castillo-Negrete, Spong, Beidler, arXiv 2509.13271 (2025) [V, abstract]: AORSA + KORC full orbit; pitch-angle increments follow asymmetric Laplace (intermittent large kicks), variance is non-diffusive with sub- or super-diffusive scaling set by initial energy. This directly contradicts QL expectations. A Peking University group added a QL diffusion term to DREAM for helicon scattering (slides only, [A]); their open questions are how the avalanche rate is suppressed and how scattered runaways are lost. A Sherwood 2025 / APS DPP 2025 "firewall effect" abstract [A] (externally launched whistlers suppress acceleration).
- Older RF: EAST and HT-7 lower hybrid and ECCD runaway studies (Zhang 2011 Phys Scr 84 015501 [U for exact numbers]); mostly reduction of E field or bulk heating, not resonant scattering. DIII-D ECH on relativistic electrons (APS DPP 2024 abstract) [A]; EC waves for post-disruption runaways need n_parallel of order 1, so O to slow-X conversion (IAEA 2024, Aleynikov et al.) [A].
- Closest to "minimum energy": nothing found. No paper states a minimum supplied-wave energy or power for runaway suppression, and none treats the wave power ledger (antenna, Poynting closure). That absence is the opening for the program's L4 work, but it is also a warning that the question may not be sharp (see thesis 2).

### 2.2 Radiation reaction (RR) in plasma
- Reviews: Di Piazza, Muller, Hatsagortsyan, Keitel RMP 84 1177 (2012) [V]; Blackburn, Rev. Mod. Plasma Phys. 4 5 (2020), arXiv 1910.13377 [V]; Gonoskov, Blackburn, Marklund, Bulanov RMP 94 045001 (2022), arXiv 2107.02161 [V].
- Experiments: Cole et al. PRX 8 011020 (2018) [V] (a0 about 25, over 500 MeV, energy loss correlated with gamma signal, classical models overestimate); Poder et al. PRX 8 031004 (2018) [V] (over 2 GeV, 4e20 W/cm2, up to 30% energy loss, quantum signatures); Wistisen et al. Nat. Commun. 9 795 (2018) [V] (aligned crystals, 178 GeV positrons); Mirzaie et al. Nat. Photon. 18 1212 (2024) [V] (4 PW, chi about 0.5). All are single-particle, vacuum-field experiments. None has a plasma collective signature.
- Collective RR in plasma: Bilbao and Silva, PRL 130 165101 (2023), arXiv 2212.12271 [V]: differential RR cooling in a strong B produces ring momentum distributions with population inversion, timescales derived, PIC-confirmed (classical and quantum). Bilbao, Ewart, Assuncao, Silva, Silva, Phys. Plasmas 31 052112 (2024), arXiv 2404.11586 [V]: generalization; Maxwellians above T of about mc^2/sqrt(3) form rings; collisional effects and non-uniform B discussed. Astrophysical: Comisso and Sironi PRL 127 255102 (2021) [U on DOI] radiative turbulence; Nattila and Beloborodov (2020-21). In tokamaks: Hirvijoki et al. JPP 81 475810502 (2015), arXiv 1502.03333 [V] and Decker et al. arXiv 1503.03881 [V]: RR plus collisions make a bump in the runaway tail; Aleynikov and Breizman PRL 114 155001 (2015) [V]: two threshold fields, synchrotron-limited energy; Andersson, Helander, Eriksson PoP 8 5221 (2001) [U]: synchrotron damping of beams.
- Consequence for novelty: RR-driven rings and bumps are published in both the laser-plasma PRL (Bilbao-Silva) and the tokamak runaway literature (Hirvijoki, Decker). Their link to wave scattering of runaways is not made explicitly in anything found, but the tokamak-ring connection is flagged by Bilbao et al. as an extrapolation.

### 2.3 Differentiable / adjoint kinetic simulation and optimal control
- Differentiable PIC: JAX-in-Cell, arXiv 2512.12160 (Ma, Jorge, Lu, Tran, Woolford) [V]; the program's own code. No other differentiable PIC with binary Coulomb collisions found. Differentiable continuum kinetic: Joglekar and Thomas JPP 88 (2022), arXiv 2206.01637 [V] (Vlasov-Poisson-Fokker-Planck, neural-network forcing, entropy loss); Carvalho et al. arXiv 2601.10885 (2026) [A] (learn Fokker-Planck operators from PIC data); SFINCS-JAX implicit adjoints [V docs].
- Runaway-specific: adjoint Fokker-Planck for runaway probability (OSTI 22493801) [A]; adjoint plus PINN, arXiv 2412.12980 [A]; Bayesian optimization of massive material injection with DREAM (Hoppe et al. JPP 2023, doi from Cambridge PII 10.1017/S0022377823000193 [U]) and fluid-vs-kinetic follow-up (JPP 2024); Sherwood 2025 abstract proposing adjoints to explore disruption-mitigation strategies [A, unread].
- Net: gradient-based optimal control of runaways with a differentiable kinetic solver is not published as far as found, but the surrounding ground (adjoint FP, Bayesian optimization of mitigation) is occupied. "We differentiated through it" is a methods result.
- Gradients through stochastic collisions: pathwise Nanbu angles and score-function (Arya et al. NeurIPS 2022 [U]); chaotic-gradient divergence (Metz et al. 2021 [U]). Program notes show this as a real limit (horizon, noise floor).

### 2.4 Relativistic collision algorithms
Takizuka-Abe 1977, Nanbu 1997 [V], Nanbu-Yonemura 1998, Sentoku-Kemp 2008, Perez et al. 2012 (relativistic COM), Higginson et al. JCP 2017 and 2020 (arbitrary weights) [A], Angus et al. arXiv 2407.19151 [V] (LLNL, exact moment restoration, relativistic). The program's restoration step is therefore not new in principle; its use for a 1e-5 weight-ratio seed population is an application result. Smilei and WarpX implement Perez; DREAM/CODE/NORSE use continuum operators including the Embreus large-angle (Moller) source.

### 2.5 Codes and data (verified where marked)
- DREAM: github.com/chalmersplasmatheory/DREAM, MIT, C++/Python [V]; CPC 268 108098 (2021). Fluid, kinetic, mixed. Examples `runaway*`, `dream_avalanche` physics test. Retools page ft.nephy.chalmers.se/~hoppe/retools lists NORSE (Matlab), CODE (Matlab), LUKE (CEA, bounce-averaged) and says NORSE/CODE are superseded by DREAM [V]. I did not find public repos for LUKE; NORSE and CODE URLs not verified.
- KORC (ORNL-Fusion, full-orbit RE code with LL RR, AORSA coupling) [U on URL]; Smilei (SmileiPIC/Smilei; LL, cLL, Niel, MC radiation; Perez collisions) [U on URL, docs retrieved earlier]; EPOCH (Warwick-Plasma/epoch; QED) [U]; WarpX (collisions `UpdateMomentumPerezElastic.H` [V doc page]). Differentiable: JAX-in-Cell, Carvalho (repo not located).
- Data: No published runaway-control dataset found. DIII-D shots are not public; the Choudhury preprints show spectra, HXR, ECE figures only. Ask Choudhury/Paz-Soldan for digitized HXR vs time and wave power, or work from figures. DREAM example inputs are the only public kinetic runaway inputs. The program must generate its own CODE/DREAM comparison tables (earlier L0 notes confirm none were retrieved).

## 3. Novelty audit of the program's results

| Claim | Prior art | Verdict |
|---|---|---|
| Differentiable PIC with binary collisions + RR | JAX-in-Cell (own); none with collisions found | New as a tool; methods venue |
| Moment-restored Nanbu-Perez for 1e-5 seed weights | Angus 2025, Higginson 2020 | Application, not new |
| Reduced LL = full LL to O(1/gamma); Bilbao ring reproduced to 8-12% | Vranic 2016, Bilbao-Silva 2023 | Verification, not new |
| Synchrotron fixed point gamma 13.1 at 2 T, E 5 mV/m; no bifurcation in E | Aleynikov-Breizman 2015 (two thresholds), Andersson 2001 | Known physics; check that "no bifurcation" is not a model artefact of the reduced ODE (their result has thresholds in a pitch-resolved treatment) |
| QL response vs resolved push: ratio 0.92-1.67, fails for 3x line spacing, trapping with 1-3 lines | Karney-style stochasticity, Ghai 2025 (non-diffusive in KORC) | Useful, partly known as Chirikov overlap; the quantitative runaway map is new |
| 5.1 J/m3 minimum supplied energy, tail exchange 16.9 W/m3 | None | New, but only for one declared horizon; not sustained; sensitive to QL error (up to ~9 J/m3) and unmodelled antenna loading |
| Evolution-strategy 3-band spectra beat one band by 1.6x in energy | None | Model-dependent, fails at amplitude x0.77; not a result |
| Quantum g(chi) | Niel 2018 | Irrelevant to tokamak chi |

Honest assessment of the 5.1 J/m3: energy supplied is not the physical cost of wave scattering. In QL theory the resonant scattering is nearly elastic in the wave frame, so the electrons' energy comes from E and goes to synchrotron; the wave only supplies the small energy exchange with the tail. The real cost is damping, antenna coupling, and mode conversion, which the model omits. A referee will say the "minimum energy" is a number about the ledger convention, not about plasma physics. Keep it as a result, not a headline.

## 4. Ranked candidate PRL theses

### Thesis A (best): Coherent-wave runaway control saturates at a trapping-limited rate; narrowband launch is not diffusive

Thesis. A single coherent helicon line resonant with a runaway cannot scatter it at the quasilinear rate. Above a threshold amplitude the response is bounce-limited: pitch-angle scattering rate saturates (grows with sqrt of amplitude through the bounce frequency, not linearly with power), the increments are non-Gaussian, and the electron is trapped in the resonance island and carried along the resonance curve rather than diffused out of the runaway region. Control then needs spectral width (at least about 30 lines with overlap parameter 2 w_b / dDelta_sp of about 3 in our map) or sweeping, and the power threshold seen in DIII-D follows from island overlap with the radiation-limited drift, not from damping. Prediction: a threshold amplitude B_w/B scaling with (nu_eff, tau_syn, gamma_res), a saturating suppression curve vs power, and a recommended launch bandwidth.

Why new. DIII-D shows the effect and a power threshold with no mechanism (Choudhury 2025, 2026). KORC shows non-diffusive intermittent kicks with no criterion (Ghai 2025). DREAM-QL (PKU) and Liu 2018 assume QL. Chirikov/Karney theory of resonance overlap is old (Karney 1978; Karney and Fisch 1979 [U]), but nobody has applied it with radiation reaction, collisions and the synchrotron-limited fixed point to runaway control. Our L4 validation already contains the discriminating observation: one and three coherent lines trapped, no diffusion; 32 lines at the overlap criterion diffuse and agree with QL within 30%.

Decisive figure/number. One two-panel figure. (a) Scattering rate (or fractional drop in the above-resonance population at fixed time) vs wave amplitude, with QL line, the resolved PIC, and the bounce-limited line, crossover marked; (b) the same population drop vs applied helicon power from the DIII-D experiment (Choudhury Fig. showing HXR vs time for ideal/non-ideal antenna) with the predicted threshold amplitude as a vertical band. Headline number: the overlap parameter at the DIII-D launch (frequency, B_w estimated from power and antenna loading) and the factor by which a coherent line is below QL.

Required runs. (1) Resolved 0D3V or 1D3V PIC with collisions and RR over the DIII-D-like case (about 8 MeV resonance, B about 2 T, n about 1e19 or lower), amplitude scan over 2 decades, line number scan 1 to 128, 3 realizations each. (2) Same with RR off and collisions off to isolate trapping. (3) DREAM with a QL pitch-angle diffusion operator (needs the PKU addition or a user-written D_theta term) at the same inputs; show it overpredicts suppression for coherent launch. (4) KORC test-particle ensemble in a prescribed single-line whistler (Ghai's model without AORSA) for the saturation and Laplace increments; this is the cross-code check. (5) Ask the DIII-D group for power scan data or digitize.

Risks. Experimental link is the gate: without it the paper is a generic wave-particle trapping paper (PoP/PRE level). The DIII-D group (Paz-Soldan, Spong, Tang) may publish this themselves. Fields in the experiment (n_parallel, B_w profile) are uncertain by factors that move the overlap parameter. Reduced geometry (homogeneous, no toroidal trapping) must be argued, or KORC must supply it. Overlap physics may be judged known. Probability of PRL acceptance after the work: about 25%, higher with a DIII-D coauthor.

### Thesis B: A power-budget law for sustained runaway suppression, and why transient suppression is cheap but sustained is not

Thesis. Because runaways are synchrotron-limited, a resonance window at gamma_res only needs to catch the flux of electrons crossing it. The minimum sustained wave resource is set by flux through the gate times the exchange per crossing, plus damping, and obeys a scaling in E/E_c, B, gamma_res that can be stated in closed form. The program's finding that a 5.1 J/m3, 0.3 s burst yields no sustained suppression (regrowth in 1.7-3 s) turns into the law that the gate must be held, with a duty cycle bounded by the regrowth time.

Why new. No paper states any minimum wave resource for runaway suppression. Closest: Choudhury (power threshold), Liu 2018 (kinetic effective critical field), Aleynikov-Breizman (energy ceiling), Hesslow 2018/2019 (effective critical field with radiation).

Decisive figure. Minimum power density (or J/m3 per second held) vs E/E_c collapsing onto a single curve for several B and gamma_res, with kinetic (DREAM or CODE) and PIC points on the same line.

Required runs. Systematic 3-parameter scan (E/E_c 3 to 30, B 1.5 to 5 T, gamma_res), sustained horizon of several regrowth times, CODE/DREAM with the same QL operator to confirm, a robust variant (the current robust search found 0 feasible out of 720: that must be resolved, e.g. by pitch-aware targets).

Risks. High. The energy ledger is convention-dependent; antenna and damping dominate the real cost; E/E_c of 20 is well above many experiments; the avalanche, bremsstrahlung and transport are omitted; amplitude error of 30% moves the answer 4 to 6 J/m3. The sustained case already failed in our own search. As it stands this is a weak thesis. Probability about 8%.

### Thesis C: Wave scattering turns the runaway tail into a radiation-reaction ring that is itself unstable

Thesis. Pitch-angle scattering by an external wave pushes runaways to large p_perp where RR is strong; the balance of drive, scattering and RR produces a ring/bump (non-monotonic in p_perp) whose inversion exceeds the whistler/ECM threshold and explains the rising ECE/synchrotron and the self-excited 30-60 MHz tones seen in helicon-off periods (Choudhury 2026). The same mechanism sets the saturated state of controlled runaways.

Why new. Hirvijoki/Decker give bumps without waves; Bilbao-Silva give rings without collisions or tokamak drive; Liu 2018 gives instability without external waves. The combination (external wave, RR ring, secondary instability) is not found.

Decisive figure. f(p_perp) at fixed p_par showing ring formation, with the ECM/whistler linear growth rate computed from that f vs the observed 30-60 MHz tones and the measured ECE enhancement.

Required runs. Resolved PIC with RR and collisions; a linear hot-plasma solver (or Liu's kinetic-instability code) on the PIC f; comparison with CODE/DREAM f including synchrotron; ECE/synchrotron synthetic spectra (SYNO/KORC).

Risks. Tokamak tau_B of 1.3 s vs tau_c of 2.2 s means collisions erase the ring partly (Bilbao 2024 already shows collisional dependence); a 0D3V homogeneous model cannot give a growth rate that depends on spatial structure; quantum physics irrelevant. Probability about 12%.

### Thesis D (weak): first differentiable optimal control of runaway suppression

Thesis. End-to-end gradients through a collisional RR PIC find wave spectra that suppress runaways with a given budget; optimal spectra are multi-band combs placed around the fixed point.

Why weak. Adjoint FP and Bayesian optimization for runaway control exist; our ES result is model-dependent and fails at amplitude x0.77; the PIC gradient is noisy with a limited horizon; no experimental or physical principle emerges. The correct venue is CPC/JCP/Phys. Plasmas or a JPP methods paper. Probability of PRL: below 5%. Value: it is the tool that makes A and B cheap (sensitivity of threshold to parameters), so use it as support, not as the thesis.

### Not recommended
- Quantum mean-recoil g(chi) in a tokamak (chi about 1e-8): irrelevant.
- Reproducing Bilbao-Silva rings with RR+collisions as stand-alone: a verification and was matched at 8-12%.

## 5. Recommended path and sequencing

1. Decide on Thesis A and spend one cycle on the discriminating experiment: coherent-vs-broadband scan of the L4 setup with RR and collisions on, extended to tokamak-relevant resonance and wave amplitude. Gate: crossover from QL to trapping-limited scattering reproduced within 30% by an analytic bounce-limit formula.
2. Cross-code check against KORC single-line test-particle runs (Ghai's code is the natural collaborator) and against DREAM with a QL term. Without a second code, a PRL referee will not accept a PIC-only criterion.
3. Contact the DIII-D helicon group (Choudhury, Paz-Soldan, Spong) with the criterion and ask for power-scan data; a joint letter is the realistic path.
4. Treat Thesis B/C as follow-up papers. Resolve the sustained-suppression failure before claiming any budget.
5. Keep the differentiable machinery as the method, citing JAX-in-Cell; publish the code paper separately.

Risks across all: the mismatch between PIC time scales (microseconds) and runaway scales (seconds) needs the accelerated-rate rescaling preserving tau_c : tau_B : tau_RF; homogeneous geometry vs toroidal; E/E_c of 6 to 20 vs experimental near-critical conditions.

## Novelty verdict 2026-11-11

Inputs: the full texts of Kang et al. NF 64 106025 (2024), Guo et al. PoP 25 032504 (2018), Guo et al. PPCF 59 044003 (2017) and Kang et al. JPP 92 E101 (2026), plus Karney and Fisch (1986). Page numbers are PDF pages (details in `papers/KEY_FACTS.md` sections 9, 12-14). Quotes are short phrases.

### Corrections that change earlier statements
- **Kang 2024 is not a trapping paper.** KEY_FACTS 9 and the PLAN revision of 2026-11-11 attributed trapping, the b > 4.75e-4 threshold and the 10^6 speed-up to Kang 2024. That came from a citation in Kang 2026. The 2024 paper has no trapping, no threshold, no quasilinear comparison of scattering and no launched wave: it is a beam-driven instability PIC, and it states "did not use any collision or radiation module" (p5). Trapping, the threshold and the 10^6 factor are in Kang 2026 only.
- **Guo 2017 contains no waves.** It is the collisions-plus-synchrotron vortex baseline.
- **Section 2.1 above says no paper states a supplied-wave power.** That is false: Guo 2018 estimates about 800 kW for ITER (p9) and Kang 2026 gives P of about 155 kW as an upper bound for KSTAR-like parameters (p10 of that paper). Both are wave-damping power balances, not thresholds.

### Part (i): multi-line overlap criterion and required launch bandwidth
- Guo 2018: one line only, a Gaussian in k_par of width 0.05 (eq. 15, p5). The scans move its centre to place the resonance between the X and O points (Fig. 5, p7-8). No second line, no overlap, no bandwidth requirement.
- Kang 2026: one coherent R-wave. Kang 2024: no launched spectrum; the "unpredicted signals" near the resonances are left for future work (p12).
- Guo 2017: no waves.
- Choudhury PPCF (not in the list but relevant): App. B says the REs "could be stochasticized in phase space" and a broader n_par spectrum "could be beneficial". This is a remark, not a criterion.
- **Verdict: OPEN.** None of the four papers treats it.
- Risks. (a) In the quasilinear regime a broad spectrum is already the Guo 2018 formalism, so overlap only has content for coherent lines (amplitude above trapping); our result must be stated in that regime. (b) A Chirikov-type criterion is textbook, so the novelty is in the specific result: the required bandwidth as a function of amplitude, energy and the O-X geometry, and a numerical test.

### Part (ii): collisions plus radiation reaction acting on the trapped-island dynamics
- Guo 2018 and 2017 do treat collisions, synchrotron damping and the wave term on the same footing. So "wave scattering plus collisions plus RR self-consistently" is **not new in the quasilinear regime**. Their mechanism is the one we would otherwise call ours: waves raise the pitch, synchrotron damping balances E at lower energy, p_O falls from 20 to 3.5 (p7, eq. 21).
- Kang 2026: "All species are assumed collisionless" (p6 of that paper); no RR. Kang 2024: collisions and RR absent by construction (p5).
- Nobody treats a coherent, trapped resonance island together with Coulomb pitch-angle scattering, friction and synchrotron damping.
- **Verdict: PARTLY COVERED.** Covered in the quasilinear limit (Guo). Open for trapped dynamics. The PLAN wording "no prior work treats self-consistently" must read "in the trapped (non-quasilinear) regime".
- Caveat: the PLAN's own DREAM result (kinetic bump at gamma 17 against the ODE) shows pitch diffusion matters; the vortex of Guo 2017 is the right baseline to reproduce first (P1).

### Part (iii): quantitative amplitude threshold tied to DIII-D helicon power
- Kang 2026 gives a single-particle trapping threshold: all electrons trapped for b > 4.75e-4 at E_0 = 3.55e-5, none for b < 3.9e-4, and P = (|w_i|/2pi)(b^2 B_0^2/2mu_0)V_p, about 155 kW (KSTAR, B0 = 3.5 T, ne = 1e19, Te = 100 eV; p6 and p10 of that paper).
- Guo 2018 gives a quasilinear amplitude (dB below 1e-4 B0, E0 = 2.5e-11) and 800 kW (p9). The Guo QL amplitude (dB/B0 of 1e-5 to 1e-4) is about 5x to 50x below Kang's trapping amplitude (5e-4), so the two papers sit on opposite sides of a crossover that nobody has located. That is a hypothesis to test in P2, not a result.
- No paper ties either number to the DIII-D helicon data. The DIII-D threshold itself is hedged in the published PPCF ("Further data is needed to determine the existence of a power threshold"), from three non-ideal shots at 0.04-0.08 MW mean power, with a PRL figure of about 150 kW.
- **Verdict: OPEN but weak.** Open because no tie exists. Weak because the experimental threshold is not established, the wave amplitude inside the plasma is unknown without ray tracing (the Kang threshold is for a uniform wave with specified E_0), and a power-balance P = Gamma W_k formula already exists in both papers. A power estimate cannot be the headline.

### Overall
Nothing in the narrowed thesis is already published as stated, but each part is a refinement of an existing result: the trapping mechanism and amplitude threshold (Kang 2026), the collisional-RR vortex control with its power estimate (Guo 2018), a one-line wave. No part is a new mechanism or a new law yet. As written, this is a good PoP/NF/JPP paper and not a PRL. It could become a PRL only if P2 and P3 give a sharp, falsifiable law, for example a single dimensionless boundary in (wave amplitude, collisionality, alpha = tau_c/tau_s, line spacing) that reproduces Guo's quasilinear control at low amplitude and Kang's trapping at high amplitude, with a bandwidth law for overlap, confirmed against a trapping-free quasilinear solver (a DREAM fork with the Kennel-Engelmann term) and a PIC. That is what to test first. If P2 shows the PIC rate following quasilinear theory up to the DIII-D amplitudes, the kill criterion in the PLAN applies and the thesis ends.

### Recommended final thesis wording
"Resonant runaway control by a launched whistler or helicon spectrum has two regimes, a quasilinear one (Guo et al. 2018) and a coherent, trapped one (Kang et al. 2026). We map the crossover between them as a function of wave amplitude, collisionality and radiation damping, show that for discrete lines the control needs the line spacing to satisfy an overlap condition set by the trapping width, which fixes the launch bandwidth, and find where the DIII-D helicon amplitudes sit on this map."
Do not claim: new trapping mechanism; first self-consistent collisions and RR with waves; an established DIII-D power threshold.

### Strongest alternative if the crossover result is not sharp
Adjoint-optimal launch spectrum for runaway energy control. Guo 2018 states that extending the adjoint method of rf current drive to energy control "remains to be carried out" (p8-9). Karney and Fisch give the formalism (eqs. 6-8: one adjoint solve gives a chosen moment for any wave source S); the extension adds synchrotron damping, the avalanche source and a quasilinear wave term, and our differentiable solver supplies the gradient with respect to the spectrum. The deliverable is the minimum-power spectrum that holds p_O below a target, constrained by the overlap condition. Overlap warning: the Sherwood 2025 abstract (entry 86) already combines the Karney-Fisch adjoint, a PINN surrogate and mitigation strategies, so the claim must be limited to wave-spectrum design. This is a PoP/NF paper, not a PRL, unless a DIII-D design result changes that.

### Next checks (not done here)
- Karney-Fisch sections IV-VI (runaway probability, current Green's function) were not transcribed; only eqs. 1-8 were read from the images.
- The Choudhury PPCF attribution of a saturation-with-wave-energy statement to Guo et al. was not found in Guo 2018; check the Choudhury reference list.
