# runaway-rf-pic

Studies of runaway-electron suppression with radio-frequency waves, built on the particle-in-cell code [JAX-in-Cell](https://github.com/uwplasma/JAX-in-Cell). They cover a no-RF runaway baseline, differentiable RF control, the minimum supplied RF energy, and the scattering of runaways by a launched comb of helicon lines at DIII-D parameters. The write-up is `paper/note.pdf`.

## Install and run

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pytest
python studies/runaway_baseline/runaway_baseline.py
```

Python 3.10 or newer. JAX-in-Cell is pinned to a commit in `pyproject.toml` (one line). Each script runs in its quick preset in 10 to 25 s on a laptop CPU; `research` comments in the scripts mark the converged settings.

## Layout

| Path | Content |
|---|---|
| `studies/` | Four JAX-in-Cell examples: `runaway_baseline`, `runaway_rf`, `runaway_rf_minimum`, `rf_source_control`, each with a short README |
| `tests/` | The four scripts in their quick presets, and a Connor-Hastie critical-force check of the collision kernel |
| `paper/` | LaTeX note, figure script, figures, compiled PDF |
| `docs/` | Numerics note for the RF control study, research log, literature survey and key facts, reference notes |
| `results/` | Analysis scripts, notes and small data for each stage of the helicon study (P1-P3, step1-3, pinning, L2, L4, next_physics) |

The helicon-scattering scripts in `results/` are archival. They were run against earlier JAX-in-Cell development branches from the repository root and are not covered by the tests. Some of them insert a local JAX-in-Cell clone into `sys.path`.

## Main results

- Baseline. In the 0D3V runs at B = 2 T, E = 5 mV/m (E/E_c = 6.5) the tail settles to a mode at gamma = 17.7, against 17.0 from DREAM. The synchrotron fixed point at gamma = 13.1 belongs to the noise-free test-particle ODE and does not bound the kinetic tail.
- Minimum RF energy. At E = 5 mV/m there is no minimum-RF problem, because the fixed point lies below 10 MeV. The study uses E = 15 mV/m as a declared numerical choice.
- Single line and comb. A coherent single line is trapping-limited at DIII-D amplitudes. For a comb, diffusion switches on at a Chirikov parameter near 1, about b = 1e-4 per line, with threshold scaling as spacing to the power 1.9.
- Above that threshold the motion of a coherent comb is bounded chaos: diffusion is transient over 1 to 10 microseconds and fills 0.5 to 0.7 of the line band. Sustained quasilinear scattering needs phase decorrelation of at least 0.01 to 0.14 of the line spacing.
- DIII-D. With estimated wave amplitudes the comb is coherent and sub-overlap (sigma 0.6 to 0.8), so the model does not explain the observed hard X-ray reduction unless parametric-decay broadening is large. Amplitude pinning at threshold is refuted.
- Predictions to test: 1 to 3 MHz frequency modulation turns sustained scattering on; Doppler backscattering amplitude near 1e-4 per line at threshold; a ring-down of about 3.7 microseconds after the source turns off; synchrotron or ECE emission rises by 15 to 25 kW per 1e15 m^-3 resonant electrons above onset.

Limits are listed in the final section of the note: one 10-line comb, uniform plasma, cold-tensor damping, a P1-type runaway distribution, and a map accurate to a factor of 1.5 to 2 per cell.

## Not included

Paper PDFs and extracted text, third-party code runs, performance profiling data, and raw arrays larger than 600 kB. The note describes them but they are not needed to read or rerun the studies.

## License

MIT, see `LICENSE`.
