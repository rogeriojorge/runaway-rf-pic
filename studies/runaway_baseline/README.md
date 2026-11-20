# runaway_baseline

Homogeneous no-RF baseline: a gyrotropic 0D3V model with an imposed total DC field, collisions (JAX-in-Cell's binary kernel), reduced Landau-Lifshitz synchrotron emission, and a test-particle ODE reference. At B = 2 T, n = 1e18 m^-3, E = 5 mV/m (6.5 E_c) the ODE has a synchrotron-limited fixed point at gamma = 13.1; the kinetic tail peaks near gamma = 17.7 once pitch-angle diffusion is resolved, which agrees with DREAM.

    python studies/runaway_baseline/runaway_baseline.py

Quick preset, about 15 s. The settings marked `research` in the script give the converged version.
