from p3lib import *
for N in (5, 10):
  ks, a = spectrum("eq", 0.57, N)
  for b in (2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3):
    r = run_S(ks, a, b, 0.57, Np=128)
    print(f"N={N} b={b:.1e} sig={r['sigma']:.2f} beats={r['beats']:.0f} bounces={r['bounces']:.0f} Lc={r['Lw']:.3f} rho={r['ratio']:.3f}+-{r['ratio_err']:.3f} S_end={r['S_end']:.2f} th_rms={r['th_rms']:.3f} t={r['time']:.0f}s", flush=True)
