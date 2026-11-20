"""usage: batch.py OUTTAG NBEATS DTFAC NP SEED sigma Gamma1 Gamma2 ...   (one sigma, several Gamma_phi/Delta_omega); saves data/OUTTAG_s{sigma}_g{G}_seed.npz"""
from pn import *
tag = sys.argv[1]; nb = float(sys.argv[2]); dtf = float(sys.argv[3]); NP = int(sys.argv[4]); seed = int(sys.argv[5]); sg = float(sys.argv[6]); Gs = [float(x) for x in sys.argv[7:]]
dd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
for G in Gs:
    fn = f"{dd}/{tag}_s{sg}_g{G}_seed{seed}.npz"
    if os.path.exists(fn): continue
    ref0 = ql_reference(); d = run('eqN10', sg, G, nb, dtf, NP, seed=seed, stop_m2=(1.4*0.5*ref0[1][-1] if os.environ.get('NOSTOP') is None else None))
    np.savez(fn, **{k: v for k, v in d.items() if k not in ("g_init",)})
    ref = ql_reference(); s, tc, sat = s_star(d, ref)
    print(f"{tag} sigma={sg} sigma_eff={d['sigma']:.2f} G={G} seed={seed} T={nb} NP={NP} dtfac={dtf} steps={d['nst']:.2e} time={d['time']:.0f}s s*={s:.3f} t_c={tc:.1f} sat={sat} m2end/Lw2={d['m2'][-1]/d['Lw']**2:.3f}", flush=True)
