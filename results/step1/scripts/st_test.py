from red_st import *
name = sys.argv[1]; sig = float(sys.argv[2]); nb = float(sys.argv[3]); dtf = float(sys.argv[4]); NP = int(sys.argv[5])
d = run_st(name, sig, nb, dtf, NP)
print(f"{name} sigma={d['sigma']:.2f} nbeats={nb} dtfac={dtf} NP={NP} steps={d['nst']:.2e} t={d['time']:.0f}s  ratio[.5-1]={slope_ratio(d,.5,1):.3f} [.25-.5]={slope_ratio(d,.25,.5):.3f} [0-.25]={slope_ratio(d,0,.25):.3f} fill_end={d['R_fill']:.3f}")
