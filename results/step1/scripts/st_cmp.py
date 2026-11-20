from red_st import *
name = sys.argv[1]; NP = int(sys.argv[2]); nb = float(sys.argv[3])
for sg in [float(x) for x in sys.argv[4:]]:
    d = run_st(name, sg, nb, 0.1, NP)
    print(f"{name} sigma={d['sigma']:.2f} periods={nb} (=1/Dd units: {nb*2*np.pi:.0f}) NP={NP} steps={d['nst']:.2e} t={d['time']:.0f}s ratio[.5-1]={slope_ratio(d,.5,1):.3f} ratio[0-1]={slope_ratio(d,0,1):.3f} fill_end={d['R_fill']:.3f}", flush=True)
