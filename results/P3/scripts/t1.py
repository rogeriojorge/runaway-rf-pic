from p3lib import *
ks, a = spectrum("eq", 0.57, 5)
for b in (1e-4, 3e-4, 1e-3):
    r = run_S(ks, a, b, 0.57, Tfix=300/ (np.sqrt(b/1e-4)*4.2e7), Np=128)
    m2 = r["m2"]; ts=r["ts"]
    print("b",b,"T",r["T"],"Dql",r["Dql"],"wb",r["wb"],"Lw2/6",r["Lw"]**2/6)
    print(" S(t)=m2/(Lw^2/6):", np.round(m2[::6]/(r["Lw"]**2/6),3))
    print(" m2/(Dql t):", np.round(m2[::6]/(r["Dql"]*ts[::6]),2), "th_rms",r["th_rms"], flush=True)
