from pc2 import *
S = setup("eqN1", 1000, 0.7, NP=4096, ramp_wb=20.0)
k0 = red_dg(S, 0.03, kappa=0.0)["dg"]; k1 = red_dg(S, 0.03, kappa=1.0, kD=0.0)["dg"]
np.savez("../data/item1_red_hist.npz", k0=k0, k1=k1); print(np.median(k0), np.median(k1), np.mean(k1 > np.median(k1)+0.01))
