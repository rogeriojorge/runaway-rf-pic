import sys, pickle
from runSr_scan import *
name = sys.argv[1]
out = scan(name, 0.0, sigs=np.array([3.5, 4.5, 6.0, 7.0]), long_beats=(100., 400.))
pickle.dump(out, open(f"../data/Sr2_{name}_hi.pkl", "wb"))
