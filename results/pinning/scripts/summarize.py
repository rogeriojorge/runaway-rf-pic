import json, glob, numpy as np
D = "../data/"
rows = []
for f in sorted(glob.glob(D+"pin_s*_b*_lam*.json")):
    o = json.load(open(f)); r = o["rec"]; h = len(r)//2
    p2 = np.mean([x["th2"] for x in r[h:]])
    rows.append((o["bpk"], o["lam"], o["dt"]*o["wbmax"], o["Np"], o["P_wave"], o["P_wave"]/o["lam"], o["P_drive"]/o["lam"], p2, o["recycled_per_particle_per_sim_s"]/o["lam"], "dt0125" in f))
rows.sort()
print("| b_pk | sigma | lam | w_b dt | Np | P_wave [mc2/s sim] | P_wave/lam | P_drive/lam | <p_perp^2> | exit rate [1/s phys] |")
import pin_model as pm
for r in rows:
    print(f"| {r[0]:.0e} | {float(pm.sigma_of_b(r[0])):.2f} | {r[1]:g} | {r[2]:.3f} | {r[3]} | {r[4]:.4g} | {r[5]:.3g} | {r[6]:.3g} | {r[7]:.1f} | {r[8]:.2f} |{' (dt test)' if r[9] else ''}")
