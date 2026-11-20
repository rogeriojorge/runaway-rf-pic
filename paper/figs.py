import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# L1 speedups
c=["expl 131k","expl 16k","expl 2k","impl 131k","impl 16k","rad 131k","coll 131k","expl f32 131k"]
b=np.array([108,21.6,5.4,1550,173,375,226,60.9]); a=np.array([50.3,13.2,4.9,916,148,310,166,14.7])
fig,ax=plt.subplots(figsize=(6,2.8)); ax.barh(c,b/a,color="#4477aa"); ax.axvline(1,c="k",lw=.8)
ax.set_xlabel("speedup (before/after), RTX A4000, 64 cells"); fig.tight_layout(); fig.savefig("fig/l1_speedup.pdf")
# L5 g(chi)
x=np.logspace(-6,3,300); g=(1+4.8*(1+x)*np.log1p(1.7*x)+2.44*x**2)**(-2/3)
fig,ax=plt.subplots(figsize=(4.5,2.8)); ax.loglog(x,g,label="Niel fit g($\\chi$)")
ax.loglog(x[x>10],1.04*x[x>10]**(-4/3)*1.0,"--",label="$\\propto\\chi^{-4/3}$")
ax.axvline(2e-8,c="gray",ls=":"); ax.set_xlabel("$\\chi$"); ax.set_ylabel("g"); ax.legend(); fig.tight_layout(); fig.savefig("fig/l5_gchi.pdf")
# L4 strategies
s=["single band","3-band random","3-band ES"]; q=[8.2,9.0,5.1]
fig,ax=plt.subplots(figsize=(4.5,2.4)); ax.bar(s,q,color=["#999","#999","#228833"]); ax.errorbar([2],[5.1],yerr=[[0.9],[1.0]],c="k",capsize=4)
ax.set_ylabel("verified $Q_+$ (J/m$^3$)"); fig.tight_layout(); fig.savefig("fig/l4_strategies.pdf")
# L2 regrowth table as curves: gap closure times
g0=[2,5,10]; t=np.array([[4.9,7.1,16.9],[6.2,8.5,18.3],[3.5,5.7,15.7]])
fig,ax=plt.subplots(figsize=(4.5,2.6))
for i,gg in enumerate(g0): ax.plot(t[i],[50,63,90],"o-",label=f"$\\gamma_0$={gg}")
ax.set_xlabel("time after suppression (s)"); ax.set_ylabel("gap closed to $\\gamma_{eq}$ (%)"); ax.legend(); fig.tight_layout(); fig.savefig("fig/l2_regrowth.pdf")
