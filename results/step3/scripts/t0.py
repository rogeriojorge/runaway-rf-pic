from pn import *
sg=float(sys.argv[1]); G=float(sys.argv[2]); nb=float(sys.argv[3]); NP=int(sys.argv[4])
ref=ql_reference()
d=run('eqN10',sg,G,nb,0.1,NP,seed=1,stop_m2=1.4*0.5*ref[1][-1])
print(d['nst'],len(d['ts']),d['time'],d['Lw'],d['Dql'],d['Dd'],d['wb']/d['Dd'],flush=True)
print(s_star(d,ref))
for i in range(0,len(d['ts']),max(len(d['ts'])//10,1)): print(round(d['beats'][i],2), d['m2'][i]/d['Lw']**2)
