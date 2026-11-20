"""Step 2: helicon (fast-wave) damping at DIII-D QRE conditions and the P -> b map.

Self-contained (numpy/scipy/matplotlib).  Run:  python3 helicon_damping.py   (about 1 minute, 1 thread)
Writes ../data/step2.json and ../fig*.png.

Conventions: fields ~ exp(i(k.x - w t)), B0 || z, k = (k_perp, 0, k_par).  SI units.
Wave energy density (cold, time average):  w = eps0/4 E^H d(wK)/dw E + |B|^2/(4 mu0),  B = k x E / w.
Power damping rate (energy loss rate)  Gamma = P_abs / w.   Gamma_P of the PLAN/P3 notes = this Gamma.
"""
import json, os, sys
import numpy as np
from scipy.special import wofz, jv

eps0 = 8.8541878128e-12; mu0 = 4e-7*np.pi; c = 2.99792458e8
qe = 1.602176634e-19; me = 9.1093837015e-31; mD = 3.3435837724e-27; mC = 1.9926e-26
F_RF = 476e6; W = 2*np.pi*F_RF
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ---------------------------------------------------------------- cold plasma with e-i collisions
def species(ne, Zeff):
    """electrons, D, C(6+) with Zeff = (nD + 36 nC)/ne,  nD + 6 nC = ne."""
    nC = (Zeff-1)/30*ne; nD = ne-6*nC
    return [(-1, me, ne), (1, mD, nD), (6, mC, nC)]

def stix(w, B, ne, Zeff, nu=0.0):
    R = 1+0j; L = 1+0j; P = 1+0j
    for Z, m, n in species(ne, Zeff):
        wp2 = n*(Z*qe)**2/(eps0*m); Om = Z*qe*B/m * (-1 if Z < 0 else 1)   # signed
        we = w+1j*nu if Z < 0 else w
        R -= wp2/(w*(we+Om)); L -= wp2/(w*(we-Om)); P -= wp2/(w*we)
    return 0.5*(R+L), 0.5*(R-L), P

def Kmat(w, B, ne, Zeff, nu=0.0):
    S, D, P = stix(w, B, ne, Zeff, nu)
    return np.array([[S, -1j*D, 0], [1j*D, S, 0], [0, 0, P]])

def Mmat(w, B, ne, Zeff, kx, kz, nu=0.0):
    K = Kmat(w, B, ne, Zeff, nu); k0 = w/c
    kk = np.array([[kz*kz, 0, -kx*kz], [0, kx*kx+kz*kz, 0], [-kx*kz, 0, kx*kx]])/k0**2   # (nn - n^2 I) with sign: n^2 I - nn
    n2 = (kx*kx+kz*kz)/k0**2
    return K - n2*np.eye(3) + np.array([[kx*kx, 0, kx*kz], [0, 0, 0], [kx*kz, 0, kz*kz]])/k0**2

def fast_kperp(w, B, ne, Zeff, kz):
    """largest-k_perp root of the cold dispersion relation (fast wave); None if evanescent."""
    S, D, P = [x.real for x in stix(w, B, ne, Zeff)]
    a = (kz*c/w)**2
    c2 = S; c1 = -(S-a)**2-(S-a)*P+D**2-a*(S-a); c0 = ((S-a)**2-D**2)*P
    r = np.roots([c2, c1, c0]); r = r[np.isreal(r)].real
    r = r[r > 0]
    if len(r) == 0: return None
    return np.sqrt(r.min())*w/c

def slow_kperp(w, B, ne, Zeff, kz):
    S, D, P = [x.real for x in stix(w, B, ne, Zeff)]
    a = (kz*c/w)**2
    r = np.roots([S, -(S-a)**2-(S-a)*P+D**2-a*(S-a), ((S-a)**2-D**2)*P]); r = r[np.isreal(r)].real
    return np.sqrt(r[r > 0].max())*w/c

def null_vec(M):
    # null vector of M via SVD
    u, s, vh = np.linalg.svd(M)
    return vh[-1].conj(), s

def wave_energy(w, B, ne, Zeff, kx, kz, E):
    h = 1e-4*w
    Kp = Kmat(w+h, B, ne, Zeff); Km = Kmat(w-h, B, ne, Zeff)
    dwK = ((w+h)*Kp-(w-h)*Km)/(2*h)
    Wc = 0.25*eps0*np.real(E.conj()@dwK@E)
    Bv = np.cross(np.array([kx, 0, kz]), E)/w
    Wb = np.real(Bv.conj()@Bv)/(4*mu0)
    return Wc+Wb, Wc, Wb, Bv

def det_at(w, B, ne, Zeff, kx, kz, nu=0.0):
    return np.linalg.det(Mmat(w, B, ne, Zeff, kx, kz, nu))

def group_velocity(w, B, ne, Zeff, kx, kz):
    """v_g = -grad_k D / dD/dw with D = det M (cold)."""
    d = lambda f, x, h: (f(x+h)-f(x-h))/(2*h)
    Dw = d(lambda ww: det_at(ww, B, ne, Zeff, kx, kz).real, w, 1e-5*w)
    Dkx = d(lambda k: det_at(w, B, ne, Zeff, k, kz).real, kx, 1e-5*kx)
    Dkz = d(lambda k: det_at(w, B, ne, Zeff, kx, k).real, kz, 1e-5*kz)
    return -Dkx/Dw, -Dkz/Dw

# ---------------------------------------------------------------- collisions
def coulomb_log(ne, Te):
    # NRL, Te > 10 Z^2 eV
    return 24-np.log(np.sqrt(ne*1e-6)/Te) if Te > 10 else 23-np.log(np.sqrt(ne*1e-6)*Te**-1.5)

def nu_ei(ne, Te, Zeff, lnL=None):
    """Spitzer-type e-i collision frequency, PPCF 68 105047 eq. B9: 4 sqrt(2 pi) e^4 lnL Sum Z^2 n_Z / (3 m^1/2 T^3/2) (Gaussian)
    = 4 sqrt(2 pi) ne Zeff e^4 lnL / (3 (4 pi eps0)^2 sqrt(me) T^1.5)  (SI)."""
    lnL = coulomb_log(ne, Te) if lnL is None else lnL
    T = Te*qe
    return 4*np.sqrt(2*np.pi)*ne*Zeff*qe**4*lnL/(3*(4*np.pi*eps0)**2*np.sqrt(me)*T**1.5)

def gamma_coll(B, ne, Te, Zeff, kz, w=W):
    """collisional power damping rate, two ways: (a) anti-Hermitian part of K, (b) complex-omega root."""
    kx = fast_kperp(w, B, ne, Zeff, kz)
    if kx is None: return None
    nu = nu_ei(ne, Te, Zeff)
    E, s = null_vec(Mmat(w, B, ne, Zeff, kx, kz))
    K = Kmat(w, B, ne, Zeff, nu)
    Ka = (K-K.conj().T)/(2j)
    P = 0.5*eps0*w*np.real(E.conj()@Ka@E)
    Wt = wave_energy(w, B, ne, Zeff, kx, kz, E)[0]
    ga = P/Wt
    # (b) Newton in complex omega
    wc = w+0j
    for _ in range(30):
        f = det_at(wc, B, ne, Zeff, kx, kz, nu); h = 1e-6*w
        df = (det_at(wc+h, B, ne, Zeff, kx, kz, nu)-det_at(wc-h, B, ne, Zeff, kx, kz, nu))/(2*h)
        dw = -f/df; wc += dw
        if abs(dw) < 1e-9*w: break
    return ga, -2*wc.imag, nu

# ---------------------------------------------------------------- Landau + TTMP (n = 0, Maxwellian electrons, k_perp rho_e << 1)
def gamma_landau(B, ne, Te, Zeff, kz, w=W):
    """P_abs = w ne sqrt(pi) zeta exp(-zeta^2) <|F|^2> /(me kz^2 vt^2), F = e Ez + i kz mu Bz  (parallel force on the n=0 resonant
    electron: Landau + transit-time magnetic pumping), <mu> = T/B, <mu^2> = 2 T^2/B^2 over the perpendicular Maxwellian.
    Equivalent to Im chi_zz of the Maxwellian plasma with e^2|E|^2 -> <|F|^2>."""
    kx = fast_kperp(w, B, ne, Zeff, kz)
    if kx is None: return None
    E, s = null_vec(Mmat(w, B, ne, Zeff, kx, kz))
    Wt, Wc, Wb, Bv = wave_energy(w, B, ne, Zeff, kx, kz, E)
    T = Te*qe; vt2 = 2*T/me; zeta = w/(abs(kz)*np.sqrt(vt2))
    Ez, Bz = E[2], Bv[2]
    F2 = (qe*abs(Ez))**2 + 2*qe*kz*(T/B)*np.real(1j*np.conj(Ez)*Bz) + 2*kz**2*(T/B)**2*abs(Bz)**2
    F2_landau = (qe*abs(Ez))**2; F2_ttmp = 2*kz**2*(T/B)**2*abs(Bz)**2
    pref = w*ne*np.sqrt(np.pi)*zeta*np.exp(-zeta**2)/(me*kz**2*vt2)
    return pref*F2/Wt, pref*F2_landau/Wt, pref*F2_ttmp/Wt, zeta

# ---------------------------------------------------------------- runaway absorption (quasilinear, n = +1 normal Doppler)
def gamma_RE(B, ne, Zeff, kz, n_res, gam=16.0, theta=0.2, Delta=0.57, Neff=11.3, w=W):
    """Quasilinear power-absorption rate of the wave by n_res [m^-3] counter-propagating runaways of Lorentz factor gam and
    pitch angle theta, spectrum of lines spaced Delta [1/m] with N_eff lines of equal peak amplitude.
    Per electron:  dE/dt = pi c^2 e^2 |psi|^2 eta / (E Delta |v_par|),  eta = 1 - kz v_par/w,
    psi = (Ex + i s Ey) J1(z)/z + i... (Pokol 2008 eq 12, n=1, with J_n'), z = k_perp p_perp/(me Omega gam).
    |psi| of the line normalised to the line's own energy density (cold fast-wave eigenvector)."""
    kx = fast_kperp(w, B, ne, Zeff, kz)
    if kx is None: return None
    E, s = null_vec(Mmat(w, B, ne, Zeff, kx, kz))
    Wt = wave_energy(w, B, ne, Zeff, kx, kz, E)[0]
    beta = np.sqrt(1-1/gam**2); v = beta*c
    vpar = -v*np.cos(theta); vperp = v*np.sin(theta)         # counter-propagating to the wave (k.v < 0, "normal")
    Om = qe*B/me; z = kx*gam*vperp/Om
    J1 = jv(1, z); J1p = 0.5*(jv(0, z)-jv(2, z))
    best = 0
    for sg in (+1, -1):                                       # sign of the co-rotating circular component
        psi = E[0]*J1/z + 1j*sg*E[1]*J1p + (abs(vpar)/vperp*np.sign(vpar))*E[2]*J1
        best = max(best, abs(psi)**2)
    eta = 1-kz*vpar/w
    Edot = np.pi*c**2*qe**2*best*eta/(gam*me*c**2*Delta*abs(vpar))     # W per electron per |E|^2 of the line (E normalised to |E|=1)
    # energy density of one line per |E|^2:  Wt / (E^H E)
    wl = Wt/np.real(E.conj()@E)
    return n_res*Edot*np.real(E.conj()@E)/(Neff*Wt), dict(psi2_over_E2=best/np.real(E.conj()@E), eta=eta, z=z, w_per_E2=wl)

# ---------------------------------------------------------------- profile average (energy density taken uniform in the wave region)
def profile_avg(n0, T0, B, Zeff, kc, an=0.5, aT=1.5, Tedge=50.0, nedge_frac=0.05, nrho=40, with_land=True):
    rho = np.linspace(0.0, 0.95, nrho)
    out = dict(coll=[], land=[], ttmp=[], vol=[])
    for r in rho:
        sh = 1-r**2
        n = n0*(nedge_frac+(1-nedge_frac)*sh**an); T = Tedge+(T0-Tedge)*sh**aT
        kz = 30.0+(kc-30.0)*sh
        gc = gamma_coll(B, n, T, Zeff, kz)
        if gc is None: continue
        gl = gamma_landau(B, n, T, Zeff, kz)
        out['coll'].append(gc[0]); out['land'].append(gl[1]); out['ttmp'].append(gl[2]); out['vol'].append(r)
    r = np.array(out['vol']); wts = r
    f = lambda k: np.trapezoid(np.array(out[k])*wts, r)/np.trapezoid(wts, r)
    return dict(coll=f('coll'), land=f('land'), ttmp=f('ttmp'), rho_cut=r.max() if len(r) else 0.0, rho=r, tot=np.array(out['coll'])+np.array(out['land'])+np.array(out['ttmp']))

# ---------------------------------------------------------------- edge / antenna loss (cavity picture)
def gamma_edge(vg, ell, d_gap, f_ant, eps_wall, c_conv, k_edge=30.0):
    """per edge hit: loss probability = f_ant*T_gap (re-absorbed by the antenna, reciprocity of the coupling) + eps_wall + c_conv
    (conversion to the slow wave, then Landau-damped).  T_gap = exp(-2 kappa d)  (WKB tunnelling through the vacuum gap,
    kappa = sqrt(k_par^2 - k0^2), k_par at the edge = 30/m).  Rate = loss per hit / (ell/vg)."""
    k0 = W/c; kap = np.sqrt(max(k_edge**2-k0**2, 0)); T = np.exp(-2*kap*d_gap)
    p = f_ant*T+eps_wall+c_conv
    return p*abs(vg)/ell, p, T

# ---------------------------------------------------------------- array factor
def array_factor(N=30, span_deg=40.0, R=2.27, nphi=np.arange(0, 400), taper="uniform", edge=1.0, ndr=3.0):
    dphi = np.deg2rad(span_deg)/N          # module pitch (rad)
    k0 = W/c; psi = ndr*k0*R*dphi           # phase advance per module for n_par = ndr at the antenna
    m = np.arange(N)-(N-1)/2
    if taper == "uniform": a = np.ones(N)
    elif taper == "hann": a = 0.5*(1+np.cos(np.pi*m/(N/2)))
    elif taper == "cosine_pedestal": a = edge+(1-edge)*np.cos(np.pi*m/N)
    A = np.array([np.sum(a*np.exp(1j*(n*dphi-psi)*np.arange(N))) for n in nphi])
    P = np.abs(A)**2
    return dphi, psi, nphi, P

def lobe_stats(nphi, P, period):
    Pm = P.max(); n0 = nphi[P.argmax()]
    above = nphi[P >= 0.5*Pm]; fwhm = (above.max()-above.min()+1) if len(above) else np.nan
    Neff = period*np.mean(P)/Pm       # Parseval: sum over one period / peak
    return n0, fwhm, Neff

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    T0LO = float(os.environ.get('T0LO', 2000)); TAG = os.environ.get('TAG', '')
    res = {}
    # ---- 1. dispersion check against the P2/P3 notes and PPCF Appendix B
    chk = {}
    for lab, B, ne, kz in [("P2/P3: n=0.8e19, B=1.4, kpar=41", 1.4, 0.8e19, 41.0),
                           ("PPCF B: n=0.5e19, B=1.4, n_par=3 (kpar=29.9)", 1.4, 0.5e19, 3*W/c)]:
        kx = fast_kperp(W, B, ne, 1.5, kz)
        vg = group_velocity(W, B, ne, 1.5, kx, kz)
        E, s = null_vec(Mmat(W, B, ne, 1.5, kx, kz)); Wt, Wc, Wb, Bv = wave_energy(W, B, ne, 1.5, kx, kz, E)
        chk[lab] = dict(kperp=kx, vg_perp_over_c=vg[0]/c, vg_par_over_c=vg[1]/c, W_over_WB=Wt/Wb, Ez_over_Ex=abs(E[2]/E[0]))
        print(lab, chk[lab])
    res['dispersion_check'] = chk
    # slow-wave root for reference (PPCF: k_perp = 1360/m)
    res['slow_kperp_check'] = {str(Z): float(slow_kperp(W, 1.4, 0.5e19, Z, 3*W/c)) for Z in (1.0, 1.5)}
    print('slow-wave k_perp (PPCF App. B quotes 1360)', res['slow_kperp_check'])

    # ---- 2. local damping vs T_e (validation of coll two ways; scalings)
    Tgrid = np.array([2, 5, 10, 20, 50, 100, 300, 500, 1000, 1500, 2165, 3000.])
    loc = {}
    for lab, ne, kz in [("n0.8e19_kz41", 0.8e19, 41.0), ("n0.5e19_kz30", 0.5e19, 30.0)]:
        rows = []
        for T in Tgrid:
            gc = gamma_coll(1.4, ne, T, 1.5, kz); gl = gamma_landau(1.4, ne, T, 1.5, kz)
            rows.append(dict(Te=T, nu_ei=gc[2], gam_coll_AH=gc[0], gam_coll_root=gc[1], gam_land_ttmp=gl[0], gam_land=gl[1], gam_ttmp=gl[2], zeta=gl[3]))
        loc[lab] = rows
    res['local'] = loc
    print("local, n=0.8e19 kz=41:")
    for r in loc["n0.8e19_kz41"]: print("  Te=%6.0f nu=%.2e Gc(AH)=%.3e Gc(root)=%.3e Gland+ttmp=%.3e zeta=%.2f" % (r['Te'], r['nu_ei'], r['gam_coll_AH'], r['gam_coll_root'], r['gam_land_ttmp'], r['zeta']))

    # ---- 3. central profile-averaged values
    cen = profile_avg(0.65e19, 2165, 1.45, 1.5, 41.0)
    # core-only
    gl0 = gamma_landau(1.45, 0.65e19, 2165, 1.5, 41.0); gc0 = gamma_coll(1.45, 0.65e19, 2165, 1.5, 41.0)
    res['central_profile_avg'] = cen; res['central_core'] = dict(coll=gc0[0], land_ttmp=gl0[0], zeta=gl0[3])
    print('central profile average', cen, 'core', res['central_core'])

    # transit and single-pass absorption
    kx0 = fast_kperp(W, 1.45, 0.65e19, 1.5, 41.0); vg0 = group_velocity(W, 1.45, 0.65e19, 1.5, kx0, 41.0)
    vg_abs = np.hypot(*vg0)
    res['vg_core'] = dict(vperp_c=vg0[0]/c, vpar_c=vg0[1]/c, vabs_c=vg_abs/c)
    print('vg core (perp,par,|.|)/c', vg0[0]/c, vg0[1]/c, vg_abs/c)

    # ---- 4. runaway absorption: normalisation, scaling with n_res
    gr, info = gamma_RE(1.45, 0.65e19, 1.5, 41.0, 1e15)
    res['RE_per_1e15'] = dict(gamma=gr, **info)
    print('Gamma_RE per n_res=1e15 m^-3:', gr, info)
    # energy-balance reference: I_RE V_loop
    # ---- 5. Monte Carlo over the parameter ranges
    NMC = 300
    b_thr_c = 1.0e-4; mu = 4*mu0
    def samp(nonideal):
        n0 = rng.uniform(0.5, 0.8)*1e19; T0 = rng.uniform(T0LO, 3000); Z = rng.uniform(1.2, 2.0)
        B = rng.uniform(1.40, 1.70); kc = rng.uniform(36, 44.0)
        an = rng.uniform(0.3, 1.0); aT = rng.uniform(1.0, 2.0)
        return n0, T0, Z, B, kc, an, aT
    store = []
    for i in range(NMC):
        n0, T0, Z, B, kc, an, aT = samp(False)
        pa = profile_avg(n0, T0, B, Z, kc, an, aT, nrho=14)
        kx = fast_kperp(W, B, n0, Z, kc); vg = group_velocity(W, B, n0, Z, kx, kc); vabs = np.hypot(*vg)
        ell = rng.uniform(1.2, 3.0)                                   # path length between edge reflections [m]
        f_ant = np.exp(rng.uniform(np.log(2e-3), np.log(5e-2)))
        eps_w = np.exp(rng.uniform(np.log(1e-5), np.log(1e-3)))
        c_conv = np.exp(rng.uniform(np.log(1e-4), np.log(1e-2)))
        d_ideal, d_non = 0.045, 0.09
        Vp = rng.uniform(15, 22)
        rr = np.linspace(0, 1, 201); sx = np.abs(2*rr-1)*0.95
        Achord = float(np.trapezoid(np.interp(sx, pa['rho'], pa['tot']), rr*ell/vabs))   # tau-integral of Gamma along a chord through the axis
        Neff = rng.choice([9.0, 11.3, 13.5])
        bthr = np.exp(rng.normal(np.log(1.0e-4), (np.log(1.6)-np.log(0.6))/(2*1.5)))   # 0.6-1.6e-4 taken as ~ +-1.5 sigma (lognormal)
        eg_i = gamma_edge(vabs, ell, d_ideal, f_ant, eps_w, 0.0)[0]; eg_n = gamma_edge(vabs, ell, d_non, f_ant, eps_w, 0.0)[0]
        egc_i = gamma_edge(vabs, ell, d_ideal, f_ant, eps_w, c_conv)[0]; egc_n = gamma_edge(vabs, ell, d_non, f_ant, eps_w, c_conv)[0]
        store.append(dict(n0=n0, T0=T0, Z=Z, B=B, kc=kc, Vp=Vp, Neff=Neff, bthr=bthr, coll=pa['coll'], land=pa['land'], ttmp=pa['ttmp'],
                          vabs=vabs, ell=ell, edge_ideal=eg_i, edge_non=eg_n, edgec_ideal=egc_i, edgec_non=egc_n, rho_cut=pa['rho_cut'], f_ant=f_ant, Achord=Achord))
    S_ = {k: np.array([s[k] for s in store]) for k in store[0]}
    np.savez(os.path.join(OUT, "data", "mc_samples"+TAG+".npz"), **S_)
    q = lambda x: [float(np.percentile(x, p)) for p in (2.5, 16, 50, 84, 97.5)]
    summ = {}
    for k in ("Achord", "coll", "land", "ttmp", "edge_ideal", "edge_non", "edgec_ideal", "edgec_non", "vabs"):
        summ[k] = q(S_[k])
    thermal = S_['coll']+S_['land']+S_['ttmp']
    summ['thermal'] = q(thermal)
    for sc, key in (("A_no_conv_non_ideal", "edge_non"), ("A_no_conv_ideal", "edge_ideal"), ("B_conv_non_ideal", "edgec_non"), ("B_conv_ideal", "edgec_ideal")):
        summ['Gamma_total_'+sc] = q(thermal+S_[key])
    res['mc_summary'] = summ
    print({k: ["%.2e" % v for v in x] for k, x in summ.items()})

    # ---- 6. P -> b map and probability of being above threshold
    shots = {"201951 non-ideal pulsed (43 kW)": (43e3, "non"), "201952 non-ideal cw (76 kW)": (76e3, "non"),
             "201953 non-ideal (70 kW)": (70e3, "non"), "ideal 201950 (120 kW)": (120e3, "ideal"), "ideal 201948 (130 kW)": (130e3, "ideal"),
             "150 kW (PRL maximum)": (150e3, "ideal"), "GENRAY 160 kW (upper bound)": (160e3, "ideal")}
    maps = {}
    for sc in ("A", "B"):
        for lab, (P, kind) in shots.items():
            key = ("edge" if sc == "A" else "edgec")+"_"+("non" if kind == "non" else "ideal")
            G = thermal+S_[key]
            utot = P/(G*S_['Vp'])                                   # J/m^3 ; u_w = b^2 B^2/(4 mu0)
            btot = np.sqrt(utot*mu/S_['B']**2)
            bthr_tot = S_['bthr']*np.sqrt(S_['Neff'])
            ratio = btot/bthr_tot
            maps[sc+" | "+lab] = dict(b_tot=q(btot), ratio=q(ratio), p_above=float(np.mean(ratio > 1)), p_above_half=float(np.mean(ratio > 0.5)),
                                      p_within_factor2=float(np.mean((ratio > 0.5) & (ratio < 2))))
    res['maps'] = maps
    for k, v in maps.items(): print("%-52s b_tot[%.1e..%.1e..%.1e]  ratio med %.2f  P(>thr)=%.2f  P(within x2)=%.2f" % (k, v['b_tot'][0], v['b_tot'][2], v['b_tot'][4], v['ratio'][2], v['p_above'], v['p_within_factor2']))
    # power threshold distribution
    Pthr = {}
    for sc in ("A", "B"):
        for kind in ("non", "ideal"):
            key = ("edge" if sc == "A" else "edgec")+"_"+kind
            G = thermal+S_[key]; bt = S_['bthr']*np.sqrt(S_['Neff'])
            Pt = G*S_['Vp']*bt**2*S_['B']**2/mu
            Pthr[sc+"_"+kind] = q(Pt)
    pp = {}
    for sc in ("A", "B"):
        key = ("edge" if sc == "A" else "edgec")+"_non"; G = thermal+S_[key]; bt = S_['bthr']*np.sqrt(S_['Neff'])
        Pt = G*S_['Vp']*bt**2*S_['B']**2/mu
        pp[sc] = dict(p_lt_43=float(np.mean(Pt < 43e3)), p_43_76=float(np.mean((Pt >= 43e3) & (Pt < 76e3))), p_76_150=float(np.mean((Pt >= 76e3) & (Pt < 150e3))), p_gt_150=float(np.mean(Pt >= 150e3)))
    res['Pthr_bins_nonideal'] = pp; print('P_thr bins', pp)
    res['single_pass_chord'] = dict(Achord_q=q(S_['Achord']), p_gt_0p1=float(np.mean(S_['Achord'] > 0.1)), p_gt_1=float(np.mean(S_['Achord'] > 1.0)))
    print('chord absorption (thermal, one core chord)', res['single_pass_chord'])
    reT = {}
    for th in (0.1, 0.2, 0.3):
        for gm in (12., 16., 20.):
            reT["theta%.1f_gam%.0f" % (th, gm)] = float(gamma_RE(1.45, 0.65e19, 1.5, 41.0, 1e15, gam=gm, theta=th)[0])
    res['RE_table_per_1e15'] = reT; print('RE table', reT)
    res['P_thr_kW'] = {k: [x/1e3 for x in v] for k, v in Pthr.items()}
    print({k: ["%.0f" % (x/1e3) for x in v] for k, v in Pthr.items()})
    # reciprocal: Gamma needed
    # ---- 7. energy balance reference and RE sink
    res['IRE_Vloop_kW'] = {str(I): I*1e3*0.055*2*np.pi*1.75/1e3 for I in (50, 100, 200)}
    # ---- 8. array factor
    af = {}
    for tp, kw in (("uniform", {}), ("hann", {}), ("cosine_pedestal_0.3", dict(taper="cosine_pedestal", edge=0.3)), ("cosine_pedestal_0.6", dict(taper="cosine_pedestal", edge=0.6))):
        t = kw.pop("taper", tp); dphi, psi, nphi, P = array_factor(taper=t, **kw)
        period = 2*np.pi/dphi
        n0, fwhm, Neff = lobe_stats(nphi, P, period)
        nfull = np.arange(0, int(period)); P2 = array_factor(taper=t, nphi=nfull, **kw)[3]
        n0, fwhm, Neff = lobe_stats(nfull, P2, period)
        af[tp] = dict(n0=int(n0), n_par=float(n0/(W/c*2.27)), fwhm_lines=float(fwhm), Neff=float(Neff), sigma_equiv=float(fwhm/2.3548))
    dphi, psi, _, _ = array_factor()
    res['array'] = dict(module_pitch_deg=float(np.rad2deg(dphi)), pitch_m=float(2.27*dphi), phase_per_module_deg=float(np.rad2deg(psi)),
                        period_lines=float(2*np.pi/dphi), by_taper=af,
                        genray=dict(sigma_n_par=0.2, sigma_lines=0.2*W/c*2.27, Neff=float(0.2*W/c*2.27*np.sqrt(2*np.pi))))
    print(json.dumps(res['array'], indent=1))
    res['b_tot_thr'] = {k: float(1e-4*np.sqrt(v['Neff'])) for k, v in af.items()}
    res['b_tot_thr']['genray_gauss'] = float(1e-4*np.sqrt(res['array']['genray']['Neff']))
    res['spacing'] = dict(antenna_R2p27=1/2.27, resonant_R1p75=1/1.75, bthr_scale_if_antenna_radius=(1/2.27/(1/1.75))**2)
    print(res['b_tot_thr'], res['spacing'])
    json.dump(res, open(os.path.join(OUT, "data", "step2"+TAG+".json"), "w"), indent=1, default=float)

    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, ls in (("n0.8e19_kz41", "-"), ("n0.5e19_kz30", "--")):
        rows = loc[lab]; T = [r['Te'] for r in rows]
        ax[0].loglog(T, [r['gam_coll_AH'] for r in rows], 'C0'+ls, label='collisional '+lab)
        ax[0].loglog(T, [r['gam_land_ttmp'] for r in rows], 'C1'+ls, label='Landau+TTMP '+lab)
    ax[0].set_xlabel('T_e [eV]'); ax[0].set_ylabel('Gamma_P (power) [1/s]'); ax[0].legend(fontsize=7); ax[0].set_ylim(1e-1, 1e9); ax[0].set_title('local damping rate, B=1.4 T, Zeff=1.5')
    ax[0].axhspan(1.4e4, 3.5e5, color='gray', alpha=0.2); ax[0].text(3, 6e4, 'needed band 1.4e4-3.5e5', fontsize=7)
    ax[1].hist(np.log10(thermal), bins=25, alpha=.6, label='thermal (coll+Landau+TTMP)')
    ax[1].hist(np.log10(thermal+S_['edge_non']), bins=25, alpha=.6, label='+ edge/antenna (no conversion)')
    ax[1].hist(np.log10(thermal+S_['edgec_non']), bins=25, alpha=.6, label='+ edge incl. slow-wave conversion')
    ax[1].axvspan(np.log10(1.4e4), np.log10(3.5e5), color='gray', alpha=0.2)
    ax[1].set_xlabel('log10 Gamma_P [1/s]'); ax[1].legend(fontsize=7); ax[1].set_title('Monte Carlo (300 draws), volume-averaged')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_damping"+TAG+".png"), dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tp in ("uniform", "hann"):
        nf = np.arange(0, 270); P2 = array_factor(taper=tp, nphi=nf)[3]
        ax[0].semilogy(nf-af[tp]['n0'], P2/P2.max(), label=tp)
    xx = np.arange(-30, 31); ax[0].semilogy(xx, np.exp(-xx**2/(2*4.5**2)), 'k--', label='GENRAY Gaussian sigma=4.5 lines')
    ax[0].set_xlim(-30, 30); ax[0].set_ylim(1e-4, 1.5); ax[0].set_xlabel('n_phi - n_0 (lines)'); ax[0].set_ylabel('power spectrum'); ax[0].legend(fontsize=7)
    ax[0].set_title('30-module array factor, 90 deg phasing')
    # P -> b
    Ps = np.logspace(np.log10(5e3), np.log10(5e5), 60)
    for sc, col in (("A", "C0"), ("B", "C3")):
        key = ("edge" if sc == "A" else "edgec")+"_non"; G = thermal+S_[key]
        bl = [np.percentile(np.sqrt(P/(G*S_['Vp'])*mu/S_['B']**2), [16, 50, 84]) for P in Ps]; bl = np.array(bl)
        ax[1].fill_between(Ps/1e3, bl[:, 0], bl[:, 2], color=col, alpha=.25); ax[1].loglog(Ps/1e3, bl[:, 1], col, label='scenario '+sc+(' (no conversion)' if sc == 'A' else ' (with conversion)'))
    bt = S_['bthr']*np.sqrt(S_['Neff']); ax[1].axhspan(np.percentile(bt, 16), np.percentile(bt, 84), color='k', alpha=.15)
    ax[1].axhline(3.4e-4, color='k', ls='--', lw=.8)
    for P in (43, 76, 150): ax[1].axvline(P, color='gray', ls=':')
    ax[1].set_xlabel('coupled power [kW]'); ax[1].set_ylabel('b_tot = dB/B0 (rss over lines)'); ax[1].legend(fontsize=7); ax[1].set_title('non-ideal gap; band = 16-84%; grey = threshold')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_array_and_map"+TAG+".png"), dpi=110)
