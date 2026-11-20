"""Cold-plasma dispersion and wave fields for the P2 single-wave study (numpy, no JAX)."""
import numpy as np

e, me, c, eps0, mu0 = 1.602176634e-19, 9.1093837015e-31, 2.99792458e8, 8.8541878128e-12, 4e-7*np.pi
mp = 1.67262192e-27

def species(B0, ne, mi_amu=2.0, Zi=1.0):
    """(plasma freq, signed cyclotron freq) for electrons and one ion species (quasi-neutral)."""
    wpe = np.sqrt(ne*e**2/(eps0*me)); Oe = -e*B0/me
    mi = mi_amu*1.66053906660e-27
    wpi = np.sqrt(ne/Zi*(Zi*e)**2/(eps0*mi)); Oi = Zi*e*B0/mi
    return [(wpe, Oe), (wpi, Oi)]

def stix(w, sp):
    S = 1 - sum(wp**2/(w**2-O**2) for wp, O in sp)
    D = sum((O/w)*wp**2/(w**2-O**2) for wp, O in sp)
    P = 1 - sum(wp**2/w**2 for wp, O in sp)
    return S, D, P

def wavematrix(nperp, npar, S, D, P):
    n2 = nperp**2+npar**2
    return np.array([[S-npar**2, -1j*D, nperp*npar],
                     [1j*D, S-n2, 0],
                     [nperp*npar, 0, P-nperp**2]], dtype=complex)

def solve_kperp(w, kpar, sp, branch="fast"):
    """Roots of det M(n_perp^2)=0 at fixed n_par (quadratic in n_perp^2 fitted from 3 evaluations)."""
    S, D, P = stix(w, sp); npar = kpar*c/w
    xs = np.array([0., 1e3, 2e3])
    ds = [np.linalg.det(wavematrix(np.sqrt(x), npar, S, D, P)).real for x in xs]
    poly = np.polyfit(xs, ds, 2)
    r = np.sort(np.roots(poly).real)
    # fast (helicon / whistler) = the smaller n_perp^2 root; the larger root is the electrostatic slow mode
    return (np.sqrt(r[0])*w/c if branch == "fast" else np.sqrt(r[-1])*w/c), r

def polarization(w, kperp, kpar, sp):
    """Unit-norm complex E eigenvector (x,y,z) and the B vector (units E/c) for plane wave k=(kperp,0,kpar)."""
    S, D, P = stix(w, sp)
    nperp, npar = kperp*c/w, kpar*c/w
    M = wavematrix(nperp, npar, S, D, P)
    uu, ss, vh = np.linalg.svd(M)
    E = vh.conj()[-1]; E = E/np.linalg.norm(E)
    n = np.array([nperp, 0, npar])
    B = np.cross(n, E)                    # B = n x E / c  -> units E/c
    return E, B, ss[-1]/ss[0]

if __name__ == "__main__":
    f, kpar = 476e6, 41.0
    w = 2*np.pi*f
    for ne in (0.5e19, 0.62e19, 0.8e19):
        sp = species(1.4, ne)
        kp, roots = solve_kperp(w, kpar, sp)
        E, B, res = polarization(w, kp, kpar, sp)
        print(f"ne={ne:.2e}: k_perp(fast)={kp:.1f} 1/m, n_perp^2 roots={roots}, |E|/|B c|={1/np.linalg.norm(B):.3f}, resid={res:.1e}")
        print("   E=", np.round(E, 3), " |B|=", np.linalg.norm(B))
    print("Omega_ce/2pi GHz", e*1.4/me/2/np.pi/1e9, " omega", w, " wpe", species(1.4,0.8e19)[0][0])
