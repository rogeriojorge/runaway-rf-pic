"""Gyro-averaged resonant (n=1) reduced model of the N-line problem, phase-exact kick integrator.  State per particle: gamma, u_z, psi_c (phase of the
central line), z.  Line j: psi_j = psi_c + (k_j-k_c) z + phi_j ; d gamma|_j = G_j F_j b_j cos psi_j dt, d u_z|_j = (k_j/omega_n) d gamma|_j
(single-line invariant gamma - (w/k) u_z), d psi_c/dt = -delta_c.  Drives: E_par, friction, LL synchrotron (time-compressed by lam),
Langevin pitch scattering (nuf x DIII-D nu_D).  Validated against the full-orbit Boris integrator (val_red.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3lib import *
from functools import partial
C = pl.c
aB = pl.e**2/(6*np.pi*pl.eps0*pl.me*C**3)*(pl.e*case.B0/pl.me)**2            # RR rate 1/s: du_perp/dt = -aB gamma u_perp

def F_table(l_c, ng=48, nt=64, grange=(14.0, 19.5), trange=(0.0, 0.7)):
    gg = np.linspace(*grange, ng); tt = np.linspace(*trange, nt)
    GG, TT = np.meshgrid(gg, tt, indexing="ij")
    F = frad_vec(l_c, wn, GG.ravel(), TT.ravel()).reshape(ng, nt)
    return gg, tt, F

class Red:
    def __init__(self, ks, bamps):
        self.ks = np.asarray(ks); self.L = len(ks); self.ic = self.L//2
        ls = [case.line(k) for k in ks]; self.ls = ls
        self.kz = jnp.asarray([l["kz"] for l in ls]); self.kzc = float(self.kz[self.ic])
        gc = case.gamma_res(self.kzc, -1, TH0)
        Fc = th.frad(ls[self.ic], wn, gc, TH0)[0]
        self.r = jnp.asarray([th.frad(l, wn, case.gamma_res(l["kz"], -1, TH0), TH0)[0]/Fc for l in ls])
        gg, tt, F = F_table(ls[self.ic])
        self.gg, self.tt, self.F = jnp.asarray(gg), jnp.asarray(tt), jnp.asarray(F)
        self.bamps = jnp.asarray(bamps); self.use_fric = 1.0; self.use_rr = 1.0; self.kappa = 0.0; self.kD = 1.0; self.exactpar = False
        self.dk_m = jnp.asarray((self.ks-self.ks[self.ic]))          # 1/m offsets
    def Fc(self, g, th_):
        ig = jnp.clip((g-self.gg[0])/(self.gg[1]-self.gg[0]), 0, len(self.gg)-1.001); it = jnp.clip((th_-self.tt[0])/(self.tt[1]-self.tt[0]), 0, len(self.tt)-1.001)
        i0, j0 = ig.astype(int), it.astype(int); fx, fy = ig-i0, it-j0
        F = self.F
        return (F[i0, j0]*(1-fx)*(1-fy)+F[i0+1, j0]*fx*(1-fy)+F[i0, j0+1]*(1-fx)*fy+F[i0+1, j0+1]*fx*fy)

    def run(self, g0, uz0, psi0, z0, phis, dt, nsteps, nrec, lam=0.0, nuf=0.0, drive=False, key=0, gcom=0.0, gind=0.0, envf=None, sub=1):
        """dt in seconds. returns final (gamma, uz) and records of gamma every nrec steps."""
        kz, r, bamps, dk = self.kz, self.r, self.bamps, self.dk_m
        Om_, wn_ = Om, wn
        Ecn = case.Ec_n*Om_; E0 = case.EoverEc*Ecn
        nu0 = (case.Zeff+1)*case.nu0*Om_*nuf
        phis = jnp.asarray(phis); L_ = self.L
        def deriv_drive(g, uz):
            p2 = g*g-1.0; p = jnp.sqrt(p2); up2 = jnp.maximum(p2-uz*uz, 1e-12); up = jnp.sqrt(up2)
            rr = self.use_rr*lam*aB*g*up2/p2          # LL: force along p-hat, dp/dt = -aB gamma p (1-xi^2)
            duz = -lam*E0 - self.use_fric*lam*Ecn*(g*g/p2)*uz/p - rr*uz
            dup = -self.use_fric*lam*Ecn*(g*g/p2)*up/p - rr*up
            return duz, dup, up
        def wave_kick(g, uz, psi, z, psi0, z0_, dt_, gE, uzE, phn, ev=1.0):
            """kick over dt_ using coefficients (detunings, couplings) evaluated at state (gE,uzE) and phases at start (psi0,z0_) advanced with those detunings."""
            p2 = gE*gE-1.0; up = jnp.sqrt(jnp.maximum(p2-uzE*uzE, 1e-12)); thp = jnp.arctan2(up, -uzE)
            Fc = self.Fc(gE, thp)
            dl = (wn_ - kz[None, :]*uzE[:, None]/gE[:, None] - 1.0/gE[:, None])*Om_
            ph = psi0[:, None] + dk[None, :]*z0_[:, None] + phis[None, :] + phn
            a_j = wn_/kz[None, :]
            Gj = up[:, None]/(gE[:, None]-uzE[:, None]/a_j)
            cj = Gj*(Fc[:, None]*r[None, :])*bamps[None, :]*Om_*ev
            if self.exactpar:
                Ej = bamps[None, :]*(wn_/kz[None, :])/jnp.sqrt(2.0)
                cj = (up/gE)[:, None]*Ej*Om_*ev
            x = dl*dt_/2
            sinc = jnp.where(jnp.abs(x) < 1e-6, 1.0, jnp.sin(x)/jnp.where(jnp.abs(x) < 1e-6, 1.0, x))
            dgj = cj*dt_*sinc*jnp.cos(ph - x)
            return jnp.sum(dgj, 1), jnp.sum(kz[None, :]/wn_*dgj, 1)
        def kappa_sum(gE, uzE, psiE, zE, phn):
            """Hamiltonian completion of the phase equation: psi_c' += -sum_j kappa_j sin(psi_j), kappa_j = F_j b_j Om (1 + dlnF/dlnu_perp)/u_perp (u_z fixed)."""
            p2 = gE*gE-1.0; up = jnp.sqrt(jnp.maximum(p2-uzE*uzE, 1e-12)); thp = jnp.arctan2(up, -uzE)
            Fc = self.Fc(gE, thp)
            eps = 1e-3; up2_ = up*(1+eps); g2_ = jnp.sqrt(1+up2_**2+uzE**2); th2_ = jnp.arctan2(up2_, -uzE)
            Dl = (self.Fc(g2_, th2_)/Fc-1.0)/eps
            ph = psiE[:, None] + dk[None, :]*zE[:, None] + phis[None, :] + phn
            kj = (Fc*(1.0+self.kD*Dl)*Om_/up)[:, None]*r[None, :]*bamps[None, :]
            if self.exactpar:
                Ej = bamps[None, :]*(wn_/kz[None, :])/jnp.sqrt(2.0)
                kj = Ej*(1.0-kz[None, :]*uzE[:, None]/(wn_*gE[:, None]))*Om_/up[:, None]
            return jnp.sum(kj*jnp.sin(ph), 1)
        def drive_rate(g, uz):
            if not drive: return 0.0*g, 0.0*g
            a1, b1, up = deriv_drive(g, uz)
            return a1, b1, up
        def drive_incr(g, uz, dt_):
            """(dg, duz) over dt_ from the drive at state (g,uz), explicit Euler in (uz,up) (called at the midpoint state)."""
            if not drive: return 0.0, 0.0
            a1, b1, up = deriv_drive(g, uz)
            uz_d = uz+dt_*a1; up_d = up+dt_*b1
            return jnp.sqrt(1+uz_d**2+up_d**2)-g, dt_*a1
        def step(carry, _):
            g, uz, psi, z, key, phn, tt = carry
            e1 = 1.0 if envf is None else envf(tt+dt/4); e2 = 1.0 if envf is None else envf(tt+dt/2)
            # stage 1: half step to the midpoint
            dgw, duzw = wave_kick(g, uz, psi, z, psi, z, dt/2, g, uz, phn, e1)
            dgd, duzd = drive_incr(g, uz, dt/2)
            gh, uzh = g+dgw+dgd, uz+duzw+duzd
            dc0 = (wn_ - kz[self.ic]*uz/g - 1.0/g)*Om_
            dch = (wn_ - kz[self.ic]*uzh/gh - 1.0/gh)*Om_
            psih = psi - 0.5*(dc0)*dt/2 - 0.5*dch*dt/2         # phase advance to mid (trapezoid)
            zh = z + dt/2*C*(0.5*uz/g+0.5*uzh/gh)
            # stage 2: full step with midpoint coefficients; phases referenced at step start
            dgw, duzw = wave_kick(g, uz, psi, z, psi, z, dt, gh, uzh, phn, e2)
            dgd, duzd = drive_incr(gh, uzh, dt)
            gn = g+dgw+dgd; uzn = uz+duzw+duzd
            if nuf != 0.0:
                key, k1 = jax.random.split(key)
                pn = jnp.sqrt(gn*gn-1.0); xi = uzn/pn; s_ = jnp.sqrt(jnp.maximum(1-xi*xi, 0.0))
                nu = nu0*gn/pn**3
                e = jax.random.normal(k1, (g.shape[0], 2))*jnp.sqrt(nu*dt)[:, None]
                xin = (xi - e[:, 0]*s_)/jnp.sqrt(1+e[:, 0]**2+e[:, 1]**2)
                uzn = pn*jnp.clip(xin, -1, 1)
            dcm = (wn_ - kz[self.ic]*uzh/gh - 1.0/gh)*Om_
            psin = psi - dcm*dt
            if self.kappa != 0.0:
                psin = psin - self.kappa*dt*e2*kappa_sum(gh, uzh, psih, zh, phn)
            zn = z + dt*C*(uzh/gh)
            phn_n = phn
            if gcom != 0.0 or gind != 0.0:
                key, k2 = jax.random.split(key)
                nz = jax.random.normal(k2, (g.shape[0], 1+L_))
                psin = psin + jnp.sqrt(2*gcom*dt)*nz[:, 0]
                if gind != 0.0: phn_n = phn + jnp.sqrt(2*gind*dt)*nz[:, 1:]
            return (gn, uzn, psin, zn, key, phn_n, tt+dt), None
        def block(carry, _):
            carry, _ = jax.lax.scan(step, carry, None, length=nrec)
            return carry, jnp.stack([carry[0], carry[1]], -1)
        carry0 = (jnp.asarray(g0), jnp.asarray(uz0), jnp.asarray(psi0), jnp.asarray(z0), jax.random.PRNGKey(key), jnp.zeros((len(g0), self.L)), 0.0)
        carry, recs = jax.lax.scan(block, carry0, None, length=nsteps//nrec)
        return carry, recs
