"""Reduced resonant model with energy accounting: identical integrator to results/P3/scripts/red.py (imported unchanged for the table and setup), run()
additionally accumulates the wave work Ew = sum d(gamma)_wave and the drive work Ed = sum d(gamma)_(E, friction, LL) per particle and records
(gamma, u_z, Ew, Ed) every nrec steps."""
import sys, os
sys.path.insert(0, os.path.expanduser("results/P3/scripts"))
from red import *
class RedE(Red):
    def run(self, g0, uz0, psi0, z0, phis, dt, nsteps, nrec, lam=0.0, nuf=0.0, drive=False, key=0, gcom=0.0, gind=0.0):
        """dt in seconds. returns final (gamma, uz) and records of gamma every nrec steps."""
        kz, r, bamps, dk = self.kz, self.r, self.bamps, self.dk_m
        Om_, wn_ = Om, wn
        Ecn = case.Ec_n*Om_; E0 = case.EoverEc*Ecn
        nu0 = (case.Zeff+1)*case.nu0*Om_*nuf
        phis = jnp.asarray(phis); L_ = self.L
        def deriv_drive(g, uz):
            p2 = g*g-1.0; p = jnp.sqrt(p2); up2 = jnp.maximum(p2-uz*uz, 1e-12); up = jnp.sqrt(up2)
            rr = lam*aB*g*up2/p2          # LL: force along p-hat, dp/dt = -aB gamma p (1-xi^2)
            duz = -lam*E0 - lam*Ecn*(g*g/p2)*uz/p - rr*uz
            dup = -lam*Ecn*(g*g/p2)*up/p - rr*up
            return duz, dup, up
        def wave_kick(g, uz, psi, z, psi0, z0_, dt_, gE, uzE, phn):
            """kick over dt_ using coefficients (detunings, couplings) evaluated at state (gE,uzE) and phases at start (psi0,z0_) advanced with those detunings."""
            p2 = gE*gE-1.0; up = jnp.sqrt(jnp.maximum(p2-uzE*uzE, 1e-12)); thp = jnp.arctan2(up, -uzE)
            Fc = self.Fc(gE, thp)
            dl = (wn_ - kz[None, :]*uzE[:, None]/gE[:, None] - 1.0/gE[:, None])*Om_
            ph = psi0[:, None] + dk[None, :]*z0_[:, None] + phis[None, :] + phn
            a_j = wn_/kz[None, :]
            Gj = up[:, None]/(gE[:, None]-uzE[:, None]/a_j)
            cj = Gj*(Fc[:, None]*r[None, :])*bamps[None, :]*Om_
            x = dl*dt_/2
            sinc = jnp.where(jnp.abs(x) < 1e-6, 1.0, jnp.sin(x)/jnp.where(jnp.abs(x) < 1e-6, 1.0, x))
            dgj = cj*dt_*sinc*jnp.cos(ph - x)
            return jnp.sum(dgj, 1), jnp.sum(kz[None, :]/wn_*dgj, 1)
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
            g, uz, psi, z, key, phn, Ew, Ed = carry
            # stage 1: half step to the midpoint
            dgw, duzw = wave_kick(g, uz, psi, z, psi, z, dt/2, g, uz, phn)
            dgd, duzd = drive_incr(g, uz, dt/2)
            gh, uzh = g+dgw+dgd, uz+duzw+duzd
            dc0 = (wn_ - kz[self.ic]*uz/g - 1.0/g)*Om_
            dch = (wn_ - kz[self.ic]*uzh/gh - 1.0/gh)*Om_
            psih = psi - 0.5*(dc0)*dt/2 - 0.5*dch*dt/2         # phase advance to mid (trapezoid)
            zh = z + dt/2*C*(0.5*uz/g+0.5*uzh/gh)
            # stage 2: full step with midpoint coefficients; phases referenced at step start
            dgw, duzw = wave_kick(g, uz, psi, z, psi, z, dt, gh, uzh, phn)
            dgd, duzd = drive_incr(gh, uzh, dt)
            gn = g+dgw+dgd; uzn = uz+duzw+duzd; Ew = Ew+dgw; Ed = Ed+dgd
            if nuf != 0.0:
                key, k1 = jax.random.split(key)
                pn = jnp.sqrt(gn*gn-1.0); xi = uzn/pn; s_ = jnp.sqrt(jnp.maximum(1-xi*xi, 0.0))
                nu = nu0*gn/pn**3
                e = jax.random.normal(k1, (g.shape[0], 2))*jnp.sqrt(nu*dt)[:, None]
                xin = (xi - e[:, 0]*s_)/jnp.sqrt(1+e[:, 0]**2+e[:, 1]**2)
                uzn = pn*jnp.clip(xin, -1, 1)
            dcm = (wn_ - kz[self.ic]*uzh/gh - 1.0/gh)*Om_
            psin = psi - dcm*dt
            zn = z + dt*C*(uzh/gh)
            phn_n = phn
            if gcom != 0.0 or gind != 0.0:
                key, k2 = jax.random.split(key)
                nz = jax.random.normal(k2, (g.shape[0], 1+L_))
                psin = psin + jnp.sqrt(2*gcom*dt)*nz[:, 0]
                if gind != 0.0: phn_n = phn + jnp.sqrt(2*gind*dt)*nz[:, 1:]
            return (gn, uzn, psin, zn, key, phn_n, Ew, Ed), None
        def block(carry, _):
            carry, _ = jax.lax.scan(step, carry, None, length=nrec)
            return carry, jnp.stack([carry[0], carry[1], carry[6], carry[7]], 0)
        carry0 = (jnp.asarray(g0), jnp.asarray(uz0), jnp.asarray(psi0), jnp.asarray(z0), (key if hasattr(key, 'shape') else jax.random.PRNGKey(key)), jnp.zeros((len(g0), self.L)), jnp.zeros(len(g0)), jnp.zeros(len(g0)))
        carry, recs = jax.lax.scan(block, carry0, None, length=nsteps//nrec)
        return carry, recs
