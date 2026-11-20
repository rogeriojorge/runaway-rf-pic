"""Homogeneous no-RF precursor: imposed total DC field, evolved bulk and seed.

This is a gyrotropic momentum-space model, not a PIC step with an unresolved gyro
orbit. Uniform E is parallel to B; the ideal external current -J_plasma holds the
TOTAL E fixed. For an initially gyrotropic homogeneous distribution, gyro rotation
does not change the modeled distribution. Collisions retain all three momentum
components. No coherent RF response may be added without a validated reduction.

The collision and magnetic LL kernels are the production JAX-in-Cell kernels.
Small |E|/(c|B|) controls the force approximation, not relative emission near zero
pitch: an absolute omitted-electric-emission diagnostic is also reported. Photons escape.
All species evolve; there is no thermostat. Energies/work are J/m^3. The reference
collision cell has length 1 m, converting density weights to the kernel's m^-2.
Unequal weights conserve collision moments in expectation: the RAW collision
defect is reported, never absorbed into an alleged conservation check. The tiny
seed needs particle/seed convergence before any physical precision claim: rare
accepted heavy-marker updates can make the defect exceed the seed-energy scale.

No avalanche, bremsstrahlung, screening, spatial transport or collective fields
are modeled here. This is not a completed runaway/suppression or RF-power model.
See Hirvijoki et al., JPP 81 (2015), equations 2.8--2.12, arXiv:1502.03333, for the
homogeneous electric/synchrotron drift. Numerical outputs are not written to disk.
"""
import os
from functools import partial

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import elementary_charge as e, epsilon_0, mass_electron as me, speed_of_light as c
from jaxincell._collisions import collide
from jaxincell._core import landau_lifshitz


def kinetic(u, mass):
    squared = jnp.sum(u * u, axis=-1)
    return mass * squared / (jnp.sqrt(1 + squared / c**2) + 1)


@partial(jax.jit, static_argnames=("blocks", "pairs", "steps"))
def evolve(u, density_weight, mass, charge, blocks, pairs, electric, magnetic,
           coulomb_log, dt, steps, seed=0):
    """Strang drift/collision/drift; the finite-step collision law still needs dt refinement.

    A midpoint RK drift uses the exact discrete kinetic-energy gradient for its
    DC/RR work, while independently quadraturing physical magnetic emission.
    AD is conditional on the random draws, pairing and collision chart decisions.
    """
    E, B = electric * jnp.array([1., 0., 0.]), magnetic * jnp.array([1., 0., 0.])
    acceleration = charge[:, None] / mass[:, None] * E
    tau = (charge / mass) * charge / (6 * jnp.pi * epsilon_0 * c**3)
    electric_power_scale = tau * (charge / mass)**2 * mass * electric**2
    zero_work = jnp.zeros((5, len(blocks)), u.dtype)  # DC, RR, magnetic emission, collision, omitted E emission

    def grouped(value):
        return jnp.stack([jnp.sum(value[start:start + count], axis=0) for start, count in blocks])

    def mean_velocity(before, after):
        def gamma(p):
            return jnp.sqrt(1 + jnp.sum((p / c)**2, axis=1, keepdims=True))
        return (before + after) / (gamma(before) + gamma(after))

    def drift(before, interval):
        first, _ = landau_lifshitz(before, jnp.zeros(3), B, charge, mass)
        middle = before + .5 * interval * (acceleration + first)
        reaction, power = landau_lifshitz(middle, jnp.zeros(3), B, charge, mass)
        after = before + interval * (acceleration + reaction)
        velocity = mean_velocity(before, after)
        dc = interval * charge * jnp.sum(E * velocity, axis=1)
        rr = interval * mass * jnp.sum(reaction * velocity, axis=1)
        electric_power = electric_power_scale * (1 + jnp.sum((middle[:, 1:] / c)**2, axis=1))
        return after, jnp.stack([grouped(density_weight * p)
                                for p in (dc, rr, interval * power, interval * electric_power)])

    def diagnostics(momentum, work, collision_momentum):
        gamma = jnp.sqrt(1 + jnp.sum((momentum / c)**2, axis=1))
        energy = kinetic(momentum, mass)
        return {"kinetic": grouped(density_weight * energy), "work": work,
                "current": grouped(density_weight * charge * momentum[:, 0] / gamma),
                "above_10MeV": grouped(density_weight * (energy > 1e7 * e)),
                "collision_momentum": collision_momentum}

    def step(carry, _):
        momentum, key, work, defect = carry
        before, first = drift(momentum, .5 * dt)
        key, draw = jax.random.split(key)
        after = collide(draw, jnp.zeros_like(before), before, density_weight * 1., mass, charge,
                        blocks, pairs, coulomb_log, dt, 1., 1., 1, relativistic=True)
        delta = mass * jnp.sum(mean_velocity(before, after) * (after - before), axis=1)
        defect += jnp.sum((density_weight * mass)[:, None] * (after - before), axis=0)
        momentum, second = drift(after, .5 * dt)
        drift_work = first + second
        work += jnp.concatenate((drift_work[:3], grouped(density_weight * delta)[None], drift_work[3:]), axis=0)
        return (momentum, key, work, defect), diagnostics(momentum, work, defect)

    start = diagnostics(u, zero_work, jnp.zeros(3))
    final, history = jax.lax.scan(jax.checkpoint(step),
                                  (u, jax.random.PRNGKey(seed), zero_work, jnp.zeros(3)), None, length=steps)
    history = jax.tree.map(lambda a, b: jnp.concatenate((a[None], b), axis=0), start, history)
    return final[0], history


def thermal_momentum(count, temperature_ev, mass, seed):
    """Maxwell--Juttner d^3p quadrature; antithetic pairs remove initial bulk current."""
    if count % 2:
        raise ValueError("the quiet thermal population requires an even marker count")
    theta = temperature_ev * e / (mass * c**2)
    radius = np.linspace(0., max(20 * np.sqrt(theta), 40 * theta), 20001)
    pdf = radius**2 * np.exp(-radius**2 / ((np.sqrt(1 + radius**2) + 1) * theta))
    cdf = np.r_[0., np.cumsum(pdf[1:] + pdf[:-1])]
    cdf /= cdf[-1]
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(count // 2, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    magnitude = np.interp(rng.permutation((np.arange(count // 2) + .5) / (count // 2)), cdf, radius)
    half = c * magnitude[:, None] * direction
    return np.concatenate((half, -half))


# Physical local inputs; overrides only shorten local validation, without rescaling physics.
ne, fraction, Te, Ti, gamma_seed, pitch = 1e18, 1e-5, 100., 100., 40., .1
B0, E_acc, logarithm = 2., .005, 15.                 # T, V/m, supplied leading-log value
count = int(os.environ.get("BASELINE_N", "512"))
seed_count = max(8, count // 8)
steps = int(os.environ.get("BASELINE_STEPS", "256"))
duration = float(os.environ.get("BASELINE_TIME", "1e-4"))
mi = 3.3435837768e-27                               # deuteron mass, kg
blocks = ((0, count), (count, seed_count), (count + seed_count, count))
pairs = tuple((a, b) for a in range(3) for b in range(a, 3))
phase = 2 * np.pi * np.arange(seed_count) / seed_count
beam = c * np.sqrt(gamma_seed**2 - 1) * np.stack(
    (np.full(seed_count, np.cos(pitch)), np.sin(pitch) * np.cos(phase), np.sin(pitch) * np.sin(phase)), axis=1)
u0 = jnp.asarray(np.concatenate((thermal_momentum(count, Te, me, 21), beam,
                                thermal_momentum(count, Ti, mi, 22))))
density = np.array([ne * (1 - fraction), ne * fraction, ne])
weight = jnp.repeat(jnp.asarray(density / np.array([count, seed_count, count])),
                    jnp.array([count, seed_count, count]))
mass = jnp.repeat(jnp.array([me, me, mi]), jnp.array([count, seed_count, count]))
charge = jnp.repeat(jnp.array([-e, -e, e]), jnp.array([count, seed_count, count]))
final, history = evolve(u0, weight, mass, charge, blocks, pairs, -E_acc, B0, logarithm,
                        duration / steps, steps)
history = jax.tree.map(np.asarray, history)
K, work = history["kinetic"], history["work"]
residual = (K - K[0] - work[:, 0] - work[:, 1]).sum(axis=1)
collision_defect = work[:, 3].sum(axis=1)
assert np.isfinite(K).all() and np.isfinite(work).all()
assert np.max(np.abs(residual - collision_defect)) < 1e-11 * K[0].sum()
print(f"Homogeneous gyrotropic no-RF model; total E_parallel={-E_acc:g} V/m, B={B0:g} T")
print(f"E/(cB)={E_acc/(c*B0):.3e}; duration={duration:g} s; dt={duration/steps:g} s")
print(f"Density weights represent thermal/seed/ion populations {density}; all evolve.")
print(f"K_initial={K[0].sum():.8g}; DC={work[-1, 0].sum():.8g}; RR={work[-1, 1].sum():.8g}; "
      f"emitted={work[-1, 2].sum():.8g} J/m^3")
print(f"Omitted electric emission along this trajectory: {work[-1, 4].sum():.8g} J/m^3; "
      "compare its absolute budget with the target, not a ratio to vanishing magnetic emission.")
print(f"RAW collision energy defect={collision_defect[-1]:.8g} J/m^3; "
      f"momentum defect={history['collision_momentum'][-1]} kg/(m^2 s)")
print(f"Expected accepted heavy thermal-electron updates from seed pairings: "
      f"{steps * count * fraction / (1 - fraction):.3g}; this default does not resolve their statistics.")
print("The closed stage ledger is bookkeeping; unequal-weight conservation needs an ensemble convergence gate.")
print("This short precursor establishes neither a sustained runaway prediction nor RF source-power control.")

fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
t = np.arange(steps + 1) * duration / steps
axes[0].plot(t, K[:, 1] / (density[1] * e * 1e6))
axes[0].set(xlabel="time (s)", ylabel="initial-seed mean energy (MeV)")
for i, name in ((0, "thermal electrons"), (2, "ions")):
    axes[1].plot(t, K[:, i] / (density[i] * e), label=name)
axes[1].set(xlabel="time (s)", ylabel="mean kinetic energy (eV)")
axes[1].legend()
axes[2].plot(t, residual, label="K-K0-W_DC-W_RR")
axes[2].plot(t, collision_defect, "--", label="raw collision defect")
axes[2].set(xlabel="time (s)", ylabel="energy density (J/m³)")
axes[2].legend()
plt.show()
