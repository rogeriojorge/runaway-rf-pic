"""Local one-CPU checks of runaway_baseline.py; writes only an external JSON report."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
os.environ.update(JAX_ENABLE_X64="1", MPLBACKEND="Agg", BASELINE_N="32",
                  BASELINE_STEPS="8", BASELINE_TIME="1e-6", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
started = time.perf_counter()
path = Path(__file__).with_name("runaway_baseline.py")
namespace = runpy.run_path(str(path))
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from jaxincell import epsilon_0

plt.close("all")
evolve, kinetic = namespace["evolve"], namespace["kinetic"]
c, e, me = namespace["c"], namespace["e"], namespace["me"]
tau = e**2 / (6 * np.pi * epsilon_0 * me * c**3)
report = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          "jax": jax.__version__, "devices": str(jax.devices()), "affinity": sorted(os.sched_getaffinity(0))}


def single(momentum, electric, magnetic, duration, steps):
    return evolve(jnp.asarray(momentum)[None], jnp.ones(1), jnp.full(1, me), jnp.full(1, -e),
                  ((0, 1),), (), electric, magnetic, 15., duration / steps, steps)


# Exact constant-B solution, including a derivative distinct from differentiating the integrator.
g0, pitch, B, duration = 40., .1, 2., .1
initial = c * np.sqrt(g0**2 - 1) * np.array([np.cos(pitch), np.sin(pitch), 0.])
beta_parallel = initial[0] / (c * g0)
gp = 1 / np.sqrt(1 - beta_parallel**2)
ab = tau * (e * B / me)**2
r = (g0 - gp) / (g0 + gp) * np.exp(-2 * ab * duration / gp)
exact_gamma = gp * (1 + r) / (1 - r)
exact_derivative = -8 * ab * duration * r / (B * (1 - r)**2)
errors = []
for steps in (32, 64, 128):
    final, history = single(initial, 0., B, duration, steps)
    gamma = float(jnp.sqrt(1 + jnp.sum((final / c)**2)))
    errors.append(abs(gamma - exact_gamma))
    emission = float(history["work"][-1, 2, 0] / (me * c**2))
assert errors[-1] / exact_gamma < 3e-6
assert 3.5 < errors[-2] / errors[-1] < 4.5
assert abs(emission / (g0 - exact_gamma) - 1) < 3e-5
gradient = float(jax.grad(lambda b: kinetic(single(initial, 0., b, duration, 128)[0], me).sum()
                          / (me * c**2))(B))
assert abs(gradient / exact_derivative - 1) < 2e-5
report["magnetic_exact"] = {"gamma_errors_32_64_128": errors,
                            "last_error_ratio": errors[-2] / errors[-1],
                            "emission_relative_error": emission / (g0 - exact_gamma) - 1,
                            "d_gamma_dB": gradient, "analytic_derivative": exact_derivative}
print("Exact magnetic cooling and analytic derivative: passed", flush=True)

# DC acceleration is an exact momentum translation, independent of the RR implementation.
electric, duration = -.005, .1
final, history = single(initial, electric, 0., duration, 32)
expected = initial + np.array([-e * electric * duration / me, 0., 0.])
np.testing.assert_allclose(np.asarray(final)[0], expected, rtol=2e-14, atol=1e-7)
expected_work = float(kinetic(expected, me) - kinetic(initial, me))
dc_work = float(history["work"][-1, 0, 0])
assert abs(dc_work / expected_work - 1) < 2e-12
electric_emission = float(history["work"][-1, 4, 0])
reference_electric_emission = tau * (e/me)**2 * me * electric**2 * (1 + (initial[1]/c)**2) * duration
assert abs(electric_emission / reference_electric_emission - 1) < 2e-14
report["electric_exact"] = {
    "momentum_relative_error": float(np.linalg.norm(np.asarray(final)[0] - expected) / np.linalg.norm(expected)),
    "work_relative_error": dc_work / expected_work - 1,
    "omitted_electric_emission_J": electric_emission,
    "emission_relative_error": electric_emission / reference_electric_emission - 1}
print("Exact DC translation and source work: passed", flush=True)

# Independent scalar p,xi equations; no calls to the production LL kernel in the reference.
electric, B, duration, g0, pitch = -.01, 1.5, .5, 15., .4
eta0, xi0 = np.sqrt(g0**2 - 1), np.cos(pitch)
initial = c * eta0 * np.array([xi0, np.sin(pitch), 0.])
ae, ab = -e * electric / (me * c), tau * (e * B / me)**2


def reference_rhs(_, state):
    p, xi = state[:2]
    gamma = np.sqrt(1 + p*p)
    return [ae * xi - ab * gamma * p * (1 - xi*xi),
            ae * (1 - xi*xi) / p + ab * xi * (1 - xi*xi) / gamma,
            ae * p * xi / gamma, ab * p*p * (1 - xi*xi)]


reference = solve_ivp(reference_rhs, (0., duration), [eta0, xi0, 0., 0.],
                      method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
errors = []
for steps in (32, 64, 128):
    final, history = single(initial, electric, B, duration, steps)
    eta = np.asarray(final)[0] / c
    measured = np.array([np.linalg.norm(eta), eta[0] / np.linalg.norm(eta),
                         history["work"][-1, 0, 0] / (me*c*c), history["work"][-1, 2, 0] / (me*c*c)])
    errors.append(float(np.linalg.norm((measured - reference) / reference)))
assert errors[-1] < 1e-4
assert 3.5 < errors[-2] / errors[-1] < 4.5
objective = jax.jit(lambda field: kinetic(single(initial, field, B, duration, 128)[0], me).sum() / (me*c*c))
derivative = float(jax.grad(objective)(electric))
h = 1e-6
finite = float((objective(electric + h) - objective(electric - h)) / (2*h))
assert abs(derivative / finite - 1) < 2e-5
report["independent_p_pitch_ode"] = {
    "relative_errors_32_64_128": errors, "last_error_ratio": errors[-2] / errors[-1],
    "AD": derivative, "finite_difference": finite, "AD_FD_relative": derivative / finite - 1}
print("Independent p-pitch ODE and deterministic AD: passed", flush=True)

# Closed equal-weight system: assess RAW conservation, without subtracting collision_work.
n = 32
u = jnp.asarray(namespace["thermal_momentum"](n, 1e4, me, 301))
w, mass, charge = jnp.full(n, 1e20/n), jnp.full(n, me), jnp.full(n, -e)
final, history = evolve(u, w, mass, charge, ((0, n),), ((0, 0),), 0., 0., 15., 1e-6, 32)
energy_error = float(history["kinetic"][-1, 0] / history["kinetic"][0, 0] - 1)
momentum_scale = float(jnp.sum(w * mass * jnp.linalg.norm(u, axis=1)))
momentum_error = float(jnp.linalg.norm(history["collision_momentum"][-1]) / momentum_scale)
assert abs(energy_error) < 2e-12 and momentum_error < 2e-12
assert float(jnp.linalg.norm(final - u) / jnp.linalg.norm(u)) > 1e-3
report["equal_weight_collision_conservation"] = {"energy_relative": energy_error,
                                                 "momentum_relative": momentum_error}
print("Raw equal-weight collision energy/momentum conservation: passed", flush=True)

# Cold-target rate and density derivative: a single collision has an analytic angular mean.
n, gamma, density, s = 128, 5., 1e20, .002
eta = np.sqrt(gamma*gamma - 1)
rate = (e**4 * density * 15. / (4*np.pi*epsilon_0**2*c**3)
        * (2*me*me*(1 + gamma)) * gamma / (me**4 * (gamma*gamma - 1)**1.5))
u = jnp.zeros((2*n, 3)).at[:n, 0].set(c*eta)
mass, charge = jnp.full(2*n, me), jnp.full(2*n, -e)


def density_response(scale, seed):
    final, _ = evolve(u, jnp.full(2*n, scale*density/n), mass, charge,
                      ((0, n), (n, n)), ((0, 1),), 0., 0., 15., s/rate, 1, seed)
    return jnp.mean(final[:n, 0] / c)


seeds = jnp.arange(16)
values = np.asarray(jax.jit(jax.vmap(density_response, in_axes=(None, 0)))(1., seeds))
gradients = np.asarray(jax.jit(jax.vmap(jax.grad(density_response), in_axes=(None, 0)))(1., seeds))
exact_value = eta * (1 - .5 * -np.expm1(-s))
exact_gradient = -.5 * eta * s * np.exp(-s)


def stderr(a):
    return a.std(ddof=1) / np.sqrt(a.size)


assert abs(values.mean() - exact_value) < 5 * stderr(values)
assert abs(gradients.mean() - exact_gradient) < 5 * stderr(gradients)
mean = jax.jit(lambda scale: jax.vmap(density_response, in_axes=(None, 0))(scale, seeds).mean())
finite = float((mean(1.0001) - mean(.9999)) / .0002)
assert abs(gradients.mean() / finite - 1) < 2e-6
report["cold_target_rate_and_gradient"] = {
    "mean": float(values.mean()), "analytic_mean": exact_value,
    "mean_standard_error": float(stderr(values)),
    "AD_mean": float(gradients.mean()), "analytic_derivative": exact_gradient,
    "AD_standard_error": float(stderr(gradients)),
    "finite_difference": finite, "AD_FD_relative": float(gradients.mean()/finite - 1)}
print("Independent collision rate, ensemble derivative and common-seed FD: passed", flush=True)

report["scope"] = "Component/integration microbenchmarks passed; no converged long-time dilute-tail or RF model."
report["elapsed_seconds"] = time.perf_counter() - started
Path(__file__).with_name("validation_results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
