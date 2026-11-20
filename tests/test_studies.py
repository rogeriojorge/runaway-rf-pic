"""Fast checks: each study runs in its default (quick) setting, and the collision kernel's
ultrarelativistic drag equals the Connor and Hastie (1975) critical force."""
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxincell import elementary_charge as e_charge, epsilon_0, mass_electron, speed_of_light as c_light
from jaxincell._collisions import collide

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ["runaway_baseline", "runaway_rf", "runaway_rf_minimum", "rf_source_control"]


@pytest.mark.parametrize("name", STUDIES)
def test_study_runs_in_quick_preset(name, tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg", JAX_ENABLE_X64="1")
    done = subprocess.run([sys.executable, str(ROOT / "studies" / name / f"{name}.py")], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=1500)
    assert done.returncode == 0, done.stderr[-2000:]


def _cold_relativistic_rate(gamma, mass_ratio, density, logarithm=15.):
    """Invariant form of Perez et al. (2012) Eq. 9 for a beam on a stationary target."""
    invariant_mass2 = 1 + mass_ratio ** 2 + 2 * mass_ratio * gamma
    return ((e_charge ** 2 / epsilon_0 / mass_electron) ** 2 * density * logarithm
            / (4 * np.pi * c_light ** 3) * invariant_mass2 * gamma
            / (mass_ratio ** 2 * (gamma ** 2 - 1) ** 1.5))


def test_ultrarelativistic_drag_is_the_connor_hastie_critical_force():
    """An electron with v -> c loses energy to cold electrons at e E_c c,
    E_c = n e^3 lnL/(4 pi eps0^2 m c^2); 7.649e-4 V/m at n = 1e18, lnL = 15, which makes the 5 mV/m
    baseline field 6.5 E_c."""
    pairs, density, logarithm, gamma = 8192, 1e18, 15., 100.
    critical = density * e_charge ** 3 * logarithm / (4 * np.pi * epsilon_0 ** 2 * mass_electron * c_light ** 2)
    assert abs(critical / 7.6486e-4 - 1) < 1e-4
    u = jnp.zeros((2 * pairs, 3)).at[:pairs, 0].set(c_light * np.sqrt(gamma ** 2 - 1))
    mass, charge = jnp.full(2 * pairs, mass_electron), jnp.full(2 * pairs, -e_charge)
    dt = 1e-3 / _cold_relativistic_rate(gamma, 1., density, logarithm)

    def lost(key):
        new = collide(key, jnp.zeros_like(u), u, jnp.full(2 * pairs, density / pairs), mass, charge,
                      ((0, pairs), (pairs, pairs)), ((0, 1),), logarithm, dt, 1., 1., 1, relativistic=True)
        gamma_new = jnp.sqrt(1 + jnp.sum(new[:pairs] ** 2, axis=1) / c_light ** 2)
        return (gamma - gamma_new) * mass_electron * c_light ** 2 / dt
    import jax
    power = np.asarray(jax.vmap(lost)(random.split(random.PRNGKey(8), 16))).mean(axis=0)
    speed = c_light * np.sqrt(1 - 1 / gamma ** 2)
    expected = e_charge * critical * speed
    assert abs(power.mean() / expected - 1) < 5 * power.std(ddof=1) / np.sqrt(power.size) / expected + 2e-3
