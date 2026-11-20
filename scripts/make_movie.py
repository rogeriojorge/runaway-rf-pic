"""Movie of the comb scattering a runaway band: coherent comb against a comb with phase noise.
Reruns the reduced N-line model of results/step3 (about 1 minute on a laptop CPU).
Usage: python scripts/make_movie.py   ->  docs/media/comb_scattering.{mp4,gif}"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, "results/step3/scripts")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from pn import setup, make_rhs, red_env, REG, TH0

SIGMA, NP, NBEATS, NFRAME = 3.0, 200, 120.0, 60
CASES = [(0.0, "coherent comb", "#D55E00"), (0.1, r"phase noise  $\Gamma_\phi=0.1\,\Delta\omega$", "#0072B2")]


def run(gphi, seed=1, dtfac=0.1):
    s = setup("eqN10", SIGMA)
    L, ic, Dd, wb = s["L"], s["ic"], s["Dd"], s["wb"]
    Tb = 2 * np.pi / Dd
    dt = dtfac / ((L + s["win_lines"]) / 2 * Dd + 4 * wb)
    r = np.random.default_rng(seed)
    gam = r.uniform(*s["gwin"], NP)
    p = np.sqrt(gam ** 2 - 1)
    th = r.uniform(TH0 - 0.01, TH0 + 0.01, NP)
    phi = r.uniform(0, 2 * np.pi, NP)
    z0 = r.uniform(0, 2 * np.pi / REG["eqN10"][1], NP)
    ph0 = r.uniform(0, 2 * np.pi, L)
    ph0 = ph0 - ph0[ic]
    red = red_env.Red(s["ks"], s["b"] * s["a"])
    rhs = make_rhs(red, 1.0)
    sq = np.sqrt(2 * gphi * Dd * dt)

    def step(c, _):
        y, W, key = c
        key, k2 = jax.random.split(key)
        W = W + sq * jax.random.normal(k2, W.shape)
        ph = W + jnp.asarray(ph0)[None, :]
        k1 = rhs(y, ph); k2_ = rhs(y + 0.5 * dt * k1, ph); k3 = rhs(y + 0.5 * dt * k2_, ph); k4 = rhs(y + dt * k3, ph)
        return (y + dt / 6 * (k1 + 2 * k2_ + 2 * k3 + k4), W, key), None

    y = jnp.stack([jnp.asarray(gam), jnp.asarray(-p * np.cos(th)), jnp.asarray(phi), jnp.asarray(z0)], 1)
    c = (y, jnp.zeros((NP, L)), jax.random.PRNGKey(7))
    beats = np.concatenate([[0], np.geomspace(1.0, NBEATS, NFRAME - 1)])
    K = 200
    chunk = jax.jit(lambda cc: jax.lax.scan(step, cc, None, length=K)[0])
    done, G = 0, [np.asarray(c[0])]
    for bt in beats[1:]:
        target = int(round(bt * Tb / dt))
        while done + K <= target:
            c = chunk(c)
            done += K
        G.append(np.asarray(c[0]))
    return beats, np.array(G), s


def main():
    data = [run(g) for g, _, _ in CASES]
    beats, _, s = data[0]
    g0 = data[0][1][0, :, 0]
    lo, hi = s["gwin"]
    fig, axs = plt.subplots(2, 2, figsize=(8, 5.2), gridspec_kw={"height_ratios": [2.2, 1]})
    plt.rcParams.update({"font.size": 11})
    sc, fl = [], []
    for j, (g, lab, col) in enumerate(CASES):
        a = axs[0, j]
        a.axhspan(lo, hi, color="0.9", lw=0)
        pts = a.scatter(np.zeros(NP), g0, s=5, color=col, alpha=0.7)
        a.set_xlim(0, 2 * np.pi); a.set_ylim(lo - 1.2 * (hi - lo), hi + 1.2 * (hi - lo))
        a.set_title(lab, fontsize=11, color=col)
        a.set_xlabel("wave phase (rad)")
        if j == 0:
            a.set_ylabel(r"electron energy $\gamma$")
        sc.append(pts)
        fill = np.mean((data[j][1][:, :, 0] - g0) ** 2, 1) / (s["Lw"] ** 2 / 6)
        b = axs[1, j]
        b.loglog(beats[1:], np.maximum(fill[1:], 1e-3), color="0.8")
        b.axhline(1, color="0.5", ls=":")
        (ln,) = b.loglog([], [], color=col)
        b.set_xlim(1, NBEATS); b.set_ylim(1e-3, 10)
        b.set_xlabel("time (beat periods)")
        if j == 0:
            b.set_ylabel("band fill")
        fl.append((ln, fill))
    fig.tight_layout()
    out = ROOT / "docs" / "media"; out.mkdir(parents=True, exist_ok=True)
    frames = out / "_frames"; frames.mkdir(exist_ok=True)
    for k in range(NFRAME):
        for j in range(2):
            G = data[j][1][k]
            sc[j].set_offsets(np.c_[np.mod(G[:, 2], 2 * np.pi), G[:, 0]])
            fl[j][0].set_data(beats[1:k + 1] if k > 0 else [], np.maximum(fl[j][1][1:k + 1], 1e-3) if k > 0 else [])
        fig.savefig(frames / f"f{k:03d}.png", dpi=110)
    for k in range(12):  # hold the last frame
        fig.savefig(frames / f"f{NFRAME + k:03d}.png", dpi=110)
    sh = lambda cmd: subprocess.run(cmd, shell=True, check=True, capture_output=True)
    sh(f"ffmpeg -y -framerate 15 -i {frames}/f%03d.png -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p -crf 26 {out}/comb_scattering.mp4")
    sh(f"ffmpeg -y -framerate 15 -i {frames}/f%03d.png -vf 'scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4' {out}/comb_scattering.gif")
    for f in frames.glob("*.png"):
        f.unlink()
    frames.rmdir()


if __name__ == "__main__":
    main()
