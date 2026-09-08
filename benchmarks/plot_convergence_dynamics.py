"""
plot_convergence_dynamics.py
============================

Result 1's OPENING figures: ``convergence_dynamics_<system>.png``, one per
system, showing <O(t)> against the exact reference as the bundle size M grows.

**Read the provenance before quoting anything from these figures.** They are
drawn from ``method_comparison_<system>_dim<D>.json`` -- RESULT 3's data, not
Result 1's -- because that file stores per-realization curves for every
observable at a single dimension, which is exactly what a time-trace panel
needs. The practical consequences:

  * the dimension is **64** by default, not the largest size Result 1 reaches;
  * there are **16 realizations** per M, not the 200 behind
    ``benchmark_accuracy_<system>.png`` and the error-decomposition figures;
  * the M values are Result 3's grid (1, 2, 4, 8, 16, 32, 64, 128, 256) capped
    at N_L, so they differ per system: A stops at 31, C at 32, and **B runs the
    full grid to 256**.

Result 1's other two figure groups (error decomposition, size invariance) come
from ``accuracy_vs_M_*`` with 200 realizations and are unaffected by any of
this. Do not describe all three as one dataset.

This script lived in ``scratch/`` until 2026-09-08 -- tracked, but filed away
from every other plotter and absent from the README, so the only way to find
the generator of a published figure was to grep for the filename. Moved here
unchanged apart from the paths, the CLI, and this note.

(The commit that moved it, 96a27c5, claims the file was untracked and that a
clean checkout could not regenerate the figure. Both are wrong: it was
committed in 65f22db. The move stands on filing, not on reachability.)

Run:  python plot_convergence_dynamics.py [--system ...] [--dim 64]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent

# Ultra-legibility, large-format plotting configuration for GitHub Markdown
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 2.4
plt.rcParams['xtick.major.width'] = 2.4
plt.rcParams['ytick.major.width'] = 2.4
plt.rcParams['xtick.major.size'] = 8.0
plt.rcParams['ytick.major.size'] = 8.0

LABELS = {
    # Chains (A, B)
    ("spin_chain", "energy"):       (r"$\langle H \rangle$  (total energy)",
                                     r"$\langle H \rangle$"),
    ("spin_chain", "sx"):           (r"$\sum_i\langle\sigma^x_i\rangle$  (bath coupling)",
                                     r"$\sum_i\langle\sigma^x_i\rangle$"),
    ("spin_chain", "coherence"):    ("dominant coherence",
                                     r"$\langle a|+\mathrm{h.c.}$"),

    ("mixed_chain", "energy"):      (r"$\langle H \rangle$  (total energy)",
                                     r"$\langle H \rangle$"),
    ("mixed_chain", "sx"):          (r"$\sum_i\langle\sigma^x_i\rangle$  (bath coupling)",
                                     r"$\sum_i\langle\sigma^x_i\rangle$"),
    ("mixed_chain", "sz"):          (r"$\sum_i\langle\sigma^z_i\rangle$  (magnetization)",
                                     r"$\sum_i\langle\sigma^z_i\rangle$"),
    ("mixed_chain", "coherence"):   ("dominant coherence",
                                     r"$\langle a|+\mathrm{h.c.}$"),

    # Oscillator (C)
    ("oscillator_bath", "energy"):  (r"$\langle H \rangle$  (total energy)",
                                     r"$\langle H \rangle$"),
    ("oscillator_bath", "x_sx"):    (r"$\langle x\otimes\sigma^x\rangle$  (osc-spin correlation)",
                                     r"$\langle x\otimes\sigma^x\rangle$"),
    ("oscillator_bath", "coherence"):("dominant coherence",
                                      r"$\langle a|+\mathrm{h.c.}$"),
}

CONFIGS = {
    "spin_chain":      ("System A (TFIM Chain)", ["energy", "sx", "coherence"]),
    "mixed_chain":     ("System B (Mixed Field Chain)",
                        ["energy", "sx", "sz", "coherence"]),
    "oscillator_bath": ("System C (Oscillator Bath)",
                        ["energy", "x_sx", "coherence"]),
}


def mean_curve(c):
    return np.mean(c, axis=0) if isinstance(c, (list, np.ndarray)) and np.ndim(c) == 2 else np.asarray(c)


def plot_convergence_system_huge(system_name, display_name, observables, dim):
    path = DATA_DIR / f"method_comparison_{system_name}_dim{dim}.json"
    if not path.exists():
        print(f"  SKIPPED {system_name}: no {path.name}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)

    point = d['point']
    tlist_meta = d.get('meta', {}).get('tlist', {})
    t0 = tlist_meta.get('t0', 0.0)
    t1 = tlist_meta.get('t1', 5.0)

    ref = point['reference']
    slb = point['methods'].get('slb', [])
    obs_names = point.get('observables', [])

    slb.sort(key=lambda x: x['M'])
    m_values = [x['M'] for x in slb]
    n_real = np.asarray(slb[0]["samples"]).shape[0] if slb else 0
    colors = plt.cm.Blues(np.linspace(0.42, 1.0, len(m_values)))

    n_obs = len(observables)
    if n_obs == 4:
        # 2x2 grid for System B (16x13 inches)
        fig, axes_grid = plt.subplots(2, 2, figsize=(16.0, 13.5), dpi=300)
        axes = axes_grid.flatten()
    elif n_obs == 3:
        # 1x3 horizontal layout with tall 8.5in height (18.0 x 8.5 inches)
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 8.5), dpi=300)
    else:
        fig, axes = plt.subplots(1, n_obs, figsize=(7.0 * n_obs, 7.5), dpi=300)
        if n_obs == 1:
            axes = [axes]

    for i, obs in enumerate(observables):
        ax = axes[i]
        title_label, y_label = LABELS.get((system_name, obs), (obs, rf"$\langle {obs} \rangle$"))

        ref_curve = mean_curve(ref['curves'][obs])
        times = np.linspace(t0, t1, len(ref_curve))

        # Plot exact reference with thick dashed line
        ax.plot(times, ref_curve, color='black', linewidth=4.0, linestyle='--', label='Exact Reference', zorder=10)

        if obs not in obs_names:
            print(f"Observable {obs} not in {system_name}")
            continue
        obs_idx = obs_names.index(obs)

        for j, result in enumerate(slb):
            M = result['M']
            samples = np.asarray(result["samples"], dtype=float)
            c = samples[:, obs_idx, :]
            curve = np.mean(c, axis=0)
            lw = 2.6 + (j / max(1, len(slb) - 1)) * 3.4
            ax.plot(times, curve, color=colors[j], linewidth=lw, label=f'SLB (M={M})', alpha=0.92)

        ax.set_xlabel("Time $t$", fontsize=20, fontweight='bold', labelpad=12)
        ax.set_ylabel(y_label, fontsize=20, fontweight='bold', labelpad=12)
        ax.set_title(title_label, fontsize=22, fontweight='bold', pad=18)
        ax.tick_params(axis='both', which='major', labelsize=17, pad=10)
        ax.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=1.5)

        # Legend placement
        if n_obs == 4:
            if i == 1: # top right
                ax.legend(loc='upper right', framealpha=0.95, fontsize=15.5, edgecolor='lightgray')
        else:
            if i == n_obs - 1:
                ax.legend(bbox_to_anchor=(1.03, 1.0), loc='upper left', framealpha=0.95, fontsize=16, edgecolor='lightgray')

    fig.suptitle(f"{display_name} (dim {dim}, $N_L={point.get('n_l', '?')}$): Convergence with Bundle Size $M$",
                 fontsize=25, fontweight='heavy', y=0.99 if n_obs == 4 else 1.05)

    plt.tight_layout()
    out_file = OUT_DIR / f"convergence_dynamics_{system_name}.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"  saved {out_file.name}  (dim {dim}, M={m_values}, "
          f"{n_real} realizations, reference {ref.get('method', 'unknown')})")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[4])
    ap.add_argument("--system", action="append", choices=sorted(CONFIGS),
                    help="repeatable; default is all three.")
    ap.add_argument("--dim", type=int, default=64,
                    help="Hilbert dimension to draw (default 64 -- the size "
                         "method_comparison covers on all three systems).")
    args = ap.parse_args()

    for name in (args.system or sorted(CONFIGS)):
        display_name, observables = CONFIGS[name]
        plot_convergence_system_huge(name, display_name, observables, args.dim)


if __name__ == "__main__":
    main()
