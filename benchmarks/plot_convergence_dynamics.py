"""
plot_convergence_dynamics.py
============================

Result 1's OPENING figures: ``convergence_dynamics_<system>.png``, one per
system, showing <O(t)> against the exact reference as the bundle size M grows.

Drawn from ``accuracy_vs_M_<system>_dim<D>.json`` -- Result 1's own files,
the same ones behind the error-decomposition and size-invariance figures --
since 2026-09-23. They store 200 realizations per M of every observable at
every time point, which is exactly what a time-trace panel needs.

Until then these figures read Result 3's ``method_comparison`` files, with 16
realizations per M, and that capped them one size lower on both chains:
Result 3's larger points (System A at 1024 and 2048, System B at 256) ran only
the exact solver and mcsolve, so the largest Result 3 size with SLB curves was
512 on A and 128 on B. Result 1 reaches 1024 on A and 256 on B. The switch
costs System B its M = 128 and 256 curves -- Result 1's ladder stops at 64 --
and gains every panel 12.5x the realizations.

  * the dimension defaults to the largest CANONICAL Result 1 file per system
    (``_dim<D>.json`` and nothing after the number -- a ``_r16`` side-run at a
    non-default realization count is never picked), so all three of Result 1's
    figure groups show the same sizes;
  * the M ladder is Result 1's: 2 to 64 on all three systems at the default
    sizes, capped at N_L below that (System A stops at 13, 21, 31, 43 and 57
    at dims 16 to 256);
  * ``M=1`` is not in Result 1's ladder; ``--min-m`` can still drop rungs.

``worst_panel_rows`` computes the table BENCHMARKS.md prints under these
figures -- each M's worst plotted panel as a percentage of that panel's
reference span, and whether that deviation is resolved above three standard
errors of its own scatter -- so the table and the figure come from one place.

History: this script lived in ``scratch/`` until 2026-09-08 (committed in
65f22db, moved in 96a27c5).

Run:  python plot_convergence_dynamics.py [--system ...] [--dim auto|N] [--min-m 2]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (add_settings_footer, format_slb_settings, result1_reference,
                    result1_samples, size_label)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent

# A deviation counts as resolved when it exceeds this many standard errors of
# the mean at the instant it peaks. The same rule the table under these
# figures has always used; below it the "deviation" is sampling scatter.
RESOLVED_SEM = 3.0

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


def largest_dim(system_name):
    """Largest CANONICAL Result 1 dimension for this system.

    Per system, not global: the systems do not reach the same sizes. Matches
    ``_dim<D>.json`` exactly, as plot_accuracy_vs_M's auto-pick and the tests'
    committed_dims do, so a suffixed side-run such as ``_dim2048_r16.json`` is
    never chosen -- its realization count differs from every other curve's.
    """
    canonical = re.compile(rf"^accuracy_vs_M_{re.escape(system_name)}_dim(\d+)\.json$")
    dims = [int(m.group(1)) for p in DATA_DIR.glob(f"accuracy_vs_M_{system_name}_dim*.json")
            if (m := canonical.match(p.name))]
    return max(dims) if dims else None


def load(system_name, dim):
    """The Result 1 file for this system and size, or None if it is absent."""
    path = DATA_DIR / f"accuracy_vs_M_{system_name}_dim{dim}.json"
    if not path.exists():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def _curves(d, item, obs):
    """(reference, samples) for one observable at one M, or None when this
    file does not store it (the oldest Result 1 files kept only the energy and
    the coherence)."""
    try:
        ref = np.asarray(result1_reference(d, obs), dtype=float)
        samples = np.asarray(result1_samples(item, obs), dtype=float)
    except (KeyError, TypeError):
        return None
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    return ref, samples


def worst_panel_rows(system_name, dim=None, min_m=2):
    """The table printed under these figures, one row per M.

    For every plotted panel: the largest deviation of the SLB mean from the
    reference over time, as a percentage of that panel's reference span, and
    that deviation in standard errors of the mean at the instant it peaks.
    The row reports the worst panel. ``resolved`` is that z above
    RESOLVED_SEM; below it the percentage is an upper bound, not a
    measurement.
    """
    dim = largest_dim(system_name) if dim is None else dim
    d, _ = load(system_name, dim)
    if d is None:
        return []
    rows = []
    for item in sorted(d["slb_sweep"], key=lambda x: x["M"]):
        if item["M"] < min_m:
            continue
        panels = []
        for obs in CONFIGS[system_name][1]:
            got = _curves(d, item, obs)
            if got is None:
                continue
            ref, s = got
            mean = s.mean(axis=0)
            sem = s.std(axis=0, ddof=1) / np.sqrt(s.shape[0])
            dev = np.abs(mean - ref)
            i = int(np.argmax(dev))
            span = float(ref.max() - ref.min())
            panels.append((100.0 * dev[i] / span, dev[i] / sem[i], obs))
        pct, z, obs = max(panels)
        rows.append({"M": int(item["M"]), "percent": float(pct), "z": float(z),
                     "panel": obs, "resolved": bool(z > RESOLVED_SEM),
                     "n_realizations": int(s.shape[0])})
    return rows


def plot_convergence_system_huge(system_name, display_name, observables, dim,
                                 min_m=2):
    if dim is None:
        print(f"  SKIPPED {system_name}: no Result 1 files")
        return
    d, path = load(system_name, dim)
    if d is None:
        print(f"  SKIPPED {system_name}: no {path.name}")
        return

    tl = d.get("meta", {}).get("tlist", {})
    sweep = sorted((x for x in d["slb_sweep"] if x["M"] >= min_m), key=lambda x: x["M"])
    m_values = [x["M"] for x in sweep]
    colors = plt.cm.Blues(np.linspace(0.42, 1.0, len(m_values)))
    n_real = 0

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
        first = _curves(d, sweep[0], obs) if sweep else None
        if first is None:
            # The oldest Result 1 files (dims 16-64) stored only the energy and
            # the coherence. Hide the panel rather than save a blank one.
            print(f"  {system_name} dim {dim}: no '{obs}' curves in {path.name} "
                  f"-- panel hidden")
            ax.set_visible(False)
            continue
        ref_curve = first[0]
        times = np.linspace(tl.get("t0", 0.0), tl.get("t1", 5.0), len(ref_curve))

        # Plot exact reference with thick dashed line
        ax.plot(times, ref_curve, color='black', linewidth=4.0, linestyle='--',
                label='Exact Reference', zorder=10)

        for j, item in enumerate(sweep):
            _, samples = _curves(d, item, obs)
            n_real = samples.shape[0]
            lw = 2.6 + (j / max(1, len(sweep) - 1)) * 3.4
            ax.plot(times, samples.mean(axis=0), color=colors[j], linewidth=lw,
                    label=f'SLB (M={item["M"]})', alpha=0.92)

        ax.set_xlabel("Time $t$", fontsize=20, fontweight='bold', labelpad=12)
        ax.set_ylabel(y_label, fontsize=20, fontweight='bold', labelpad=12)
        ax.set_title(title_label, fontsize=22, fontweight='bold', pad=18)
        ax.tick_params(axis='both', which='major', labelsize=17, pad=10)
        ax.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=1.5)

        # Legend placement
        if n_obs == 4:
            if i == 1:  # top right panel, lower right corner: the bath-coupling
                # curve runs along the top of this panel at late times, and an
                # upper-right legend sat on top of the exact reference there.
                ax.legend(loc='lower right', framealpha=0.95, fontsize=15.5, edgecolor='lightgray')
        else:
            if i == n_obs - 1:
                ax.legend(bbox_to_anchor=(1.03, 1.0), loc='upper left', framealpha=0.95,
                          fontsize=16, edgecolor='lightgray')

    fig.suptitle(f"{display_name} ({size_label(system_name, dim)}, "
                 f"$N_L={d.get('n_l', '?')}$): "
                 f"Convergence with Bundle Size $M$",
                 fontsize=25, fontweight='heavy', y=0.99 if n_obs == 4 else 1.05)

    plt.tight_layout()

    # Settings on the figure, not only in the prose. These figures are read
    # away from the document -- in talks, in issues -- and the two questions
    # they kept raising were how many runs each curve averages and whether the
    # spread is drawn. Both are answered here, from the file's own metadata.
    # fontsize 15, not the helper's default 9: these panels carry 22pt titles
    # and 20pt axis labels, so the default caption is a fifth the size of
    # everything around it and unreadable at normal viewing scale.
    add_settings_footer(
        fig,
        format_slb_settings(M=m_values, substeps=d.get("substeps"),
                            n_realizations=n_real, swept=True),
        "mean curves only, no band drawn",
        f"reference: {d.get('reference_method', 'unknown')}",
        f"source: {path.name}",
        fontsize=15, wrap_chars=105, y=-0.04,
    )

    out_file = OUT_DIR / f"convergence_dynamics_{system_name}.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f"  saved {out_file.name}  (dim {dim}, M={m_values}, "
          f"{n_real} realizations, reference {d.get('reference_method', 'unknown')})")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[4])
    ap.add_argument("--system", action="append", choices=sorted(CONFIGS),
                    help="repeatable; default is all three.")
    ap.add_argument("--dim", default="auto",
                    help="Hilbert dimension to draw, or 'auto' (the default) "
                         "for the largest canonical Result 1 file each system "
                         "has.")
    ap.add_argument("--min-m", type=int, default=2,
                    help="lowest bundle size to draw (default 2, the bottom "
                         "of Result 1's ladder).")
    args = ap.parse_args()

    for name in (args.system or sorted(CONFIGS)):
        display_name, observables = CONFIGS[name]
        dim = largest_dim(name) if args.dim == "auto" else int(args.dim)
        plot_convergence_system_huge(name, display_name, observables, dim,
                                     min_m=args.min_m)
        for r in worst_panel_rows(name, dim, min_m=args.min_m):
            print(f"    M={r['M']:>3}  worst {r['panel']:10s} {r['percent']:7.2f}% of span  "
                  f"{r['z']:5.1f} s.e.m.  {'resolved' if r['resolved'] else 'NOT resolved'}")


if __name__ == "__main__":
    main()
