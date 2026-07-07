"""
plot_frontier.py
================

ANALYSIS/FIGURE HALF of the accuracy-vs-cost frontier benchmark (Result 3).
Reads the data written by run_frontier.py and draws the frontier figure; runs
in seconds.

The SLB averaging levels (N_RUNS_SWEEP) are applied HERE by subsampling the
saved raw runs: for each level N, the estimate at each M is the mean of the
first N of the saved runs, its error bar the time-averaged SEM of those runs,
and its cost N x the saved per-run wall-clock. Levels exceeding the number of
saved runs are dropped with a notice. The saved substeps-guard verdict is
re-printed (and warned about) so the integrator-convergence evidence travels
with the figure.

The figure (unchanged filename):  benchmark_frontier_<system>.png
Run:  python plot_frontier.py [--system spin_chain|oscillator_bath|<big names>|all]
"""

from __future__ import annotations

import argparse

import numpy as np

from common import (
    add_settings_footer, as_array, load_data, tavg_bias_sem_rmse,
    MC_ATOL, MC_RTOL, SUBSTEPS,
)

N_RUNS_SWEEP = [8, 16, 32]      # SLB averaging levels; analysis-time choice

DEFAULT_SYSTEMS = ["spin_chain", "oscillator_bath"]


def derive_slb(doc, levels):
    """levels -> {"cost": [...], "rmse": [...], "sem": [...]} per level, plus
    the swept M values, from the saved raw runs."""
    reference = as_array(doc["reference"])
    m_values = [row["M"] for row in doc["slb_sweep"]]
    slb = {n: {"cost": [], "rmse": [], "sem": []} for n in levels}
    for row in doc["slb_sweep"]:
        samples = np.asarray(row["samples"], dtype=float)
        for n in levels:
            _, sem, rmse = tavg_bias_sem_rmse(samples[:n], reference)
            slb[n]["cost"].append(n * row["per_run_cost"])
            slb[n]["rmse"].append(rmse)
            slb[n]["sem"].append(sem)
    return m_values, slb


def figure(name, doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = doc["meta"]["params"]
    guard = doc["guard"]
    flag = ("OK" if guard["ok"]
            else f"WARNING: bias moved {guard['rel_change']:.0%} on "
                 f"{guard['substeps'][0]}->{guard['substeps'][1]} substeps -- "
                 f"RAISE SUBSTEPS and re-run run_frontier.py")
    print(f"  substeps guard from the run (M={guard['M']}): "
          f"bias {guard['bias'][0]:.3e} -> {guard['bias'][1]:.3e}  [{flag}]")

    levels = [n for n in N_RUNS_SWEEP if n <= meta["N_RUNS_MAX"]]
    dropped = [n for n in N_RUNS_SWEEP if n > meta["N_RUNS_MAX"]]
    if dropped:
        print(f"  note: levels {dropped} exceed the {meta['N_RUNS_MAX']} saved "
              f"runs and are dropped")

    m_values, slb = derive_slb(doc, levels)
    mc = doc["mc"]

    # blue gradient generated from the levels, so any choice of run counts
    # works (lighter = fewer runs, darker = more)
    _blues = plt.cm.Blues(np.linspace(0.45, 0.9, len(levels)))
    slb_blues = {n: _blues[i] for i, n in enumerate(sorted(levels))}

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for n in levels:
        d = slb[n]
        ax.errorbar(d["cost"], d["rmse"], yerr=d["sem"],
                    fmt="s-", color=slb_blues[n], lw=1.8, ms=6, capsize=3,
                    label=f"SLB (N={n} runs)")
    n_hi = max(levels)                             # label the darkest curve
    d_hi = slb[n_hi]
    for x, y, m_eff in zip(d_hi["cost"], d_hi["rmse"], m_values):
        ax.annotate(f"M={m_eff}", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color=slb_blues[n_hi])

    ax.errorbar([r["cost"] for r in mc], [r["rmse"] for r in mc],
                yerr=[r["sem"] for r in mc],
                fmt="o--", color="tab:purple", lw=1.8, ms=7, capsize=3,
                label="qutip.mcsolve")
    for r in mc:
        ax.annotate(f"{r['ntraj']}", (r["cost"], r["rmse"]),
                    textcoords="offset points",
                    xytext=(5, -11), fontsize=7, color="tab:purple")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock cost (s)   (lower is better)")
    ax.set_ylabel(r"time-averaged RMSE in $\langle H\rangle$   (lower is better)")
    ax.set_title(
        rf"{name} (dim {doc['dim']}, $N_L$={doc['n_l']}): RMSE-vs-cost frontier"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    per_run = [row["per_run_cost"] for row in doc["slb_sweep"]]
    tmin, tmax = min(per_run) * 1000, max(per_run) * 1000
    slb_caption = (f"SLB: sweep M={m_values}, {SUBSTEPS} RK4 substep(s)/step, "
                   f"N in {levels} runs (one run {tmin:.0f}-{tmax:.0f} ms)")
    mc_caption = (f"mcsolve: sweep ntraj={[r['ntraj'] for r in mc]}, "
                  f"single-thread, atol={MC_ATOL:g}/rtol={MC_RTOL:g}")
    add_settings_footer(
        fig, slb_caption, mc_caption,
        "metric = time-averaged RMSE; error bar = S/sqrt(N), the uncertainty "
        "of the single estimate shown; full-Lindblad reference",
        "serial timing: both methods parallelize across runs/trajectories "
        "alike, so k cores shift both curves equally",
        fontsize=11,
    )
    fig.savefig(f"benchmark_frontier_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benchmark_frontier_{name}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--system", default="all",
                    help="a system name with a data file (default preset names "
                         "or big-preset names), or 'all' for the default pair")
    args = ap.parse_args()
    names = DEFAULT_SYSTEMS if args.system == "all" else [args.system]
    for name in names:
        print(f"[{name}]")
        figure(name, load_data(f"frontier_{name}.json"))


if __name__ == "__main__":
    main()
