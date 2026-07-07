"""
plot_cost_scaling.py
====================

ANALYSIS/FIGURE HALF of the cost-scaling benchmark (Result 2). Reads the data
written by run_cost_scaling.py and draws the two-panel figure; runs in seconds,
so figure styling and even the accuracy target can be iterated without
re-running the (slow) sweeps.

The iso-accuracy quantities are DERIVED HERE from the saved M sweep:

    M*(N)      = smallest swept M whose N_ACC-run RMSE <= TARGET_RMSE
    iso cost   = wall-clock of that M* estimate
    achieved   = the RMSE that M* actually delivers (hugs the target from
                 below, in discrete steps, because M is swept on a grid)

The figure (unchanged filename: benchmark_cost_scaling_<system>.png):

  (top)    wall-clock vs N: full mesolve, SLB fixed-M (one solve), SLB
           iso-accuracy, each with its measured log-log slope; feasibility wall.
  (bottom) fixed-M RMSE vs N with jackknife error bars (climbing: the mechanism
           forcing M to grow), the target line, and the M*-achieved RMSE curve
           with its error bars -- the M* labels sit on THIS curve, at the RMSE
           each M* actually attains, not on the fixed-M curve.

Run:  python plot_cost_scaling.py [--system ...] [--target 0.02]
"""

from __future__ import annotations

import argparse

import numpy as np

from common import (
    add_settings_footer, as_array, load_data, SUBSTEPS,
)

TARGET_RMSE = 0.02      # accuracy target defining the iso-accuracy curve;
                        # analysis-time choice, changeable without re-running


def derive_iso(points, target):
    """Per dimension: (m_star, iso_cost, achieved_rmse, achieved_std), NaN where
    the sweep has no reference or never reaches the target."""
    out = []
    for p in points:
        row = (np.nan, np.nan, np.nan, np.nan)
        for e in p["m_sweep"] or []:
            if e["rmse"] is not None and e["rmse"] <= target:
                row = (e["M"], e["cost"], e["rmse"],
                       np.nan if e["rmse_std"] is None else e["rmse_std"])
                break
        out.append(row)
    return [np.array(v, dtype=float) for v in zip(*out)]


def fixed_m_rows(points, m_rep):
    """RMSE (+std) of the M_REP entry of each sweep (the fixed-M accuracy curve)."""
    rmse, std = [], []
    for p in points:
        hit = next((e for e in p["m_sweep"] or []
                    if e["M"] == min(m_rep, p["n_l"])), None)
        rmse.append(np.nan if hit is None or hit["rmse"] is None else hit["rmse"])
        std.append(np.nan if hit is None or hit["rmse_std"] is None
                   else hit["rmse_std"])
    return np.array(rmse), np.array(std)


def fit_slope(dims, times):
    m = np.isfinite(times)
    d, t = dims[m], times[m]
    if len(d) < 2:
        return None
    if len(d) >= 4:
        sel = d >= d.max() / 8.0
        if sel.sum() >= 3:
            d, t = d[sel], t[sel]
    return float(np.polyfit(np.log(d), np.log(t), 1)[0])


def _slope_label(base, dims, times):
    e = fit_slope(dims, times)
    return base + (rf"  — $\propto N^{{{e:.1f}}}$" if e else "")


def figure(name, doc, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = doc["meta"]
    m_rep, n_acc = meta["params"]["M_REP"], meta["params"]["N_ACC"]
    floor = meta["params"]["SWEEP_STOP_RMSE"]
    if target < floor:
        print(f"  WARNING: target {target} is below the sweep floor {floor}; "
              f"the saved sweep may stop before reaching it - re-run "
              f"run_cost_scaling.py with a lower SWEEP_STOP_RMSE.")

    points = doc["points"]
    dims = as_array([p["dim"] for p in points])
    t_full = as_array([p["t_full"] for p in points])
    t_slb = as_array([p["t_slb_fixed"] for p in points])
    t_dav = as_array([p.get("t_davies") for p in points])  # absent in old data
    wall_dim = doc["wall_dim"]
    rmse_fix, rmse_fix_std = fixed_m_rows(points, m_rep)
    mstar, iso_cost, iso_rmse, iso_rmse_std = derive_iso(points, target)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.25]}
    )

    # ---- top: cost ----
    ff = np.isfinite(t_full)
    ax.loglog(dims[ff], t_full[ff], "o-", color="tab:red", lw=2, ms=7,
              label=_slope_label("full mesolve (exact)", dims, t_full))
    ax.loglog(dims, t_slb, "s-", color="tab:green", lw=2, ms=7,
              label=_slope_label(f"SLB, fixed M={m_rep}", dims, t_slb))
    ii = np.isfinite(iso_cost)
    ax.loglog(dims[ii], iso_cost[ii], "^--", color="tab:blue", lw=2, ms=8,
              label=_slope_label(f"SLB, iso-accuracy (RMSE={target})",
                                 dims, iso_cost))
    dd = np.isfinite(t_dav)
    if dd.any():   # construction cost: NOT part of any solve time above
        ax.loglog(dims[dd], t_dav[dd], "d:", color="tab:orange", lw=1.8, ms=7,
                  label=_slope_label(r"Davies construction ($N_L$ operators)",
                                     dims, t_dav))
    if wall_dim is not None:
        ax.axvline(wall_dim, color="tab:red", ls="--", alpha=0.5)
        ax.text(wall_dim, ax.get_ylim()[0] * 1.6, "full mesolve\nimpractical past here",
                color="tab:red", fontsize=8, ha="center", va="bottom")
    ax.set_ylabel("wall-clock time for one estimate (s)")
    ax.set_title(f"{name}: cost scaling — fixed-M is cheapest, iso-accuracy is the honest scaling")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ---- bottom: fixed-M accuracy (with error bars) + target + the RMSE the
    # iso-accuracy choice M* actually achieves (labels live on THAT curve) ----
    rr = np.isfinite(rmse_fix)
    axr.errorbar(dims[rr], rmse_fix[rr], yerr=rmse_fix_std[rr], fmt="s-",
                 color="tab:green", lw=2, ms=7, capsize=3,
                 label=f"RMSE at fixed M={m_rep}")
    axr.axhline(target, color="tab:blue", ls="--", alpha=0.7,
                label=f"target {target}")
    jj = np.isfinite(iso_rmse)
    axr.errorbar(dims[jj], iso_rmse[jj], yerr=iso_rmse_std[jj], fmt="^--",
                 color="tab:blue", lw=1.6, ms=8, capsize=3, alpha=0.9,
                 label=r"RMSE achieved at $M^*$ (iso-accuracy)")
    for x, y, ms in zip(dims[jj], iso_rmse[jj], mstar[jj]):
        axr.annotate(f"M*={int(ms)}", (x, y), textcoords="offset points",
                     xytext=(6, -11), fontsize=8, color="tab:blue",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                               alpha=0.7))
    if wall_dim is not None:
        axr.axvline(wall_dim, color="tab:red", ls="--", alpha=0.5)
    axr.set_yscale("log")
    axr.set_xscale("log")
    axr.set_ylabel(f"time-avg RMSE\n(SLB, {n_acc} runs)")
    axr.set_xlabel(r"Hilbert-space dimension $N$")
    axr.legend(loc="lower right", fontsize=8)
    axr.grid(True, which="both", alpha=0.3)
    axr.margins(y=0.3)

    add_settings_footer(
        fig,
        f"fixed-M: one SLB realization (M={m_rep}); iso-accuracy: smallest swept M "
        f"with {n_acc}-run RMSE<={target} vs exact (target applied at analysis time)",
        f"{SUBSTEPS} RK4 substep(s)/step; error bars: delete-one jackknife over the "
        f"{n_acc} runs; iso-accuracy computable only up to the reference wall",
        "Davies construction timed separately and included in no solve time",
        "operation-count expectation: O(N^3) per solve for SLB vs O(N^5) for full mesolve",
    )
    out = f"benchmark_cost_scaling_{name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--system", default="all",
                    choices=["spin_chain", "oscillator_bath", "all"])
    ap.add_argument("--target", type=float, default=TARGET_RMSE,
                    help=f"iso-accuracy RMSE target (default {TARGET_RMSE})")
    args = ap.parse_args()
    names = (["spin_chain", "oscillator_bath"] if args.system == "all"
             else [args.system])
    for name in names:
        figure(name, load_data(f"cost_scaling_{name}.json"), args.target)


if __name__ == "__main__":
    main()
