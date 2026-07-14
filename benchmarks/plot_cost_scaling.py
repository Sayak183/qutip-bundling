"""
plot_cost_scaling.py
====================

UPDATED: Cost scaling benchmark with Davies construction time included
and an easily accessible TARGET_RMSE switch.
"""

from __future__ import annotations

import argparse
import numpy as np

from common import (
    add_settings_footer, as_array, load_data,
)

# --- CONFIGURATION (TARGET ACCURACY SWITCH) ---
# Published default 0.02; override per-figure with --target for exploration.
TARGET_RMSE = 0.02
SHOW_ERROR_BUDGET = True   # stacked bias^2/SEM^2 bars under the cost panel,
                           # for the M* points (mirrors Result 4's budget);
                           # answers "is the iso slope physics or noise?"
# ----------------------------------------------

def derive_iso(points, target):
    """(m_star, iso_cost, achieved_rmse, bias_sq, sem_sq).

    bias^2/SEM^2 come from the run's stored MSE budget (same decomposition as
    Result 4: implied bias^2 = observed MSE - SEM^2); NaN for data files from
    before the budget was recorded."""
    out = []
    for p in points:
        row = (np.nan, np.nan, np.nan, np.nan, np.nan)
        for e in p["m_sweep"] or []:
            if e.get("rmse") is not None and e["rmse"] <= target:
                mse, sem_sq = e.get("mse"), e.get("sem_sq")
                if mse is None or sem_sq is None:
                    b2 = s2 = np.nan
                else:
                    b2, s2 = max(0.0, mse - sem_sq), sem_sq
                row = (e["M"], e["cost"], e["rmse"], b2, s2)
                break
        out.append(row)
    return [np.array(v, dtype=float) for v in zip(*out)]

def fixed_m_stats(points, m_rep):
    """(rmse, bias, sem) for the fixed M_REP."""
    rmse, bias, sem = [], [], []
    n_acc = 16
    for p in points:
        hit = next((e for e in p["m_sweep"] or []
                    if e["M"] == min(m_rep, p["n_l"])), None)
        if hit is None or hit.get("rmse") is None:
            rmse.append(np.nan); bias.append(np.nan); sem.append(np.nan)
        else:
            fluct = hit["rmse_std"] * np.sqrt(n_acc)
            s = fluct / np.sqrt(n_acc)
            b = np.sqrt(max(0, hit["rmse"]**2 - s**2))
            rmse.append(hit["rmse"]); bias.append(b); sem.append(s)
    return np.array(rmse), np.array(bias), np.array(sem)

def fit_slope(dims, times):
    m = np.isfinite(times)
    d, t = dims[m], times[m]
    if len(d) < 2: return None
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
    points = doc["points"]
    dims = as_array([p["dim"] for p in points])
    n_ls = as_array([p["n_l"] for p in points])
    t_full = as_array([p["t_full"] for p in points])
    t_slb = as_array([p["t_slb_fixed"] for p in points])
    t_dav = as_array([p.get("t_davies") for p in points])
    
    mstar, iso_cost, iso_rmse, iso_b2, iso_s2 = derive_iso(points, target)
    budget_ok = SHOW_ERROR_BUDGET and np.isfinite(iso_b2).any()
    if SHOW_ERROR_BUDGET and not budget_ok:
        print("  note: no MSE budget in this data file (older run) - "
              "re-run run_cost_scaling.py to enable the budget panel")
    if budget_ok:
        fig, (ax, ax_bar) = plt.subplots(
            2, 1, figsize=(8.5, 7.4), gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(8.5, 5.6))

    # ---- Cost Scaling (single panel) ----
    ff = np.isfinite(t_full)
    ax.loglog(dims[ff], t_full[ff], "o-", color="tab:red", lw=2, label=_slope_label("full mesolve", dims, t_full))
    # extrapolation guide: continue the exact solver past its feasibility wall
    # at its fitted slope, so the divergence from SLB is visible, not implied.
    s_full = fit_slope(dims, t_full)
    if s_full and ff.any() and dims[~ff & np.isfinite(dims)].size:
        d0, y0 = dims[ff][-1], t_full[ff][-1]
        d_ext = np.array(sorted(set(dims[dims >= d0])))
        if len(d_ext) > 1:
            ax.loglog(d_ext, y0 * (d_ext / d0) ** s_full, "--", color="tab:red",
                      lw=1.4, alpha=0.45,
                      label=rf"full mesolve, extrapolated $\propto N^{{{s_full:.1f}}}$")
    ax.loglog(dims, t_slb, "s-", color="tab:green", lw=2, label=_slope_label(f"fixed M={m_rep}", dims, t_slb))
    ii = np.isfinite(iso_cost)
    ax.loglog(dims[ii], iso_cost[ii], "^--", color="tab:blue", lw=2, label=_slope_label(f"iso-accuracy (target {target})", dims, iso_cost))
    
    # Plot the Davies Construction Time
    dd = np.isfinite(t_dav)
    ax.loglog(dims[dd], t_dav[dd], "x:", color="gray", lw=1.5, alpha=0.8, label=_slope_label("Davies construction", dims, t_dav))

    # Annotate M*
    for x, y, ms in zip(dims[ii], iso_cost[ii], mstar[ii]):
        ax.annotate(f"M*={int(ms)}", (x, y), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=8)

    ax.set_ylabel("wall-clock time (s)")
    # ticks at the measured points, each labelled with BOTH sizes: the Hilbert
    # dimension and the Lindblad operator count it implies -- N_L is what
    # actually drives the full solver's cost and the bundling payoff.
    ax.set_xticks(dims)
    ax.set_xticklabels([f"{int(d)}\n" + rf"$N_L$={int(l)}"
                        for d, l in zip(dims, n_ls)], fontsize=8)
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.set_xlabel("Hilbert dimension N  (with Lindblad operator count $N_L$)")
    ax.set_title(f"{name}: cost scaling with M* annotations")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    if budget_ok:
        # MSE budget at each M* point, mirroring Result 4's error budget:
        # bias-dominated bars mean the iso slope is the bundling physics
        # (M* must grow to cut bias); noise-dominated bars would mean more
        # sampling could lower it. Fractions of the observed MSE.
        jj = np.isfinite(iso_b2)
        tot = iso_b2[jj] + iso_s2[jj]
        xpos = np.arange(jj.sum())
        ax_bar.bar(xpos, iso_b2[jj] / tot, 0.55, color="tab:blue",
                   edgecolor="black", label=r"systematic bias$^2$")
        ax_bar.bar(xpos, iso_s2[jj] / tot, 0.55, bottom=iso_b2[jj] / tot,
                   color="lightsteelblue", edgecolor="black",
                   label=r"statistical noise (SEM$^2$)")
        ax_bar.set_xticks(xpos)
        ax_bar.set_xticklabels([f"{int(d)}\n" + rf"$M^*$={int(m)}"
                                for d, m in zip(dims[jj], mstar[jj])],
                               fontsize=8)
        ax_bar.set_ylim(0, 1.12)
        ax_bar.set_ylabel("MSE share")
        ax_bar.set_title(r"MSE budget at $M^*$ (bias vs. statistical noise)",
                         fontsize=10)
        ax_bar.legend(fontsize=7, ncol=2, loc="upper right")

    fig.tight_layout()
    segs = [
        f"fixed-M: one SLB realization (M={m_rep}); iso-accuracy: smallest swept M "
        f"with {n_acc}-run RMSE<={target} vs exact (target applied at analysis time)",
        f"{meta['substeps']} RK4 substep(s)/step (from the run's own metadata); "
        f"Davies construction timed separately and included in no solve time",
    ]
    val = doc.get("native_ref_validation")
    if any(str(p.get("reference_method", "")).startswith("native")
           for p in points):
        v = (f", validated vs mesolve at dim {val['dim']} "
             f"(max dev {val['max_abs_dev']:.0e})") if val else ""
        segs.append("iso-accuracy beyond the wall uses a full-dissipator "
                    f"native-RK4 reference{v}; the exact-cost curve remains "
                    "qutip mesolve")
    add_settings_footer(fig, *segs)
    out = f"benchmark_cost_scaling_{name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all")
    # Command line argument overrides the hardcoded config if provided
    ap.add_argument("--target", type=float, default=TARGET_RMSE)
    args = ap.parse_args()
    names = ["spin_chain", "oscillator_bath"] if args.system == "all" else [args.system]
    for name in names:
        figure(name, load_data(f"cost_scaling_{name}.json"), args.target)

if __name__ == "__main__":
    main()