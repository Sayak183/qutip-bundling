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
    add_settings_footer, as_array, load_data, SUBSTEPS,
)

# --- CONFIGURATION (TARGET ACCURACY SWITCH) ---
# Change this value to loosen or tighten the iso-accuracy requirement.
# Default was 0.02. We are setting it to 0.05 to observe the curve flattening.
TARGET_RMSE = 0.05
# ----------------------------------------------

def derive_iso(points, target):
    """(m_star, iso_cost, achieved_rmse, achieved_bias, achieved_sem)."""
    out = []
    for p in points:
        row = (np.nan, np.nan, np.nan, np.nan, np.nan)
        for e in p["m_sweep"] or []:
            if e["rmse"] is not None and e["rmse"] <= target:
                n_acc = 16 
                fluct = e["rmse_std"] * np.sqrt(n_acc)
                sem = fluct / np.sqrt(n_acc)
                bias = np.sqrt(max(0, e["rmse"]**2 - sem**2))
                row = (e["M"], e["cost"], e["rmse"], bias, sem)
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
        if hit is None or hit["rmse"] is None:
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
    t_full = as_array([p["t_full"] for p in points])
    t_slb = as_array([p["t_slb_fixed"] for p in points])
    t_dav = as_array([p.get("t_davies") for p in points])
    
    mstar, iso_cost, iso_rmse, iso_bias, iso_sem = derive_iso(points, target)
    fix_rmse, fix_bias, fix_sem = fixed_m_stats(points, m_rep)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.5, 9.0), sharex=True, 
                                  gridspec_kw={"height_ratios": [2, 1.5]})

    # ---- Top: Cost Scaling ----
    ff = np.isfinite(t_full)
    ax.loglog(dims[ff], t_full[ff], "o-", color="tab:red", lw=2, label=_slope_label("full mesolve", dims, t_full))
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
    ax.set_title(f"{name}: cost scaling with M* annotations")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    # ---- Bottom: Error Decomposition ----
    axr.loglog(dims, fix_rmse, "s-", color="tab:green", label="RMSE (fixed M)")
    axr.loglog(dims[ii], iso_rmse[ii], "^--", color="tab:blue", label="RMSE (iso-accuracy M*)")
    axr.loglog(dims[ii], iso_bias[ii], "v:", color="tab:blue", alpha=0.5, label="Bias (iso-accuracy)")
    axr.loglog(dims[ii], iso_sem[ii], ".:", color="tab:blue", alpha=0.5, label="SEM (iso-accuracy)")
    
    axr.axhline(target, color="black", ls="--", alpha=0.3, label="target")
    axr.set_ylabel("Error components")
    axr.set_xlabel("Hilbert dimension N")
    axr.legend(fontsize=7, ncol=2); axr.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = f"benchmark_cost_scaling_{name}.png"
    fig.savefig(out, dpi=110); plt.close(fig)
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