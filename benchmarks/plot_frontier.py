"""
plot_frontier.py
================
UPDATED: Frontier benchmark with separate switches for Ensemble RMSE and Single RMSE.
N_RUNS_SWEEP is system-specific.

CRITICAL FIX: mcsolve ALWAYS plots its ensemble convergence curve (cost vs ensemble RMSE), 
providing a baseline to compare against SLB's 1-run or N-run performance.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from common import (
    add_settings_footer, as_array, load_data,
    MC_ATOL, MC_RTOL,
)

# --- CONFIGURATION (TOGGLES FOR FRONTIER PLOT) ---
# Define different N sweeps for each system here:
N_RUNS_SWEEP = {
    "spin_chain": [2, 4, 8],   
    "oscillator_bath": [8,16]   
}

PLOT_ENSEMBLE_RMSE = True
PLOT_SINGLE_RMSE   = False
PLOT_BIAS = False
PLOT_SEM  = False
PLOT_STD  = False

# Choose what the error bars on the RMSE curves represent ("SEM" or "STD")
ERROR_BAR_TYPE = "SEM" 

DEFAULT_SYSTEMS = ["spin_chain", "oscillator_bath"]
# -------------------------------------------------

def derive_slb_stats(doc, n_runs):
    reference = as_array(doc["reference"])
    m_values = [row["M"] for row in doc["slb_sweep"]]
    stats = {
        "cost_ens": [], "cost_single": [], 
        "rmse_ens": [], "rmse_single": [], 
        "sem": [], "bias": [], "std": []
    }
    
    for row in doc["slb_sweep"]:
        # Extract the raw trajectories and truncate to the requested n_runs
        samples = np.asarray(row["samples"], dtype=float)[:n_runs]
        
        # Calculate time-dependent statistics
        mean = samples.mean(axis=0)
        std = samples.std(axis=0, ddof=1)
        bias = np.abs(mean - reference)
        sem = std / np.sqrt(n_runs)
        
        # Calculate the two types of RMSE
        rmse_ens = np.sqrt(bias**2 + sem**2)
        rmse_single = np.sqrt(bias**2 + std**2)
        
        # Time-average the results
        stats["cost_ens"].append(n_runs * row["per_run_cost"])
        stats["cost_single"].append(row["per_run_cost"])
        
        stats["rmse_ens"].append(float(np.mean(rmse_ens)))
        stats["rmse_single"].append(float(np.mean(rmse_single)))
        stats["sem"].append(float(np.mean(sem)))
        stats["bias"].append(float(np.mean(bias)))
        stats["std"].append(float(np.mean(std)))
        
    return m_values, stats

def figure(name, doc):
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    
    # Grab the specific N sweep for the current system
    current_n_sweep = N_RUNS_SWEEP.get(name, [16, 32])
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(current_n_sweep)))

    # Re-derive stats for the highest N for M-labeling
    n_hi = max(current_n_sweep)
    m_vals_hi, stats_hi = derive_slb_stats(doc, n_hi)

    for i, n in enumerate(current_n_sweep):
        m_values, stats = derive_slb_stats(doc, n)
        
        # Determine error bars
        err_vals = stats["std"] if ERROR_BAR_TYPE == "STD" else stats["sem"]
            
        if PLOT_ENSEMBLE_RMSE:
            ax.errorbar(stats["cost_ens"], stats["rmse_ens"], yerr=err_vals,
                        fmt="s-", color=colors[i], lw=2, label=f"SLB Ens ($N_r$={n})")
        
        if PLOT_SINGLE_RMSE:
            ax.errorbar(stats["cost_single"], stats["rmse_single"], yerr=err_vals,
                        fmt="X-", color=colors[i], lw=2, label=f"SLB 1-run (stats from $N_r$={n})")
            
        if PLOT_BIAS:
            x_vals = stats["cost_ens"] if PLOT_ENSEMBLE_RMSE else stats["cost_single"]
            ax.plot(x_vals, stats["bias"], "o:", color=colors[i], alpha=0.5, label=f"Bias ($N_r$={n})")
        if PLOT_SEM:
            x_vals = stats["cost_ens"] if PLOT_ENSEMBLE_RMSE else stats["cost_single"]
            ax.plot(x_vals, stats["sem"], "v:", color=colors[i], alpha=0.5, label=f"SEM ($N_r$={n})")
        if PLOT_STD:
            x_vals = stats["cost_ens"] if PLOT_ENSEMBLE_RMSE else stats["cost_single"]
            ax.plot(x_vals, stats["std"], "d:", color=colors[i], alpha=0.5, label=f"Std Dev ($N_r$={n})")

    # Label M values on the darkest (highest N) curve
    y_label_vals = None
    x_label_vals = None
    
    if PLOT_ENSEMBLE_RMSE: 
        y_label_vals = stats_hi["rmse_ens"]
        x_label_vals = stats_hi["cost_ens"]
    elif PLOT_SINGLE_RMSE: 
        y_label_vals = stats_hi["rmse_single"]
        x_label_vals = stats_hi["cost_single"]
    elif PLOT_BIAS: 
        y_label_vals = stats_hi["bias"]
        x_label_vals = stats_hi["cost_ens"]

    if y_label_vals is not None and x_label_vals is not None:
        for x, y, m_eff in zip(x_label_vals, y_label_vals, m_vals_hi):
            ax.annotate(f"M={m_eff}", (x, y), textcoords="offset points", 
                        xytext=(6, 6), fontsize=9, color=colors[-1], fontweight='bold')

    mc = doc["mc"]
    # FAIRNESS: mcsolve must be shown under the SAME error definition as SLB.
    # Its ensemble is the ntraj-trajectory average (rmse = sqrt(bias^2+SEM^2),
    # cost = the full run); its single-run analogue is ONE trajectory
    # (Std = SEM*sqrt(ntraj), cost = the run divided by ntraj). Plotting SLB
    # single-run against an mcsolve ensemble would be a rigged race -- and it
    # would understate our own method, which is the frontier figure's whole
    # point.
    mc_bias = [r["bias"] for r in mc]
    mc_sem = [r["sem"] for r in mc]
    mc_std = [r["sem"] * np.sqrt(r["ntraj"]) for r in mc]
    if PLOT_SINGLE_RMSE and not PLOT_ENSEMBLE_RMSE:
        mc_cost_ens = [r["cost"] / r["ntraj"] for r in mc]
        mc_rmse_ens = [float(np.sqrt(b ** 2 + s ** 2))
                       for b, s in zip(mc_bias, mc_std)]
        mc_label = "mcsolve (1 trajectory)"
    else:
        mc_cost_ens = [r["cost"] for r in mc]
        mc_rmse_ens = [r["rmse"] for r in mc]
        mc_label = "mcsolve (ensemble)"

    mc_err = mc_std if ERROR_BAR_TYPE == "STD" else mc_sem

    ax.errorbar(mc_cost_ens, mc_rmse_ens, yerr=mc_err, fmt="o--", color="tab:purple", 
                lw=1.8, ms=7, capsize=3, label=mc_label)
    for r, c, y in zip(mc, mc_cost_ens, mc_rmse_ens):
        ax.annotate(f"{r['ntraj']}", (c, y), textcoords="offset points", xytext=(5, -11), fontsize=7, color="tab:purple")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock cost (s)   (lower is better)")
    ax.set_ylabel(r"time-averaged error in $\langle H\rangle$   (lower is better)")
    ax.set_title(rf"{name} (dim {doc['dim']}, $N_L$={doc['n_l']})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    
    if PLOT_ENSEMBLE_RMSE and not PLOT_SINGLE_RMSE:
        footer_math = "Ens RMSE\u00b2 = bias\u00b2 + SEM\u00b2"
    elif PLOT_SINGLE_RMSE and not PLOT_ENSEMBLE_RMSE:
        footer_math = "Single RMSE\u00b2 = bias\u00b2 + StdDev\u00b2"
    else:
        footer_math = "Ens RMSE\u00b2 = bias\u00b2 + SEM\u00b2 | Single RMSE\u00b2 = bias\u00b2 + StdDev\u00b2"
    
    add_settings_footer(
        fig,
        f"SLB: sweep M={m_vals_hi}, {doc['meta']['substeps']} RK4 substep(s)/step "
        f"(from the run's own metadata), N in {current_n_sweep} runs",
        f"mcsolve: sweep ntraj={[r['ntraj'] for r in doc['mc']]}, single-thread, "
        f"atol={MC_ATOL:g}/rtol={MC_RTOL:g}",
        f"{footer_math}; error bar = {ERROR_BAR_TYPE}",
    )
    
    suffix = ""
    if not PLOT_ENSEMBLE_RMSE: suffix += "_noEns"
    if PLOT_SINGLE_RMSE: suffix += "_Single"
    if PLOT_BIAS: suffix += "_Bias"
    if PLOT_SEM: suffix += "_SEM"
    if PLOT_STD: suffix += "_STD"
    
    out_file = f"benchmark_frontier_{name}{suffix}.png"
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all")
    args = ap.parse_args()
    names = DEFAULT_SYSTEMS if args.system == "all" else [args.system]
    for name in names:
        print(f"[{name}]")
        figure(name, load_data(f"frontier_{name}.json"))
        print(f"  saved plot")

if __name__ == "__main__":
    main()