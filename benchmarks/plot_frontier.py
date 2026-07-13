"""
plot_frontier.py
================
UPDATED: Frontier benchmark with individual toggles and an error bar switch.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from common import (
    add_settings_footer, as_array, load_data, tavg_bias_sem_rmse,
    MC_ATOL, MC_RTOL,
)

# --- CONFIGURATION (TOGGLES FOR FRONTIER PLOT) ---
N_RUNS_SWEEP = [16, 32] 
PLOT_RMSE = True
PLOT_BIAS = False
PLOT_SEM  = False
PLOT_STD  = False

# Choose what the error bars on the RMSE curves represent:
# Set to "SEM" for Standard Error of the Mean, or "STD" for Standard Deviation
ERROR_BAR_TYPE = "SEM" 

DEFAULT_SYSTEMS = ["spin_chain", "oscillator_bath"]
# -------------------------------------------------

def derive_slb_stats(doc, n_runs):
    reference = as_array(doc["reference"])
    m_values = [row["M"] for row in doc["slb_sweep"]]
    stats = {"cost": [], "rmse": [], "sem": [], "bias": [], "std": []}
    for row in doc["slb_sweep"]:
        samples = np.asarray(row["samples"], dtype=float)
        bias, sem, rmse = tavg_bias_sem_rmse(samples[:n_runs], reference)
        stats["cost"].append(n_runs * row["per_run_cost"])
        stats["rmse"].append(rmse)
        stats["sem"].append(sem)
        stats["bias"].append(bias)
        # Calculate standard deviation from SEM
        stats["std"].append(sem * np.sqrt(n_runs))
    return m_values, stats

def figure(name, doc):
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(N_RUNS_SWEEP)))

    # Re-derive stats for the highest N for M-labeling
    n_hi = max(N_RUNS_SWEEP)
    m_vals_hi, stats_hi = derive_slb_stats(doc, n_hi)

    for i, n in enumerate(N_RUNS_SWEEP):
        m_values, stats = derive_slb_stats(doc, n)
        
        if PLOT_RMSE:
            # Apply the selected error bar type
            err_vals = stats["std"] if ERROR_BAR_TYPE == "STD" else stats["sem"]
            ax.errorbar(stats["cost"], stats["rmse"], yerr=err_vals,
                        fmt="s-", color=colors[i], lw=2, label=f"SLB (N={n}) - RMSE")
            
        if PLOT_BIAS:
            ax.plot(stats["cost"], stats["bias"], "o:", color=colors[i], alpha=0.5, label=f"Bias (N={n})")
        if PLOT_SEM:
            ax.plot(stats["cost"], stats["sem"], "v:", color=colors[i], alpha=0.5, label=f"SEM (N={n})")
        if PLOT_STD:
            ax.plot(stats["cost"], stats["std"], "d:", color=colors[i], alpha=0.5, label=f"Std Dev (N={n})")

    # Label M values on the darkest (highest N) curve
    # Attach labels to RMSE by default, fallback to other active metrics if RMSE is hidden
    y_label_vals = stats_hi["rmse"]
    if not PLOT_RMSE:
        if PLOT_BIAS: y_label_vals = stats_hi["bias"]
        elif PLOT_SEM: y_label_vals = stats_hi["sem"]
        elif PLOT_STD: y_label_vals = stats_hi["std"]

    for x, y, m_eff in zip(stats_hi["cost"], y_label_vals, m_vals_hi):
        ax.annotate(f"M={m_eff}", (x, y), textcoords="offset points", 
                    xytext=(6, 6), fontsize=9, color=colors[-1], fontweight='bold')

    mc = doc["mc"]
    if PLOT_RMSE:
        # Apply the selected error bar type for mcsolve
        if ERROR_BAR_TYPE == "STD":
            mc_err = [r["sem"] * np.sqrt(r["ntraj"]) for r in mc]
        else:
            mc_err = [r["sem"] for r in mc]
            
        ax.errorbar([r["cost"] for r in mc], [r["rmse"] for r in mc],
                    yerr=mc_err, fmt="o--", color="tab:purple", lw=1.8, ms=7, capsize=3, label="qutip.mcsolve")
        for r in mc:
            ax.annotate(f"{r['ntraj']}", (r["cost"], r["rmse"]), textcoords="offset points", xytext=(5, -11), fontsize=7, color="tab:purple")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock cost (s)   (lower is better)")
    ax.set_ylabel(r"time-averaged RMSE in $\langle H\rangle$   (lower is better)")
    ax.set_title(rf"{name} (dim {doc['dim']}, $N_L$={doc['n_l']})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    add_settings_footer(
        fig,
        f"SLB: sweep M={m_vals_hi}, {doc['meta']['substeps']} RK4 substep(s)/step "
        f"(from the run's own metadata), N in {N_RUNS_SWEEP} runs",
        f"mcsolve: sweep ntraj={[r['ntraj'] for r in doc['mc']]}, single-thread, "
        f"atol={MC_ATOL:g}/rtol={MC_RTOL:g}",
        "error bar = S/sqrt(N), the uncertainty of the single estimate shown; "
        "both methods parallelize across runs/trajectories alike",
    )
    fig.savefig(f"benchmark_frontier_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all")
    args = ap.parse_args()
    names = DEFAULT_SYSTEMS if args.system == "all" else [args.system]
    for name in names:
        print(f"[{name}]")
        figure(name, load_data(f"frontier_{name}.json"))
        print(f"  saved benchmark_frontier_{name}.png")

if __name__ == "__main__":
    main()