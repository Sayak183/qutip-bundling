"""
plot_frontier.py
================
UPDATED: Frontier benchmark with M-labels restored and toggle for Bias/SEM.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from common import (
    add_settings_footer, as_array, load_data, tavg_bias_sem_rmse,
    MC_ATOL, MC_RTOL, SUBSTEPS,
)

# --- CONFIGURATION ---
N_RUNS_SWEEP = [2, 4] 
PLOT_DECOMPOSITION = True  # Toggle for Bias/SEM
DEFAULT_SYSTEMS = ["spin_chain", "oscillator_bath"]
# ---------------------

def derive_slb_stats(doc, n_runs):
    reference = as_array(doc["reference"])
    m_values = [row["M"] for row in doc["slb_sweep"]]
    stats = {"cost": [], "rmse": [], "sem": [], "bias": []}
    for row in doc["slb_sweep"]:
        samples = np.asarray(row["samples"], dtype=float)
        bias, sem, rmse = tavg_bias_sem_rmse(samples[:n_runs], reference)
        stats["cost"].append(n_runs * row["per_run_cost"])
        stats["rmse"].append(rmse)
        stats["sem"].append(sem)
        stats["bias"].append(bias)
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
        ax.errorbar(stats["cost"], stats["rmse"], yerr=stats["sem"],
                    fmt="s-", color=colors[i], lw=2, label=f"SLB (N={n}) - RMSE")
        
        if PLOT_DECOMPOSITION:
            ax.plot(stats["cost"], stats["bias"], "v:", color=colors[i], alpha=0.5, label=f"Bias (N={n})")
            ax.plot(stats["cost"], stats["sem"], ".:", color=colors[i], alpha=0.5, label=f"SEM (N={n})")

    # RESTORED: Label M values on the darkest (highest N) curve
    for x, y, m_eff in zip(stats_hi["cost"], stats_hi["rmse"], m_vals_hi):
        ax.annotate(f"M={m_eff}", (x, y), textcoords="offset points", 
                    xytext=(6, 6), fontsize=9, color=colors[-1], fontweight='bold')

    mc = doc["mc"]
    ax.errorbar([r["cost"] for r in mc], [r["rmse"] for r in mc],
                yerr=[r["sem"] for r in mc], fmt="o--", color="tab:purple", lw=1.8, ms=7, capsize=3, label="qutip.mcsolve")
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