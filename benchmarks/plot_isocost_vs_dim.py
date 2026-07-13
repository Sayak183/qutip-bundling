"""
plot_isocost_vs_dim.py
======================

UPDATED: Iso-accuracy cost-vs-dimension benchmark (Result 4).
Displays the wall-clock cost to reach a specified TARGET_RMSE for SLB 
(at various averaging levels) versus mcsolve.

Includes an automated Error Budget subplot (Bias² vs SEM²) cascading
below the main plot, calculated strictly from the trajectories plotted.
Supports different N-run lists for different systems.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from common import add_settings_footer, as_array, load_data, tavg_rmse

# --- CONFIGURATION ---
TARGET_RMSE = 0.02       # accuracy target; must stay >= the sweep floor recorded in the data
NTRAJ_EXTRAP_MAX = 20000 # Beyond this, mcsolve is considered impractical

# Define specific averaging levels to plot for each system
SYSTEM_N_RUNS = {
    "spin_chain": [4, 8, 16],
    "oscillator_bath": [16,32,64]
}
# ---------------------

def derive_slb(point, n_runs, target):
    """
    Returns (m_star, cost, reached, bias_sq, sem_sq) for one dimension 
    at one specific averaging level.
    """
    last = None
    reference = as_array(point["reference"])
    
    for row in point["slb_sweep"]:
        # Extract exactly n_runs
        samples = np.asarray(row["samples"], dtype=float)[:n_runs]
        
        # 1. Total Observed MSE (exact error of the plotted point)
        ensemble_mean = np.mean(samples, axis=0)
        total_mse = np.mean((ensemble_mean - reference)**2)
        rmse = np.sqrt(total_mse)
        
        # 2. Statistical Variance (SEM^2) strictly from these n_runs
        # ddof=1 ensures unbiased sample variance estimate over the trajectories
        if n_runs > 1:
            var_single_run = np.mean(np.var(samples, axis=0, ddof=1))
        else:
            var_single_run = 0.0
        sem_sq = var_single_run / n_runs
        
        # 3. Implied Systematic Bias^2
        bias_sq = max(0.0, total_mse - sem_sq)
        
        last = (row["M"], n_runs * row["per_run_cost"], True, bias_sq, sem_sq)
        
        if rmse <= target:
            return last
            
    # If the target was never reached, return data for the largest M tested
    last = (row["M"], n_runs * row["per_run_cost"], False, bias_sq, sem_sq)
    return last

def derive_mc(point, target):
    """(ntraj_star, cost, reachable) derived from saved S^2 fit."""
    s2 = np.mean([np.mean(np.square(r["rmse_repeats"])) * r["ntraj"]
                  for r in point["mc_fit"]])
    t_per_traj = np.mean([r["per_traj_time"] for r in point["mc_fit"]])
    ntraj_star = float(s2) / (target ** 2)
    reachable = ntraj_star <= NTRAJ_EXTRAP_MAX
    cost = float(t_per_traj) * min(ntraj_star, NTRAJ_EXTRAP_MAX)
    return ntraj_star, cost, reachable

def derive(doc, target, n_runs_list):
    points = doc["points"]
    out = {
        "dims": as_array([p["dim"] for p in points]),
        "full_cost": as_array([p["t_full"] for p in points]),
        "slb": {}, "mc_cost": [], "mc_star": [], "mc_ok": [],
    }
    
    for n in n_runs_list:
        rows = [derive_slb(p, n, target) for p in points]
        out["slb"][n] = {
            "mstar": np.array([r[0] for r in rows]),
            "cost": np.array([r[1] for r in rows]),
            "ok": np.array([r[2] for r in rows]),
            "bias_sq": np.array([r[3] for r in rows]),
            "sem_sq": np.array([r[4] for r in rows]),
        }
        
    for p in points:
        nt, c, ok = derive_mc(p, target)
        out["mc_star"].append(nt)
        out["mc_cost"].append(c)
        out["mc_ok"].append(ok)
        
    out["mc_cost"] = np.array(out["mc_cost"])
    out["mc_ok"] = np.array(out["mc_ok"])
    return out

def figure(name, out, target, substeps, n_runs_list):
    plt.switch_backend("Agg")
    d = out["dims"]
    if len(d) == 0: return

    # Create a 2-panel figure with the main plot on top and the bar chart below
    fig, (ax_main, ax_bar) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 8), 
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.25}
    )

    # --- TOP PANEL: Main Scaling Plot ---
    # Plot exact reference line
    ax_main.loglog(d, out["full_cost"], "o:", color="tab:gray", lw=1.5, ms=5, alpha=0.6,
                   label="exact mesolve (reference)")
    
    # Plot SLB lines
    greens = {4: "#a1d99b", 8: "#41ab5d", 16: "#006d2c"}
    for n in n_runs_list:
        c = greens.get(n, "tab:green")
        ax_main.loglog(d, out["slb"][n]["cost"], "s-", color=c, lw=2, ms=7,
                       label=f"SLB (N={n} runs)")

    # Plot mcsolve line
    ax_main.loglog(d, out["mc_cost"], "o-", color="tab:purple", lw=2, ms=8,
                   label="mcsolve (tune ntraj)")

    # Line Annotations
    n_max = max(n_runs_list)
    for x, y, m, ok in zip(d, out["slb"][n_max]["cost"], out["slb"][n_max]["mstar"],
                           out["slb"][n_max]["ok"]):
        ax_main.annotate(f"M*={int(m)}", (x, y), xytext=(5, -12), textcoords="offset points", 
                         fontsize=9, color=greens.get(n_max, "tab:green"))
        
    for x, y, nt, ok in zip(d, out["mc_cost"], out["mc_star"], out["mc_ok"]):
        label = f"ntraj≈{int(round(nt)):,}" if (ok and np.isfinite(nt)) else f"ntraj≳{NTRAJ_EXTRAP_MAX:,}"
        ax_main.annotate(label, (x, y), xytext=(5, 6), textcoords="offset points", 
                         fontsize=9, color="tab:purple")

    ax_main.set_ylabel("wall-clock cost to reach target (s)")
    ax_main.set_title(f"{name}: cost to reach RMSE={target} — SLB vs mcsolve")
    ax_main.legend(loc="upper left", fontsize=9)
    ax_main.grid(True, which="both", alpha=0.3)


    # --- BOTTOM PANEL: Error Budget Bar Chart ---
    # Calculate widths and offsets to handle multiple N-levels automatically
    width = 0.8 / len(n_runs_list) if len(n_runs_list) > 0 else 0.4
    offsets = np.linspace(-width*(len(n_runs_list)-1)/2, width*(len(n_runs_list)-1)/2, len(n_runs_list)) if len(n_runs_list) > 1 else [0]
    
    # Unified colors for the bar chart
    color_bias = '#006400' # Dark green
    color_sem  = '#98FB98' # Light green
    
    x_positions = np.arange(len(d))
    
    for idx, n in enumerate(n_runs_list):
        bias_sq_vals = out["slb"][n]["bias_sq"]
        sem_sq_vals  = out["slb"][n]["sem_sq"]
        
        # Only label the very first iteration to prevent legend duplicates
        lbl_b = 'Systematic Bias²' if x_positions[0]==0 and idx==0 else ""
        lbl_s = 'Statistical SEM²' if x_positions[0]==0 and idx==0 else ""
        
        # Added edgecolor='black' to separate the grouped bars
        ax_bar.bar(x_positions + offsets[idx], bias_sq_vals, width, 
                   color=color_bias, edgecolor='black', linewidth=0.7, label=lbl_b)
        ax_bar.bar(x_positions + offsets[idx], sem_sq_vals, width, bottom=bias_sq_vals, 
                   color=color_sem, edgecolor='black', linewidth=0.7, label=lbl_s)
        
        # Add N-level label above each bar so they remain identifiable
        for i, (bias, sem) in enumerate(zip(bias_sq_vals, sem_sq_vals)):
            total_height = bias + sem
            y_offset = (target**2) * 0.02
            ax_bar.text(x_positions[i] + offsets[idx], total_height + y_offset, f"N={n}", 
                        ha='center', va='bottom', fontsize=7, color='black')

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([str(dim) for dim in d])
    ax_bar.set_title(r"MSE Budget at $M^*$ (Bias vs. Statistical Noise)", fontsize=10)
    ax_bar.set_xlabel(r"Hilbert-space dimension $N$")
    ax_bar.set_ylabel("MSE")
    ax_bar.grid(True, axis='y', alpha=0.3)
    
    target_mse = target**2
    ax_bar.axhline(target_mse, color='red', linestyle='--', linewidth=1.5, label='Target MSE')
    
    # Legend outside the bar chart to keep it clean
    ax_bar.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # Add footer and save
    add_settings_footer(
        fig,
        f"iso-accuracy: smallest M at each N={n_runs_list} runs reaching RMSE={target}; "
        f"{substeps} RK4 substep(s)/step",
        "computable only to the exact-reference wall; '≳' = mcsolve needs impractical trajectory count",
        fontsize=9,
    )
    
    out_name = f"benchmark_isocost_vs_dim_{name}.png"
    fig.savefig(out_name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all", choices=["spin_chain", "oscillator_bath", "all"])
    ap.add_argument("--target", type=float, default=TARGET_RMSE)
    args = ap.parse_args()
    
    names = ["spin_chain", "oscillator_bath"] if args.system == "all" else [args.system]
    for name in names:
        doc = load_data(f"isocost_vs_dim_{name}.json")
        # Fetch the specific N runs list for this system, defaulting to [4]
        n_runs_list = SYSTEM_N_RUNS.get(name, [4])
        
        out = derive(doc, args.target, n_runs_list)
        figure(name, out, args.target, doc["meta"]["substeps"], n_runs_list)

if __name__ == "__main__":
    main()