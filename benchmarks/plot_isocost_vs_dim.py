"""
plot_isocost_vs_dim.py
======================

UPDATED: Iso-accuracy cost-vs-dimension benchmark (Result 4).
Displays the wall-clock cost to reach a specified TARGET_RMSE for SLB 
versus mcsolve. 

CRITICAL FIX: mcsolve always calculates the cost of the ensemble required 
to hit the target. SLB plots the cost of 1 run (if ESTIMATE_TYPE="single") 
or N runs (if "ensemble") to reach that exact same target.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from common import add_settings_footer, as_array, load_data, tavg_rmse

# --- CONFIGURATION ---
TARGET_RMSE = 0.02       # accuracy target
# Which estimate's error must reach the target?
#   "ensemble" : SLB's N-run average -> sqrt(bias^2 + SEM^2). The LIKE-FOR-LIKE
#                comparison: mcsolve reaches the target the only way it can --
#                by averaging ntraj trajectories -- so both methods are then
#                judged as averaged estimates. This is the headline definition.
#   "single"   : SLB must reach the target with ONE run -> sqrt(bias^2+Std^2).
#                NOTE THE ASYMMETRY: mcsolve CANNOT be measured this way (one
#                trajectory's error is fixed at S; no ntraj makes a single
#                trajectory accurate), so it is still shown at its averaged
#                optimum. SLB is therefore held to a STRICTER standard than its
#                competitor and the resulting speedups UNDERSTATE it. The figure
#                says so on its face. Useful for exploring, conservative to
#                quote.
ESTIMATE_TYPE = "ensemble" # Options: "ensemble" or "single"
NTRAJ_EXTRAP_MAX = 20000 # Beyond this, mcsolve is considered impractical

# Define specific sampling levels to plot for each system
SYSTEM_N_RUNS = {
    "spin_chain": [4, 8, 16],
    "oscillator_bath": [16, 32, 64]
}
# ---------------------

def derive_slb(point, n_runs, target, est_type):
    """
    Returns (m_star, cost, reached, bias_sq, noise_sq) for one dimension.
    """
    last = None
    reference = as_array(point["reference"])
    
    for row in point["slb_sweep"]:
        # Extract exactly n_runs to compute statistics
        samples = np.asarray(row["samples"], dtype=float)[:n_runs]
        
        # 1. Total Observed MSE of the ensemble mean
        ensemble_mean = np.mean(samples, axis=0)
        total_mse = np.mean((ensemble_mean - reference)**2)
        
        # 2. Statistical Variance
        if n_runs > 1:
            var_single_run = np.mean(np.var(samples, axis=0, ddof=1)) # This is Std^2
        else:
            var_single_run = 0.0
        sem_sq = var_single_run / n_runs
        
        # 3. Implied Systematic Bias^2
        bias_sq = max(0.0, total_mse - sem_sq)
        
        # 4. Apply toggle logic for Cost and RMSE
        if est_type == "single":
            noise_sq = var_single_run
            cost = row["per_run_cost"] # COST OF 1 RUN
        else:
            noise_sq = sem_sq
            cost = n_runs * row["per_run_cost"] # COST OF N RUNS
            
        rmse = np.sqrt(bias_sq + noise_sq)
        
        last = (row["M"], cost, True, bias_sq, noise_sq)
        
        if rmse <= target:
            return last
            
    # If target never reached, return data for largest M tested
    last = (row["M"], cost, False, bias_sq, noise_sq)
    return last

def derive_mc(point, target):
    """
    (ntraj_star, cost, reachable) derived from saved S^2 fit.
    mcsolve ALWAYS calculates the ensemble cost required to reach the target.
    """
    s2 = np.mean([np.mean(np.square(r["rmse_repeats"])) * r["ntraj"]
                  for r in point["mc_fit"]])
    t_per_traj = np.mean([r["per_traj_time"] for r in point["mc_fit"]])
    
    # Ensemble can be averaged down
    ntraj_star = float(s2) / (target ** 2)
    reachable = ntraj_star <= NTRAJ_EXTRAP_MAX
    cost = float(t_per_traj) * min(ntraj_star, NTRAJ_EXTRAP_MAX) # COST OF ENSEMBLE
        
    return ntraj_star, cost, reachable

def derive(doc, target, n_runs_list, est_type):
    points = doc["points"]
    out = {
        "dims": as_array([p["dim"] for p in points]),
        "full_cost": as_array([p["t_full"] for p in points]),
        "slb": {}, "mc_cost": [], "mc_star": [], "mc_ok": [],
    }
    
    for n in n_runs_list:
        rows = [derive_slb(p, n, target, est_type) for p in points]
        out["slb"][n] = {
            "mstar": np.array([r[0] for r in rows]),
            "cost": np.array([r[1] for r in rows]),
            "ok": np.array([r[2] for r in rows]),
            "bias_sq": np.array([r[3] for r in rows]),
            "noise_sq": np.array([r[4] for r in rows]),
        }
        
    for p in points:
        nt, c, ok = derive_mc(p, target)
        out["mc_star"].append(nt)
        out["mc_cost"].append(c)
        out["mc_ok"].append(ok)
        
    out["mc_cost"] = np.array(out["mc_cost"])
    out["mc_ok"] = np.array(out["mc_ok"])
    return out

def figure(name, out, target, substeps, n_runs_list, est_type):
    plt.switch_backend("Agg")
    d = out["dims"]
    if len(d) == 0: return

    fig, (ax_main, ax_bar) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 8), 
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.25}
    )

    # --- TOP PANEL: Main Scaling Plot ---
    ax_main.loglog(d, out["full_cost"], "o:", color="tab:gray", lw=1.5, ms=5, alpha=0.6,
                   label="exact mesolve (reference)")
    
    greens = {4: "#a1d99b", 8: "#41ab5d", 16: "#006d2c", 32: "#00441b", 64: "#002a11"}
    for n in n_runs_list:
        c = greens.get(n, "tab:green")
        if est_type == "single":
            lbl = f"SLB 1-run (stats from N={n})"
        else:
            lbl = f"SLB Ensemble (N={n} runs)"
            
        ax_main.loglog(d, out["slb"][n]["cost"], "s-", color=c, lw=2, ms=7, label=lbl)

    # mcsolve is now always presented as an ensemble
    ax_main.loglog(d, out["mc_cost"], "o-", color="tab:purple", lw=2, ms=8, label="mcsolve (tune ntraj to target)")

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
    est_label_cap = "Ensemble" if est_type == "ensemble" else "Single-Run"
    ax_main.set_title(f"{name}: cost to reach {est_label_cap} RMSE={target} — SLB vs mcsolve")
    ax_main.legend(loc="upper left", fontsize=9)
    ax_main.grid(True, which="both", alpha=0.3)


    # --- BOTTOM PANEL: Error Budget Bar Chart ---
    width = 0.8 / len(n_runs_list) if len(n_runs_list) > 0 else 0.4
    offsets = np.linspace(-width*(len(n_runs_list)-1)/2, width*(len(n_runs_list)-1)/2, len(n_runs_list)) if len(n_runs_list) > 1 else [0]
    
    color_bias = '#006400' 
    color_noise  = '#98FB98' 
    x_positions = np.arange(len(d))
    noise_str = "SEM²" if est_type == "ensemble" else "Std²"

    for idx, n in enumerate(n_runs_list):
        bias_sq_vals = out["slb"][n]["bias_sq"]
        noise_sq_vals  = out["slb"][n]["noise_sq"]
        
        lbl_b = 'Systematic Bias²' if x_positions[0]==0 and idx==0 else ""
        lbl_s = f'Statistical {noise_str}' if x_positions[0]==0 and idx==0 else ""
        
        ax_bar.bar(x_positions + offsets[idx], bias_sq_vals, width, 
                   color=color_bias, edgecolor='black', linewidth=0.7, label=lbl_b)
        ax_bar.bar(x_positions + offsets[idx], noise_sq_vals, width, bottom=bias_sq_vals, 
                   color=color_noise, edgecolor='black', linewidth=0.7, label=lbl_s)
        
        for i, (bias, noise) in enumerate(zip(bias_sq_vals, noise_sq_vals)):
            total_height = bias + noise
            y_offset = (target**2) * 0.02
            ax_bar.text(x_positions[i] + offsets[idx], total_height + y_offset, f"N={n}", 
                        ha='center', va='bottom', fontsize=7, color='black')

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([str(dim) for dim in d])
    ax_bar.set_title(rf"MSE Budget at $M^*$ (Bias² vs. {noise_str})", fontsize=10)
    ax_bar.set_xlabel(r"Hilbert-space dimension $N$")
    ax_bar.set_ylabel("MSE")
    ax_bar.grid(True, axis='y', alpha=0.3)
    
    target_mse = target**2
    ax_bar.axhline(target_mse, color='red', linestyle='--', linewidth=1.5, label='Target MSE')
    ax_bar.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # Add explicit math string to footer
    footer_math = "Ens RMSE\u00b2 = bias\u00b2 + SEM\u00b2" if est_type == "ensemble" else "Single RMSE\u00b2 = bias\u00b2 + Std\u00b2"

    segs = [
        f"iso-accuracy: smallest M reaching {est_type} RMSE={target}; "
        f"{substeps} RK4 substep(s)/step",
        f"{footer_math}; computable only to the exact-reference wall; '≳' = mcsolve needs impractical trajectory count",
    ]
    if est_type == "single":
        segs.append(
            "ASYMMETRIC COMPARISON: SLB must reach the target with ONE run, "
            "while mcsolve is shown at its averaged ntraj* optimum (a single "
            "trajectory can never reach the target) -- these speedups therefore "
            "UNDERSTATE SLB; the like-for-like numbers are the ensemble ones")
    add_settings_footer(fig, *segs, fontsize=9)
    
    suffix = f"_{est_type}" if est_type != "ensemble" else ""
    out_name = f"benchmark_isocost_vs_dim_{name}{suffix}.png"
    
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
        n_runs_list = SYSTEM_N_RUNS.get(name, [4])
        
        out = derive(doc, args.target, n_runs_list, ESTIMATE_TYPE)
        figure(name, out, args.target, doc["meta"]["substeps"], n_runs_list, ESTIMATE_TYPE)

if __name__ == "__main__":
    main()