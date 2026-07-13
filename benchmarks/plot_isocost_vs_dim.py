"""
plot_isocost_vs_dim.py
======================

UPDATED: Iso-accuracy cost-vs-dimension benchmark (Result 4).
Displays the wall-clock cost to reach a specified TARGET_RMSE for SLB 
(at various averaging levels) versus mcsolve.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from common import add_settings_footer, as_array, load_data, tavg_rmse

# --- CONFIGURATION ---
TARGET_RMSE = 0.02   # accuracy target; must stay >= the sweep floor recorded in the data (0.01), else M* is unreachable -- use --target for exploration
N_RUNS_LIST = [4, 8, 16]        # Averaging levels shown
NTRAJ_EXTRAP_MAX = 20000        # Beyond this, mcsolve is considered impractical
# ---------------------

def derive_slb(point, n_runs, target):
    """(m_star, cost, reached) for one dimension at one averaging level."""
    last = None
    for row in point["slb_sweep"]:
        samples = np.asarray(row["samples"], dtype=float)[:n_runs]
        rmse = tavg_rmse(samples, as_array(point["reference"]))
        last = row
        if rmse <= target:
            return row["M"], n_runs * row["per_run_cost"], True
    return last["M"], n_runs * last["per_run_cost"], False

def derive_mc(point, target):
    """(ntraj_star, cost, reachable) derived from saved S^2 fit."""
    s2 = np.mean([np.mean(np.square(r["rmse_repeats"])) * r["ntraj"]
                  for r in point["mc_fit"]])
    t_per_traj = np.mean([r["per_traj_time"] for r in point["mc_fit"]])
    ntraj_star = float(s2) / (target ** 2)
    reachable = ntraj_star <= NTRAJ_EXTRAP_MAX
    cost = float(t_per_traj) * min(ntraj_star, NTRAJ_EXTRAP_MAX)
    return ntraj_star, cost, reachable

def derive(doc, target):
    points = doc["points"]
    out = {
        "dims": as_array([p["dim"] for p in points]),
        "full_cost": as_array([p["t_full"] for p in points]),
        "slb": {}, "mc_cost": [], "mc_star": [], "mc_ok": [],
    }
    for n in N_RUNS_LIST:
        rows = [derive_slb(p, n, target) for p in points]
        out["slb"][n] = {
            "mstar": np.array([r[0] for r in rows]),
            "cost": np.array([r[1] for r in rows]),
            "ok": np.array([r[2] for r in rows]),
        }
    for p in points:
        nt, c, ok = derive_mc(p, target)
        out["mc_star"].append(nt); out["mc_cost"].append(c); out["mc_ok"].append(ok)
    out["mc_cost"] = np.array(out["mc_cost"])
    out["mc_ok"] = np.array(out["mc_ok"])
    return out

def figure(name, out, target, substeps):
    plt.switch_backend("Agg")
    d = out["dims"]
    if len(d) == 0: return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Plot lines
    ax.loglog(d, out["full_cost"], "o:", color="tab:gray", lw=1.5, ms=5, alpha=0.6,
              label="exact mesolve (reference)")
    
    greens = {4: "#a1d99b", 8: "#41ab5d", 16: "#006d2c"}
    for n in N_RUNS_LIST:
        c = greens.get(n, "tab:green")
        ax.loglog(d, out["slb"][n]["cost"], "s-", color=c, lw=2, ms=7,
                  label=f"SLB (N={n} runs)")

    ax.loglog(d, out["mc_cost"], "o-", color="tab:purple", lw=2, ms=8,
              label="mcsolve (tune ntraj)")

    # Annotations
    n_max = max(N_RUNS_LIST)
    for x, y, m, ok in zip(d, out["slb"][n_max]["cost"], out["slb"][n_max]["mstar"],
                           out["slb"][n_max]["ok"]):
        ax.annotate(f"M*={int(m)}", (x, y), xytext=(5, -12), textcoords="offset points", 
                    fontsize=8, color=greens[n_max])
    for x, y, nt, ok in zip(d, out["mc_cost"], out["mc_star"], out["mc_ok"]):
        label = f"ntraj≈{int(round(nt)):,}" if (ok and np.isfinite(nt)) else f"ntraj≳{NTRAJ_EXTRAP_MAX:,}"
        ax.annotate(label, (x, y), xytext=(5, 6), textcoords="offset points", 
                    fontsize=8, color="tab:purple")

    ax.set_ylabel("wall-clock cost to reach target (s)")
    ax.set_xlabel(r"Hilbert-space dimension $N$")
    ax.set_title(f"{name}: cost to reach RMSE={target} — SLB vs mcsolve")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    add_settings_footer(
        fig,
        f"iso-accuracy: smallest M at each N={N_RUNS_LIST} runs reaching RMSE={target}; "
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
        figure(name, derive(doc, args.target), args.target,
               doc["meta"]["substeps"])

if __name__ == "__main__":
    main()