"""
plot_isocost_vs_dim.py
======================

UPDATED: Iso-accuracy cost-vs-dimension benchmark (Result 4).
Displays the wall-clock cost to reach TARGET_REL of each observable's own span, for SLB 
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
from isocost_config import run_counts

# --- CONFIGURATION ---
# Accuracy target, as a fraction of each observable's own span over the
# reference trajectory. A single ABSOLUTE tolerance cannot serve six observables
# that differ by three orders of magnitude in scale: at oscillator dim 64, an
# RMSE of 0.02 is 0.008% of n2's span and 374% of the coherence's. Scoring
# against each observable's own scale is one standard for all of them.
#
# 3% is close to what the old absolute 0.02 already demanded of the energy on
# the chains (4% at mixed dim 64, 2.8% at spin dim 512), so the chains barely
# move; the oscillator, where the artefact lived, moves a lot.
TARGET_REL = 0.03

# Kept for the --target flag and for single-observable legacy files.
TARGET_RMSE = 0.02       # absolute fallback
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

# Sampling levels come from isocost_config.SYSTEM_N_RUNS, shared with the
# data-generation script. Edit them there, then regenerate the data before
# plotting if the largest configured count increases.
# ---------------------

def observable_targets(point, rel=None):
    """Per-observable accuracy target: ``rel`` times that observable's span.

    The span is taken over the reference trajectory, so the target is a fixed
    fraction of the signal each observable actually shows. Returns one target
    per observable, in the same order as the reference rows.
    """
    reference, _ = _obs_axis(point)
    rel = TARGET_REL if rel is None else rel
    span = reference.max(axis=1) - reference.min(axis=1)
    return rel * np.maximum(span, 1e-12)


def _obs_axis(point):
    """(reference (n_obs, n_times), labels).

    Old single-observable files load as a one-entry set labelled "energy", so
    both schemas go through the same code and the section can be replotted
    before every system has been regenerated.
    """
    reference = np.atleast_2d(as_array(point["reference"]))
    labels = point.get("observables") or ["energy"]
    if len(labels) != reference.shape[0]:
        labels = [f"obs{j}" for j in range(reference.shape[0])]
    return reference, labels


def derive_slb(point, n_runs, target, est_type):
    """(m_star, cost, reached, bias_sq, noise_sq, binding) for one dimension.

    ``m_star`` is the cheapest bundle count reaching the target on EVERY
    observable, not on the energy alone -- the energy is the easiest quantity
    this suite measures, so tuning to it reports the best case rather than the
    cost of using the method. ``binding`` names the observable that sets it, and
    the bias/noise split returned is that observable's, since it is the one the
    operating point is chosen for.
    """
    last = None
    reference, labels = _obs_axis(point)
    # A scalar target is applied to every observable; the vector form is what
    # observable_targets() supplies, one entry per observable.
    target = np.broadcast_to(np.asarray(target, dtype=float),
                             (reference.shape[0],))

    for row in point["slb_sweep"]:
        # Extract exactly n_runs to compute statistics. NumPy slicing silently
        # returns fewer rows when the request is too large, so validate first:
        # a 32-run label must never be derived from a 16-run dataset.
        all_samples = np.asarray(row["samples"], dtype=float)
        available = all_samples.shape[0]
        if n_runs > available:
            raise ValueError(
                f"Result 4 requests N_r={n_runs} for dim {point['dim']}, "
                f"M={row['M']}, but the data contains only {available} runs. "
                f"Rerun run_isocost_vs_dim.py with the shared SYSTEM_N_RUNS "
                f"configuration, or lower the configured count, then replot."
            )
        samples = all_samples[:n_runs]
        if samples.ndim == 2:                 # legacy file: energy only
            samples = samples[:, None, :]
        n_actual = samples.shape[0]

        # Per observable, so the binding one can be identified rather than
        # averaged away. Averaging the MSE across observables would let one
        # large coherence error hide behind five small ones.
        ensemble_mean = np.mean(samples, axis=0)                 # (n_obs, n_t)
        total_mse = np.mean((ensemble_mean - reference) ** 2, axis=1)
        if n_actual > 1:
            var_single_run = np.mean(np.var(samples, axis=0, ddof=1), axis=1)
        else:
            var_single_run = np.zeros(samples.shape[1])
        sem_sq = var_single_run / n_actual
        bias_sq = np.maximum(0.0, total_mse - sem_sq)

        if est_type == "single":
            noise_sq = var_single_run
            cost = row["per_run_cost"]                   # COST OF 1 RUN
        else:
            noise_sq = sem_sq
            cost = n_actual * row["per_run_cost"]        # COST OF N RUNS

        rmse = np.sqrt(bias_sq + noise_sq)
        # "Worst" means furthest from its OWN target, not largest in absolute
        # terms -- otherwise the biggest-scale observable is always named.
        worst = int(np.argmax(rmse / target))
        last = (row["M"], cost, True,
                float(bias_sq[worst]), float(noise_sq[worst]), labels[worst])

        # Every observable must clear its own target, not the first to do so.
        if bool(np.all(rmse <= target)):
            return last

    # Target never reached: report the largest M tried, and which observable
    # was still missing it there.
    return (row["M"], cost, False,
            float(bias_sq[worst]), float(noise_sq[worst]), labels[worst])

def derive_mc(point, target):
    """
    (ntraj_star, cost, reachable) derived from saved S^2 fit.
    mcsolve ALWAYS calculates the ensemble cost required to reach the target.
    """
    # rmse_repeats is (repeat,) on old files and (repeat, observable) on new
    # ones. mcsolve must clear every observable's own target, exactly as SLB
    # must, so S^2 is taken per observable and the binding one is whichever
    # needs the most trajectories -- not whichever has the largest raw error.
    def _s2_of(r):
        # PREFER s_repeats: the spread ACROSS trajectories, which is what S
        # means. Files written before that field existed fall back to
        # rmse_repeats, which OVERESTIMATES S.
        #
        # tavg_rmse combines (sample mean - reference)^2 with SEM^2, and for an
        # unbiased estimator the realized deviation of the sample mean IS
        # sampling fluctuation with variance SEM^2 -- the same quantity counted
        # twice. Measured on the mixed chain at dim 16, rmse * sqrt(ntraj)
        # overestimates S by 1.23-1.46x, so ntraj* = (S/target)^2 comes out
        # 1.5-2.1x too large and every projected mcsolve cost with it. On a
        # fallback file the resulting speedups are upper bounds.
        direct = r.get("s_repeats")
        if direct is not None:
            a = np.asarray(direct, dtype=float)
            per_obs = (np.mean(a, axis=0) if a.ndim == 2
                       else np.atleast_1d(np.mean(a)))
            return np.square(np.asarray(per_obs, dtype=float))

        a = np.asarray(r["rmse_repeats"], dtype=float)
        per_obs = (np.mean(np.square(a), axis=0) if a.ndim == 2
                   else np.atleast_1d(np.mean(np.square(a))))
        return np.asarray(per_obs, dtype=float) * r["ntraj"]

    s2 = np.mean([_s2_of(r) for r in point["mc_fit"]], axis=0)
    target = np.broadcast_to(np.asarray(target, dtype=float), s2.shape)
    t_per_traj = np.mean([r["per_traj_time"] for r in point["mc_fit"]])
    
    # Ensemble can be averaged down
    # The binding observable is the one demanding the most trajectories.
    ntraj_star = float(np.max(s2 / (target ** 2)))
    reachable = ntraj_star <= NTRAJ_EXTRAP_MAX
    cost = float(t_per_traj) * min(ntraj_star, NTRAJ_EXTRAP_MAX) # COST OF ENSEMBLE
        
    return ntraj_star, cost, reachable

def derive(doc, target, n_runs_list, est_type):
    points = doc["points"]
    out = {
        "dims": as_array([p["dim"] for p in points]),
        "n_ls": as_array([p["n_l"] for p in points]),
        "full_cost": as_array([p["t_full"] for p in points]),
        "slb": {}, "mc_cost": [], "mc_star": [], "mc_ok": [],
    }
    
    # One target vector per point: TARGET_REL of each observable's own span at
    # that dimension. Passing the scalar through would restore the artefact.
    targets = {id(p): observable_targets(p) for p in points}

    for n in n_runs_list:
        rows = [derive_slb(p, n, targets[id(p)], est_type) for p in points]
        out["slb"][n] = {
            "mstar": np.array([r[0] for r in rows]),
            "cost": np.array([r[1] for r in rows]),
            "ok": np.array([r[2] for r in rows]),
            "bias_sq": np.array([r[3] for r in rows]),
            "noise_sq": np.array([r[4] for r in rows]),

            "binding": [r[5] for r in rows],
        }
        
    for p in points:
        nt, c, ok = derive_mc(p, targets[id(p)])
        out["mc_star"].append(nt)
        out["mc_cost"].append(c)
        out["mc_ok"].append(ok)
        
    out["mc_cost"] = np.array(out["mc_cost"])
    out["mc_ok"] = np.array(out["mc_ok"])
    return out

def figure(name, out, target, substeps_text, n_runs_list, est_type):
    plt.switch_backend("Agg")
    d = out["dims"]
    n_ls = out["n_ls"]
    if len(d) == 0: return

    fig, (ax_main, ax_bar) = plt.subplots(
        nrows=2, ncols=1, figsize=(9.5, 9), 
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.55}
    )

    # --- TOP PANEL: Main Scaling Plot ---
    ax_main.loglog(d, out["full_cost"], "o:", color="tab:gray", lw=1.5, ms=5, alpha=0.6,
                   label="exact mesolve (reference)")
    
    greens = {4: "#006d2c", 8: "#006d2c", 16: "#006d2c", 32: "#00441b", 64: "#002a11"}
    for n in n_runs_list:
        c = greens.get(n, "tab:green")
        if est_type == "single":
            lbl = f"SLB 1-run (stats from $N_r$={n})"
        else:
            lbl = f"SLB Ensemble ($N_r$={n} runs)"
            
        ax_main.loglog(d, out["slb"][n]["cost"], "s-", color=c, lw=2, ms=7, label=lbl)

    # mcsolve is now always presented as an ensemble
    ax_main.loglog(d, out["mc_cost"], "o-", color="tab:purple", lw=2, ms=8, label="mcsolve (tune ntraj to target)")

    # Line Annotations
    n_max = max(n_runs_list)
    bindings = out["slb"][n_max].get("binding") or [None] * len(d)
    for x, y, m, ok, bind in zip(d, out["slb"][n_max]["cost"],
                                 out["slb"][n_max]["mstar"],
                                 out["slb"][n_max]["ok"], bindings):
        # Name the observable M* satisfies LAST. Without it the reader sees a
        # larger M* than the energy-only figures reported, with no reason why.
        tag = f"M*={int(m)}" + (f"\n({bind})" if bind and bind != "energy" else "")
        ax_main.annotate(tag, (x, y), xytext=(5, -12), textcoords="offset points",
                         fontsize=10, color=greens.get(n_max, "tab:green"))
        
    for x, y, nt, ok in zip(d, out["mc_cost"], out["mc_star"], out["mc_ok"]):
        label = f"ntraj≈{int(round(nt)):,}" if (ok and np.isfinite(nt)) else f"ntraj≳{NTRAJ_EXTRAP_MAX:,}"
        ax_main.annotate(label, (x, y), xytext=(5, 6), textcoords="offset points", 
                         fontsize=11, color="tab:purple")

    ax_main.set_ylabel("wall-clock cost to reach target (s)", fontsize=13)
    ax_main.set_xlabel("Hilbert dimension N  (with Lindblad operator count $N_L$)",
                       fontsize=13)
    ax_main.set_xticks(d)
    ax_main.set_xticklabels([f"{int(dd)}\n" + rf"$N_L$={int(nl)}"
                             for dd, nl in zip(d, n_ls)], fontsize=11)
    ax_main.minorticks_off()
    est_label_cap = "Ensemble" if est_type == "ensemble" else "Single-Run"
    ax_main.set_title(
        f"{name}: cost to reach {TARGET_REL:.0%} of each observable's span"
        f" — SLB vs mcsolve", fontsize=14)
    ax_main.legend(loc="upper left", fontsize=11)
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
                        ha='center', va='bottom', fontsize=9, color='black')

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([f"{int(dd)}\n" + rf"$N_L$={int(nl)}"
                            for dd, nl in zip(d, n_ls)], fontsize=11)
    ax_bar.set_title(rf"MSE Budget at $M^*$ (Bias² vs. {noise_str})", fontsize=13)
    ax_bar.set_xlabel(r"Hilbert-space dimension $N$", fontsize=13)
    ax_bar.set_ylabel("MSE", fontsize=13)
    ax_bar.grid(True, axis='y', alpha=0.3)
    
    target_mse = target**2
    ax_bar.axhline(target_mse, color='red', linestyle='--', linewidth=1.5, label='Target MSE')
    ax_bar.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # Add explicit math string to footer
    footer_math = "Ens RMSE\u00b2 = bias\u00b2 + SEM\u00b2" if est_type == "ensemble" else "Single RMSE\u00b2 = bias\u00b2 + Std\u00b2"

    segs = [
        f"iso-accuracy: smallest M whose {est_type} RMSE clears "
        f"{TARGET_REL:.0%} of EVERY observable's own span (the binding one is "
        f"named at each point); one absolute tolerance cannot serve "
        f"observables differing by 10^3 in scale; "
        f"{substeps_text}",
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
    ap.add_argument("--system", default="all",
                    choices=["spin_chain", "mixed_chain", "oscillator_bath", "all"])
    ap.add_argument("--target", type=float, default=TARGET_RMSE)
    args = ap.parse_args()
    
    names = (["spin_chain", "mixed_chain", "oscillator_bath"]
             if args.system == "all" else [args.system])
    for name in names:
        doc = load_data(f"isocost_vs_dim_{name}.json")
        n_runs_list = run_counts(name)
        
        out = derive(doc, args.target, n_runs_list, ESTIMATE_TYPE)
        pairs = [(int(p["dim"]), int(p["substeps"])) for p in doc["points"]]
        unique_substeps = sorted({ss for _, ss in pairs})
        if len(unique_substeps) == 1:
            substeps_text = f"{unique_substeps[0]} RK4 substep(s)/step"
        else:
            settings = ", ".join(f"dim {dim}: {ss}" for dim, ss in pairs)
            substeps_text = f"RK4 substeps/step by dimension ({settings})"
        figure(
            name, out, args.target, substeps_text, n_runs_list, ESTIMATE_TYPE,
        )

if __name__ == "__main__":
    main()
