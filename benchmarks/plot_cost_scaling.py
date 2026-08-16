"""
plot_cost_scaling.py
====================

UPDATED: Cost scaling benchmark (Result 2) with hardcoded switches for 
estimate_type and error_type at the top.
Includes a dynamic two-panel layout showing the MSE share (Bias² vs SEM²/Std²),
and CRITICAL FIX: The plotted wall-clock time now correctly reflects 1 run 
for Single-Run estimates and N runs for Ensemble estimates.
"""

from __future__ import annotations

import argparse
import numpy as np

from common import add_settings_footer, as_array, load_data

# --- CONFIGURATION ---
# Iso-accuracy target per system. The target is not a claim about absolute
# accuracy -- it fixes a common operating point so SLB and mcsolve are compared
# doing the same job. It is chosen per system so the iso curve carries
# information, and the three values differ by two orders of magnitude, which is
# itself a result:
#
#   System C, 0.005  -- a TIGHTER target is needed, because M*=1 clears 0.02 at
#                       every size and the curve would be a flat, useless line.
#   System B, 0.02   -- discriminating as it stands: M* climbs 4 -> 64.
#   System A, 0.05   -- a LOOSER target is needed. At 0.02 this system misses at
#                       every dimension except 4, because M can never exceed N_L
#                       and even M = N_L parks just above the target (0.024 to
#                       0.029 across dims 8-512). An all-missed curve says only
#                       "impossible". At 0.05 it becomes quantitative and says
#                       something sharper: M* ~ N_L (1/3, 8/13, 31/31, 57/57,
#                       64/73), i.e. essentially every operator is needed. That
#                       IS the no-compression result, measured rather than
#                       asserted.
#
# Quote the target whenever quoting a speedup from these panels; a cost at 0.05
# is not comparable with a cost at 0.005.
TARGET_VAL = 0.02
TARGET_BY_SYSTEM = {"spin_chain": 0.05, "mixed_chain": 0.02,
                    "oscillator_bath": 0.005}

# Which estimate's error defines M*?
#   "ensemble" : error of the N_ACC-run AVERAGE  -> sqrt(bias^2 + SEM^2).
#                The suite-wide convention and what the published figures use;
#                it matches how the method is actually used (run several,
#                average).
#   "single"   : error of ONE run -> sqrt(bias^2 + Std^2). Harsher: no
#                averaging, so noise stays full size and M* comes out larger.
# Either is defensible; the footer always states which one produced the figure.
# COMMITTED DEFAULT: "single". Result 2 costs ONE run and pairs it with that
# run's own error (RMSE^2 = bias^2 + StdDev^2) -- the per-solve question the
# cost curves ask. The unsuffixed headline figures are the single-run view;
# switching to "ensemble" writes _ensemble_rmse-suffixed files so the
# committed headline can never be silently rewritten under a non-default
# setting.
ESTIMATE_TYPE = "single"
ERROR_TYPE = "rmse"       # Options: "rmse", "bias", "sem", "std"

SHOW_NATIVE_CURVE = True  # cost of the native full-dissipator exact solve
                          # (same equation as mesolve, no superoperators):
                          # the leanest exact route, and the accuracy
                          # reference past the mesolve wall
# ---------------------

def get_metrics(e, n_acc):
    """All error components of one swept-M entry, from the run's RECORDED MSE
    budget: mse (observed MSE of the plotted mean) and sem_sq (the statistical
    part), both written by run_cost_scaling.py.

    NOTE the distinction that must not be conflated: `rmse_std` in the data is
    the delete-one JACKKNIFE spread -- the uncertainty OF the RMSE estimate --
    which is NOT the SEM (the noise INSIDE the estimate). Measured against this
    data the two differ by 1.1x-3x, so the SEM is taken from sem_sq, never from
    the jackknife.

        bias^2 = mse - sem_sq        (systematic, from bundling)
        sem    = sqrt(sem_sq)        (noise of the n_acc-run mean)
        std    = sem * sqrt(n_acc)   (spread of ONE run)
    """
    mse, sem_sq = e.get("mse"), e.get("sem_sq")
    if mse is None or sem_sq is None:      # data file predates the budget
        return None
    bias_sq = max(0.0, mse - sem_sq)
    bias = np.sqrt(bias_sq)
    sem = np.sqrt(sem_sq)
    var_single = sem_sq * n_acc
    std_single = np.sqrt(var_single)
    return {
        "ensemble": {"rmse": np.sqrt(bias_sq + sem_sq), "bias": bias,
                     "sem": sem, "std": std_single, "noise_sq": sem_sq},
        "single":   {"rmse": np.sqrt(bias_sq + var_single), "bias": bias,
                     "sem": sem, "std": std_single, "noise_sq": var_single},
    }

def derive_iso(points, target, est_type, err_type, n_acc=16):
    """(m_star, iso_cost, bias_sq, noise_sq) per dimension, plus a parallel list
    of "unreached" markers: dimensions whose sweep contains data but where NO
    swept M meets the target under the selected definition. Those are drawn as
    hollow markers at the largest swept M rather than silently omitted -- a gap
    must never leave the reader guessing between "impossible" and "not
    measured"."""
    out, unreached = [], []
    for p in points:
        row = (np.nan, np.nan, np.nan, np.nan)
        last = None
        for e in p.get("m_sweep") or []:
            metrics = get_metrics(e, n_acc)   # None on diverged/legacy rows
            if metrics is None:
                continue
            last = e
            val = metrics[est_type][err_type]
            if val <= target:
                # COST SEMANTICS: the run stores the wall-clock of the WHOLE
                # N_ACC-run estimate. The ensemble error belongs to that whole
                # estimate, so its cost is the stored value as-is; the
                # single-run error belongs to ONE run, so its cost is the
                # stored value divided by N_ACC. Cost and error must always
                # describe the same object.
                base_cost = e.get("cost")
                cost = (base_cost / n_acc if est_type == "single"
                        else base_cost)
                row = (e.get("M"), cost, metrics[est_type]["bias"] ** 2,
                       metrics[est_type]["noise_sq"])
                break
        else:
            if last is not None:      # sweep had data, target never met
                base_cost = last.get("cost")
                # keep the error split at the largest swept M: it is what
                # explains the miss, and the bottom panel draws it
                lm = get_metrics(last, n_acc)[est_type]
                unreached.append(
                    (p.get("dim"),
                     base_cost / n_acc if est_type == "single" else base_cost,
                     last.get("M"),
                     lm["bias"] ** 2,
                     lm["noise_sq"]))
        out.append(row)
    arrs = [np.array(v, dtype=float) for v in zip(*out)]
    return arrs + [unreached]

def fixed_m_stats(points, m_rep):
    """(rmse, bias, sem) for the fixed M_REP."""
    rmse, bias, sem = [], [], []
    n_acc = 16
    for p in points:
        hit = next((e for e in p.get("m_sweep") or [] if e.get("M") == min(m_rep, p.get("n_l", m_rep))), None)
        if hit is None or hit.get("rmse") is None or hit.get("rmse_std") is None:
            rmse.append(np.nan); bias.append(np.nan); sem.append(np.nan)
        else:
            metrics = get_metrics(hit, n_acc)
            rmse.append(hit.get("rmse"))
            bias.append(metrics["ensemble"]["bias"])
            sem.append(metrics["ensemble"]["sem"])
    return np.array(rmse), np.array(bias), np.array(sem)

# Timings below this are dominated by fixed overhead -- interpreter dispatch,
# object construction, allocator warm-up -- not by the algorithm being measured.
# Including them biases every fitted slope downward, sometimes badly: on the
# spin chain mesolve costs 0.032 s at dim 4 and 0.021 s at dim 8, LOWER at the
# larger size, which cannot be true of a cost that grows with N. Those two
# points alone pull the fit from N^4.9 to N^3.0 and make the extrapolated line
# visibly undershoot the trend it extrapolates from.
FIT_FLOOR_SECONDS = 0.1

# A slope through two points is one ratio, not a scaling law. Below this many
# surviving points the number is reported as a local slope and not as an
# exponent.
FIT_MIN_POINTS = 3


def fit_slope(dims, times, floor=FIT_FLOOR_SECONDS):
    """(exponent, n_points_used) fitted over the curve's meaningful tail.

    Two filters, both necessary:

    * the floor drops points too CHEAP to measure -- mesolve costs 0.032 s at
      dim 4 and 0.021 s at dim 8 on the spin chain, i.e. less at the larger
      size, which is timer noise rather than a cost;
    * the monotone suffix drops points anomalously EXPENSIVE at small size --
      the Davies construction costs 0.4 s at dim 4 against 0.1 s at dim 32,
      first-call warm-up rather than algorithm.

    A cost that grows with N is monotone in N. Where the measured series is
    not, something other than the algorithm dominates, so the fit is taken over
    the longest monotone run ending at the largest dimension.

    Returns (None, 0) when fewer than two points survive.
    """
    m = np.isfinite(times)
    d, t = dims[m], times[m]
    if len(d) < 2:
        return None, 0

    # ORDER MATTERS. The monotone test runs on the full series first, so the
    # fitted range stays CONTIGUOUS in dimension. Applying the floor first
    # punches holes, and the monotone test then cannot see that a warm-up
    # outlier at the smallest dimension is out of order: on the oscillator the
    # Davies series is 0.143, 0.0162, 0.089, 0.454, 1.95, and floor-first kept
    # dims 8, 64, 128 -- skipping 16 and 32 -- for a nonsense N^0.9 against a
    # true tail near N^2.2.
    start = len(t) - 1
    while start > 0 and t[start - 1] <= t[start]:
        start -= 1
    d, t = d[start:], t[start:]

    # then drop leading points too cheap to be meaningful, while >= 2 remain
    while len(d) > 2 and t[0] < floor:
        d, t = d[1:], t[1:]
    if len(d) < 2:
        return None, 0
    return float(np.polyfit(np.log(d), np.log(t), 1)[0]), int(len(d))


def _slope_label(base, dims, times):
    e, n = fit_slope(dims, times)
    if e is None:
        return base
    if n < FIT_MIN_POINTS:
        # honest about what a two-point number is
        return base + rf"  — local slope $N^{{{e:.1f}}}$ ({n} pts)"
    return base + rf"  — $\propto N^{{{e:.1f}}}$ ({n} pts)"

def figure(name, doc, target, est_type, err_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = doc["meta"]
    m_rep, n_acc = meta["params"]["M_REP"], meta["params"]["N_ACC"]
    points = doc["points"]
    dims = as_array([p.get("dim") for p in points])
    n_ls = as_array([p.get("n_l") for p in points])
    t_full = as_array([p.get("t_full") for p in points])
    t_slb = as_array([p.get("t_slb_fixed") for p in points])
    t_dav = as_array([p.get("t_davies") for p in points])
    
    # Scale fixed-M cost based on estimate type
    if est_type == "ensemble":
        t_slb = t_slb * n_acc
        
    mstar, iso_cost, iso_bias_sq, iso_noise_sq, unreached = derive_iso(
        points, target, est_type, err_type, n_acc)

    # Implement the dual-panel layout
    fig, (ax_main, ax_bar) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 9.2), 
        gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.5}
    )

    # ---- TOP PANEL: Cost Scaling ----
    ff = np.isfinite(t_full)
    ax_main.loglog(dims[ff], t_full[ff], "o-", color="tab:red", lw=2, label=_slope_label("full mesolve", dims, t_full))
    
    s_full, _ = fit_slope(dims, t_full)
    if s_full and ff.any() and dims[~ff & np.isfinite(dims)].size:
        d0, y0 = dims[ff][-1], t_full[ff][-1]
        d_ext = np.array(sorted(set(dims[dims >= d0])))
        if len(d_ext) > 1:
            ax_main.loglog(d_ext, y0 * (d_ext / d0) ** s_full, "--", color="tab:red",
                           lw=1.4, alpha=0.45,
                           label=rf"full mesolve, extrapolated $\propto N^{{{s_full:.1f}}}$")
                      
    slb_lbl = "1 run" if est_type == "single" else f"$N_r$={n_acc} runs"
    if SHOW_NATIVE_CURVE:
        # Second EXACT route: the same Lindblad equation propagated by the
        # package's native fixed-step RK4 with ALL N_L operators (no
        # superoperators -> no memory blow-up). Same exponent as mesolve,
        # better prefactor; it survives past the mesolve wall and is the
        # accuracy reference there. Its agreement with mesolve, wherever both
        # ran, is stated in the footer.
        # Plot the native curve ONLY where its accuracy is CERTIFIED: mesolve
        # ran here too (native matched it), or the substep-halving self-check
        # passed. An uncertifiable solve is withheld -- a wall-clock time whose
        # answer you cannot trust is not a result. (Hence the curve stops at
        # dim 32 on the oscillator: dim 64's self-check failed.)
        def _certified(p):
            # A native solve is certified if mesolve also ran here (they
            # matched) or the substep-halving self-check is good. Newer runs
            # record an explicit boolean "passed"; older runs recorded only the
            # numeric "max_abs_dev", so fall back to comparing that against the
            # tolerance directly -- otherwise good references (dev ~1e-9) from
            # an older run get dropped merely for lacking the newer field.
            if p.get("t_native_ref") is None:
                return False
            if str(p.get("reference_method", "")) == "mesolve":
                return True
            sc = p.get("native_ref_selfcheck")
            if not sc:
                return False
            if sc.get("passed") is not None:
                return bool(sc["passed"])
            dev = sc.get("max_abs_dev")
            tol = sc.get("tol", 1e-4)
            return dev is not None and dev <= tol
        t_nat = as_array([p.get("t_native_ref") if _certified(p) else None
                          for p in points])
        nn = np.isfinite(t_nat)
        if nn.any():
            ax_main.loglog(dims[nn], t_nat[nn], "d-.", color="darkred", lw=1.8,
                           alpha=0.85,
                           label=_slope_label("native full dissipator (exact)",
                                              dims, t_nat))
    ax_main.loglog(dims, t_slb, "s-", color="tab:green", lw=2, label=_slope_label(f"fixed M={m_rep} [{slb_lbl}]", dims, t_slb))
    
    ii = np.isfinite(iso_cost)
    # both SLB curves must name the SAME kind of object (1 run vs an
    # N-run estimate), so a reader never compares a 16-run cost against a
    # 1-run cost without noticing.
    iso_label = f"iso-{err_type} (target {target}) [{slb_lbl}]"
    ax_main.loglog(dims[ii], iso_cost[ii], "^--", color="tab:blue", lw=2, label=_slope_label(iso_label, dims, iso_cost))
    
    dd = np.isfinite(t_dav)
    if unreached:
        ud = np.array([u[0] for u in unreached], dtype=float)
        uc = np.array([u[1] for u in unreached], dtype=float)
        ax_main.loglog(ud, uc, "^", mfc="none", mec="tab:blue", ms=9, mew=1.6,
                       ls="none",
                       label="target not reached in sweep (largest swept M)")
        for x, y, u in zip(ud, uc, unreached):
            ax_main.annotate(f"M={int(u[2])}\n(missed)", (x, y),
                             textcoords="offset points", xytext=(6, -14),
                             fontsize=10, color="tab:blue", alpha=0.9)
    ax_main.loglog(dims[dd], t_dav[dd], "x:", color="gray", lw=1.5, alpha=0.8, label=_slope_label("Davies construction", dims, t_dav))

    # Annotate M*
    for x, y, ms in zip(dims[ii], iso_cost[ii], mstar[ii]):
        ax_main.annotate(f"M*={int(ms)}", (x, y), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=11)

    ax_main.set_ylabel("wall-clock time (s)", fontsize=13)
    ax_main.set_xticks(dims)
    ax_main.set_xticklabels([f"{int(d)}\n" + rf"$N_L$={int(l)}" for d, l in zip(dims, n_ls)], fontsize=11)
    ax_main.tick_params(axis="x", which="minor", labelbottom=False)
    ax_main.set_xlabel("Hilbert dimension N  (with Lindblad operator count $N_L$)",
                       fontsize=13)
    
    est_label_cap = "Ensemble" if est_type == "ensemble" else "Single-Run"
    ax_main.set_title(f"{name}: cost scaling with M* annotations "
                      f"({est_label_cap} {err_type.upper()})", fontsize=14)
    ax_main.legend(fontsize=11)
    ax_main.grid(True, which="both", alpha=0.3)

    # ---- BOTTOM PANEL: MSE Share Bar Chart ----
    # Every swept dimension appears. Those that reached the target are drawn at
    # their M*; those that missed are drawn at the largest M they swept, hatched
    # and labelled. Showing only the former left System A with a single bar,
    # which reads as a broken figure rather than as the result it is -- and the
    # missed dimensions carry the more informative split, since a bias-dominated
    # bar is *why* the target was out of reach.
    reached_dims = list(dims[ii])
    v_dims = list(dims[ii]) + [u[0] for u in unreached]
    v_mstar = list(mstar[ii]) + [u[2] for u in unreached]
    v_bias_sq = list(iso_bias_sq[ii]) + [u[3] for u in unreached]
    v_noise_sq = list(iso_noise_sq[ii]) + [u[4] for u in unreached]
    order = np.argsort(np.array(v_dims, dtype=float))
    v_dims = np.array(v_dims, dtype=float)[order]
    v_mstar = np.array(v_mstar, dtype=float)[order]
    v_bias_sq = np.array(v_bias_sq, dtype=float)[order]
    v_noise_sq = np.array(v_noise_sq, dtype=float)[order]
    missed = np.array([d not in reached_dims for d in v_dims])

    if len(v_dims) > 0:
        x_positions = np.arange(len(v_dims))
        width = 0.5
        
        # Calculate shares (0.0 to 1.0)
        totals = v_bias_sq + v_noise_sq
        totals[totals == 0] = 1e-12 # safety fallback
        
        bias_shares = v_bias_sq / totals
        noise_shares = v_noise_sq / totals
        
        # Dynamically set legend based on ensemble vs single
        noise_label = "SEM²" if est_type == "ensemble" else "Std²"
        
        hatches = ["//" if m else "" for m in missed]
        for x, b, n, h in zip(x_positions, bias_shares, noise_shares, hatches):
            ax_bar.bar(x, b, width, color="#1f77b4", edgecolor="black", hatch=h)
            ax_bar.bar(x, n, width, bottom=b, color="#B0C4DE",
                       edgecolor="black", hatch=h)
        # legend proxies, so the hatch meaning is stated once
        ax_bar.bar(0, 0, 0, color="#1f77b4", edgecolor="black", label="Bias²")
        ax_bar.bar(0, 0, 0, color="#B0C4DE", edgecolor="black", label=noise_label)
        if missed.any():
            ax_bar.bar(0, 0, 0, color="white", edgecolor="black", hatch="//",
                       label="target missed (largest swept M)")

        ax_bar.set_xticks(x_positions)
        ax_bar.set_xticklabels(
            [f"{int(d)}\n" + (rf"$M={int(m)}$" if ms else rf"$M^*={int(m)}$")
             for d, m, ms in zip(v_dims, v_mstar, missed)], fontsize=11)
        ax_bar.set_ylabel("MSE share", fontsize=13)
        ax_bar.set_title(rf"MSE budget at $M^*$ (Bias² vs. {noise_label})", fontsize=13)
        ax_bar.set_ylim(0, 1.05)
        # outside the axes: the bars reach y=1 everywhere, so any
        # in-axes placement covers data
        ax_bar.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                      fontsize=10, frameon=False)
    else:
        # Fallback if target was completely unreachable across the board
        ax_bar.text(0.5, 0.5, "No data reached target", ha='center', va='center')
        ax_bar.set_axis_off()

    # Dynamic footer math breakdown
    footer_math = "RMSE\u00b2 = bias\u00b2 + SEM\u00b2" if est_type == "ensemble" else "RMSE\u00b2 = bias\u00b2 + Std\u00b2"

    segs = [
        f"fixed-M: cost of {slb_lbl} (M={m_rep}); iso-accuracy: smallest swept M "
        f"where {est_type} {err_type.upper()}<={target} vs exact",
        f"{footer_math}; {meta['substeps']} RK4 substep(s)/step; Davies construction timed separately",
    ]
    # the two exact routes and their agreement, wherever both ran
    val = doc.get("native_vs_mesolve")
    if SHOW_NATIVE_CURVE and val and val.get("max_devs"):
        worst = max(val["max_devs"])
        segs.append(
            f"two exact routes: qutip mesolve and the native full dissipator "
            f"({val['substeps']} substeps) agree to {worst:.0e} at every "
            f"dimension where both run; past the mesolve wall the native "
            f"route supplies the accuracy reference")
    elif SHOW_NATIVE_CURVE and any(
            str(p.get("reference_method", "")).startswith("native")
            for p in points):
        segs.append("iso-accuracy beyond the mesolve wall uses the native "
                    "full-dissipator reference")
    add_settings_footer(fig, *segs, fontsize=11.5, wrap_chars=140)
    
    # Append suffix to filename if non-default switches are used
    metric_suffix = ("" if (est_type == "single" and err_type == "rmse")
                     else f"_{est_type}_{err_type}")
    out = f"benchmark_cost_scaling_{name}{metric_suffix}.png"
    
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all",
                    choices=["spin_chain", "mixed_chain", "oscillator_bath", "all"])
    args = ap.parse_args()
    
    names = ([  "spin_chain", "mixed_chain", "oscillator_bath"]
             if args.system == "all" else [args.system])
    for name in names:
        doc = load_data(f"cost_scaling_{name}.json")
        target = TARGET_BY_SYSTEM.get(name, TARGET_VAL)
        figure(name, doc, target, ESTIMATE_TYPE, ERROR_TYPE)

if __name__ == "__main__":
    main()