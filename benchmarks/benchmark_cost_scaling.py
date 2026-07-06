"""
benchmark_cost_scaling.py
=========================

Cost scaling against the *exact* solver, reported two ways so the speed claim is
qualified by accuracy.

  * full mesolve -- evolves the full density matrix with all N_L collapse
    operators. Cost ~O(N^5); becomes intractable past a wall.
  * SLB          -- evolves the density matrix with M (<< N_L) bundled
    operators. Cost of one solve ~O(N^3).

Two cost curves for SLB:

  (1) fixed M = M_REP. One solve is cheap and scales ~O(N^3). BUT at fixed M the
      accuracy degrades as N grows: N_L grows, so a fixed bundle count resolves
      the dissipator less finely and the RMSE climbs with N. A pure fixed-M
      speed plot therefore compares at a moving accuracy target.

  (2) iso-accuracy. At each N we choose the smallest M that brings the
      time-averaged RMSE of <H(t)> to TARGET_RMSE (measured against the exact
      solve), and plot the cost of THAT estimate. This is the honest "cost to
      reach a fixed accuracy" scaling. Because the required M grows with N, its
      slope is steeper than the fixed-M O(N^3) -- but it still beats the exact
      solver. It is only computable up to the reference wall (tuning M needs the
      exact answer).

The figure has two panels sharing the dimension axis:
  (top)    wall-clock per solve vs N: full, SLB fixed-M, SLB iso-accuracy, each
           with its measured log-log slope.
  (bottom) the fixed-M accuracy vs N (RMSE climbing), the TARGET_RMSE line, and
           the RMSE actually achieved by the M needed to hit the target at each N
           (annotated) -- i.e. the mechanism behind curve (2).

mcsolve is not shown here (its raw per-trajectory slope makes a cost-vs-size axis
the wrong comparison); see the accuracy-cost frontier (Result 3).

Produces, per system:  benchmark_cost_scaling_<system>.png

Run:  python benchmark_cost_scaling.py   (this is the slow benchmark)
"""

from __future__ import annotations

import time
import numpy as np
import qutip

from benchmark_scaling import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST, SUBSTEPS,
    FULL_TIME_BUDGET, MAX_FULL_DIM, add_settings_footer,
)
from qutip_bundling import davies_operators, mesolve_ensemble

M_REP = 8               # representative fixed bundle size (matches other figures)
N_ACC = 16              # SLB runs averaged to measure accuracy at each size
TARGET_RMSE = 0.02      # accuracy target for the iso-accuracy cost curve
M_ISO_GRID = [1, 2, 4, 8, 16, 32, 64, 128]   # M values searched to hit the target

SYSTEMS = [
    ("spin_chain",      build_spin_chain,      [2, 3, 4, 5]),     # dims 4..32
    ("oscillator_bath", build_oscillator_bath, [4, 8, 16, 32]),            # dims 8..64
]


def tavg_rmse(samples, n_eff, reference):
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    bias = np.abs(mean - reference)
    sem = std / np.sqrt(n_eff)
    return float(np.mean(np.sqrt(bias ** 2 + sem ** 2)))


def tavg_rmse_with_bootstrap(samples, reference, n_boot=256, seed=12345):
    """Time-averaged RMSE plus a bootstrap standard error.

    The RMSE itself is the value used throughout the benchmarks: at each time it
    combines the bias of the sample mean with the SEM over independent SLB runs,
    then averages over time.  The error bar is the bootstrap spread of that same
    scalar estimator under resampling of the independent runs.
    """
    n_eff = samples.shape[0]
    rmse = tavg_rmse(samples, n_eff, reference)
    if n_eff < 2 or n_boot <= 1:
        return rmse, np.nan

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_eff, size=n_eff)
        boot[b] = tavg_rmse(samples[idx], n_eff, reference)
    return rmse, float(np.std(boot, ddof=1))


def slb_estimate(H, rho0, c_ops, m, n_runs):
    """(time-avg samples matrix, wall-clock for the n_runs estimate)."""
    t0 = time.perf_counter()
    ens = mesolve_ensemble(H, rho0, TLIST, c_ops, M=m, e_ops=[H],
                           n_realizations=n_runs, rng=100, backend="native",
                           substeps=SUBSTEPS)
    return np.real(ens.samples[:, 0, :]), time.perf_counter() - t0


def iso_accuracy_point(H, rho0, c_ops, reference, n_l):
    """Smallest M (from M_ISO_GRID) whose N_ACC-run RMSE <= TARGET_RMSE, and the
    cost/RMSE of that estimate.

    Returns (m_star, iso_cost, iso_rmse, iso_rmse_err, reached).
    """
    last_m, last_cost, last_rmse, last_err = None, np.nan, np.nan, np.nan
    for mc in M_ISO_GRID:
        mc = min(mc, n_l)
        samples, dt = slb_estimate(H, rho0, c_ops, mc, N_ACC)
        r, rerr = tavg_rmse_with_bootstrap(samples, reference, seed=20000 + mc)
        last_m, last_cost, last_rmse, last_err = mc, dt, r, rerr
        if r <= TARGET_RMSE:
            return mc, dt, r, rerr, True
        if mc >= n_l:
            break
    # Target not reached; report the best value tried, but mark it as not reached.
    return last_m, last_cost, last_rmse, last_err, False


def run(name, build, sizes):
    dims, t_full, t_slb = [], [], []
    rmse_fix, rmse_fix_err = [], []
    mstar, iso_cost, iso_rmse, iso_rmse_err = [], [], [], []
    full_feasible, wall_dim = True, None
    print(f"[{name}]  target RMSE = {TARGET_RMSE}")
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        c_ops = davies_operators(H, X, gamma)
        n_l = len(c_ops)
        dims.append(dim)

        reference = None
        if full_feasible and dim <= MAX_FULL_DIM:
            try:
                t0 = time.perf_counter()
                res = qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H])
                tf = time.perf_counter() - t0
                reference = np.real(res.expect[0])
                t_full.append(tf)
                if tf > FULL_TIME_BUDGET:
                    full_feasible, wall_dim = False, dim
            except MemoryError:
                t_full.append(np.nan)
                full_feasible, wall_dim = False, dim
        else:
            t_full.append(np.nan)
            if wall_dim is None:
                wall_dim = dim

        # fixed-M cost: one realization
        m = min(M_REP, n_l)
        t0 = time.perf_counter()
        mesolve_ensemble(H, rho0, TLIST, c_ops, M=m, e_ops=[H],
                         n_realizations=1, rng=0, backend="native", substeps=SUBSTEPS)
        t_slb.append(time.perf_counter() - t0)

        # accuracy + iso-accuracy, where a reference exists
        if reference is not None:
            samp, _ = slb_estimate(H, rho0, c_ops, m, N_ACC)
            rf, rferr = tavg_rmse_with_bootstrap(samp, reference, seed=10000 + dim)
            rmse_fix.append(rf)
            rmse_fix_err.append(rferr)
            ms, ic, ir, irerr, reached = iso_accuracy_point(H, rho0, c_ops, reference, n_l)
            mstar.append(ms if reached else np.nan)
            iso_cost.append(ic if reached else np.nan)
            iso_rmse.append(ir if reached else np.nan)
            iso_rmse_err.append(irerr if reached else np.nan)
            tag = f"M*={ms}" if reached else f"M*>{ms} (target missed)"
        else:
            rmse_fix.append(np.nan)
            rmse_fix_err.append(np.nan)
            mstar.append(np.nan)
            iso_cost.append(np.nan)
            iso_rmse.append(np.nan)
            iso_rmse_err.append(np.nan)
            tag = "(no ref)"

        ff = t_full[-1]
        print(f"  dim={dim:4d}  full={('%.3g s' % ff) if np.isfinite(ff) else '  (wall)'}"
              f"   SLB(M={m})={t_slb[-1]:.3g}s   RMSE_fix={('%.2e' % rmse_fix[-1]) if np.isfinite(rmse_fix[-1]) else ' -- '}"
              f"   iso: {tag}")
    return (np.array(dims), np.array(t_full), np.array(t_slb),
            np.array(rmse_fix), np.array(rmse_fix_err),
            np.array(mstar, dtype=float), np.array(iso_cost),
            np.array(iso_rmse), np.array(iso_rmse_err), wall_dim)


def fit_slope(dims, times):
    m = np.isfinite(times)
    d, t = dims[m], times[m]
    if len(d) < 2:
        return None
    if len(d) >= 4:
        sel = d >= d.max() / 8.0
        if sel.sum() >= 3:
            d, t = d[sel], t[sel]
    return float(np.polyfit(np.log(d), np.log(t), 1)[0])


def _slope_label(base, dims, times):
    e = fit_slope(dims, times)
    return base + (rf"  — $\propto N^{{{e:.1f}}}$" if e else ""), e


def figure(name, dims, t_full, t_slb, rmse_fix, rmse_fix_err,
           mstar, iso_cost, iso_rmse, iso_rmse_err, wall_dim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab_full, _ = _slope_label("full mesolve (exact)", dims, t_full)
    lab_fix, _ = _slope_label(f"SLB, fixed M={M_REP}", dims, t_slb)
    lab_iso, _ = _slope_label(f"SLB, iso-accuracy (RMSE={TARGET_RMSE})", dims, iso_cost)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.25]}
    )

    # ---- top: cost ----
    ff = np.isfinite(t_full)
    ax.loglog(dims[ff], t_full[ff], "o-", color="tab:red", lw=2, ms=7, label=lab_full)
    ax.loglog(dims, t_slb, "s-", color="tab:green", lw=2, ms=7, label=lab_fix)
    ii = np.isfinite(iso_cost)
    ax.loglog(dims[ii], iso_cost[ii], "^--", color="tab:blue", lw=2, ms=8, label=lab_iso)
    if wall_dim is not None:
        ax.axvline(wall_dim, color="tab:red", ls="--", alpha=0.5)
        ax.text(wall_dim, ax.get_ylim()[0] * 1.6, "full mesolve\nimpractical past here",
                color="tab:red", fontsize=8, ha="center", va="bottom")
    ax.set_ylabel("wall-clock time for one estimate (s)")
    ax.set_title(f"{name}: cost scaling — fixed-M is cheapest, iso-accuracy is the honest scaling")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ---- bottom: fixed-M accuracy + target + M*-achieved accuracy ----
    rr = np.isfinite(rmse_fix)
    axr.errorbar(dims[rr], rmse_fix[rr], yerr=rmse_fix_err[rr], fmt="s-",
                 color="tab:green", lw=2, ms=7, capsize=3,
                 label=f"RMSE at fixed M={M_REP}")
    axr.axhline(TARGET_RMSE, color="tab:blue", ls="--", alpha=0.7, label=f"target {TARGET_RMSE}")

    ii = np.isfinite(iso_rmse)
    axr.errorbar(dims[ii], iso_rmse[ii], yerr=iso_rmse_err[ii], fmt="^--",
                 color="tab:blue", lw=2, ms=8, capsize=3,
                 label="RMSE achieved at M*")
    axr.set_xscale("log")
    axr.set_yscale("log")
    for x, y, ms in zip(dims[ii], iso_rmse[ii], mstar[ii]):
        if np.isfinite(ms):
            axr.annotate(f"M*={int(ms)}", (x, y), textcoords="offset points",
                         xytext=(6, -12), fontsize=8, color="tab:blue",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    if wall_dim is not None:
        axr.axvline(wall_dim, color="tab:red", ls="--", alpha=0.5)
    axr.set_ylabel("time-avg RMSE\nagainst exact")
    axr.set_xlabel(r"Hilbert-space dimension $N$")
    axr.legend(loc="lower right", fontsize=8)
    axr.grid(True, which="both", alpha=0.3)
    axr.margins(y=0.25)

    add_settings_footer(
        fig,
        f"bottom-panel bars: bootstrap SE over {N_ACC} SLB runs; iso-accuracy: smallest M with "
        f"{N_ACC}-run RMSE<={TARGET_RMSE} vs exact",
        f"{SUBSTEPS} RK4 substep(s)/step; iso-accuracy computable only up to the reference wall",
        "operation-count expectation: O(N^3) per solve for SLB vs O(N^5) for full mesolve",
    )
    fig.savefig(f"benchmark_cost_scaling_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    for name, build, sizes in SYSTEMS:
        figure(name, *run(name, build, sizes))


if __name__ == "__main__":
    main()
