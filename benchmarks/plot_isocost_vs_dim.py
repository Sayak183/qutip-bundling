"""
plot_isocost_vs_dim.py
======================

ANALYSIS/FIGURE HALF of the iso-accuracy cost-vs-dimension benchmark (Result 4).
Reads the data written by run_isocost_vs_dim.py and draws the two-panel figure;
runs in seconds.

Everything target- and level-dependent is DERIVED HERE from the raw saved data:

    * SLB, per averaging level n in N_RUNS_LIST: the first n of the saved
      N_RUNS_MAX runs are subsampled at each swept M; M*(n) is the smallest M
      whose n-run time-averaged RMSE reaches TARGET_RMSE, and its cost is
      n x per-run wall-clock. Fewer runs -> larger statistical floor -> larger
      M* : the levels genuinely re-optimize M, they are not mere shifts.
    * mcsolve: S^2 is averaged from the saved fit rows (rmse^2 x ntraj, valid
      because the trajectory mean is unbiased), then ntraj* = (S/target)^2 and
      cost = ntraj* x per-trajectory time; "impractical" past NTRAJ_EXTRAP_MAX.
    * speedup (bottom panel) = mcsolve cost / SLB cost, one line per level.

So the accuracy target AND the averaging levels can both be changed and the
figure redrawn without re-running the benchmark.

The figure (unchanged filename):  benchmark_isocost_vs_dim_<system>.png
Run:  python plot_isocost_vs_dim.py [--system ...] [--target 0.02]
"""

from __future__ import annotations

import argparse

import numpy as np

from common import add_settings_footer, as_array, load_data, tavg_rmse, SUBSTEPS

TARGET_RMSE = 0.02              # accuracy both methods must reach; analysis-time
N_RUNS_LIST = [4, 8, 16]        # SLB averaging levels shown (each re-optimizes M)
NTRAJ_EXTRAP_MAX = 20000        # beyond this, "reaching the target" is impractical


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
    """(ntraj_star, cost, reachable) from the saved S^2 fit rows."""
    s2 = np.mean([np.mean(np.square(r["rmse_repeats"])) * r["ntraj"]
                  for r in point["mc_fit"]])
    t_per_traj = np.mean([r["per_traj_time"] for r in point["mc_fit"]])
    ntraj_star = float(s2) / (target ** 2)
    reachable = ntraj_star <= NTRAJ_EXTRAP_MAX
    cost = float(t_per_traj) * min(ntraj_star, NTRAJ_EXTRAP_MAX)
    return ntraj_star, cost, reachable


def derive(doc, target):
    meta = doc["meta"]["params"]
    if target < meta["SWEEP_STOP_RMSE"]:
        print(f"  WARNING: target {target} is below the sweep floor "
              f"{meta['SWEEP_STOP_RMSE']}; the saved sweep may stop before "
              f"reaching it - re-run run_isocost_vs_dim.py with a lower "
              f"SWEEP_STOP_RMSE.")
    if min(N_RUNS_LIST) < meta["SWEEP_MIN_RUNS"]:
        print(f"  WARNING: level n={min(N_RUNS_LIST)} is below the sweep's "
              f"guaranteed SWEEP_MIN_RUNS={meta['SWEEP_MIN_RUNS']}; small-n "
              f"levels may not reach the target within the saved sweep.")
    if max(N_RUNS_LIST) > meta["N_RUNS_MAX"]:
        raise ValueError(f"level n={max(N_RUNS_LIST)} exceeds the "
                         f"N_RUNS_MAX={meta['N_RUNS_MAX']} runs saved per M")

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

    for p, i in zip(points, range(len(points))):
        per_n = " ".join(f"N{n}:M*={out['slb'][n]['mstar'][i]}"
                         f"{'' if out['slb'][n]['ok'][i] else '>='}"
                         f"/{out['slb'][n]['cost'][i]:.2f}s" for n in N_RUNS_LIST)
        nt = out["mc_star"][i]
        print(f"  dim={p['dim']:4d}  {per_n} | mcsolve ntraj*~{int(round(nt)):,}"
              f"{'' if out['mc_ok'][i] else ' (impractical)'} "
              f"{out['mc_cost'][i]:.2f}s")
    return out


def figure(name, out, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = out["dims"]
    if len(d) == 0:
        print(f"  [{name}] no points below the wall — nothing to plot")
        return

    greens = {4: "#a1d99b", 8: "#41ab5d", 16: "#006d2c"}   # light -> dark by run count
    n_max = max(N_RUNS_LIST)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # ---- top: iso-accuracy cost vs dimension ----
    ax.loglog(d, out["full_cost"], "o:", color="tab:gray", lw=1.5, ms=5, alpha=0.6,
              label="exact mesolve (reference)")
    for n in N_RUNS_LIST:
        c = greens.get(n, "tab:green")
        ax.loglog(d, out["slb"][n]["cost"], "s-", color=c, lw=2, ms=7,
                  label=f"SLB (N={n} runs, tune M)")
    ax.loglog(d, out["mc_cost"], "o-", color="tab:purple", lw=2, ms=8,
              label="mcsolve (tune ntraj)")
    # annotate M* on the darkest SLB curve only (avoids clutter)
    for x, y, m, ok in zip(d, out["slb"][n_max]["cost"], out["slb"][n_max]["mstar"],
                           out["slb"][n_max]["ok"]):
        ax.annotate(f"M*={m}" + ("" if ok else "≥"), (x, y),
                    textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color=greens[n_max])
    for x, y, nt, ok in zip(d, out["mc_cost"], out["mc_star"], out["mc_ok"]):
        label = (f"ntraj≈{int(round(nt)):,}" if (ok and np.isfinite(nt))
                 else f"ntraj≳{int(NTRAJ_EXTRAP_MAX):,}")
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=8, color="tab:purple")
    ax.set_ylabel("wall-clock cost to reach target (s)")
    ax.set_title(f"{name}: cost to reach RMSE={target} — SLB vs mcsolve, vs dimension")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ---- bottom: speedup (mcsolve / SLB) vs dimension, one line per N level ----
    mc = out["mc_cost"]
    mc_reached = out["mc_ok"]
    for n in N_RUNS_LIST:
        c = greens.get(n, "tab:green")
        sc = out["slb"][n]["cost"]
        sok = out["slb"][n]["ok"]
        valid = sok & (sc > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(valid, mc / sc, np.nan)
        both = valid & mc_reached            # both hit target -> exact speedup
        lower = valid & ~mc_reached          # mcsolve impractical -> lower bound
        axr.plot(d[valid], ratio[valid], "-", color=c, lw=1.5, alpha=0.7,
                 label=f"vs SLB N={n}")
        axr.plot(d[both], ratio[both], "D", color=c, ms=6)
        axr.plot(d[lower], ratio[lower], "^", color=c, ms=9, mfc="white")
    axr.axhline(1.0, color="gray", ls="--", alpha=0.6)
    axr.set_yscale("log")
    axr.set_xscale("log")
    axr.set_ylabel("speedup\n(mcsolve / SLB)")
    axr.set_xlabel(r"Hilbert-space dimension $N$")
    axr.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
               title="◆ reach target   △ mcsolve impractical",
               title_fontsize=8, frameon=False)
    axr.grid(True, which="both", alpha=0.3)
    axr.text(0.02, 0.94,
             "above 1 = SLB cheaper; rising = advantage widens with $N$",
             transform=axr.transAxes, fontsize=8, va="top")

    add_settings_footer(
        fig,
        f"iso-accuracy: smallest M at each of N={N_RUNS_LIST} runs / ntraj reaching "
        f"time-avg RMSE={target} vs exact; {SUBSTEPS} RK4 substep(s)/step, single-thread",
        "target and levels applied at analysis time from the saved sweep; computable "
        "only to the exact-reference wall; '≳' = mcsolve needs an impractical "
        "trajectory count",
        fontsize=10,
    )
    fig.savefig(f"benchmark_isocost_vs_dim_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benchmark_isocost_vs_dim_{name}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--system", default="all",
                    choices=["spin_chain", "oscillator_bath", "all"])
    ap.add_argument("--target", type=float, default=TARGET_RMSE,
                    help=f"iso-accuracy RMSE target (default {TARGET_RMSE})")
    args = ap.parse_args()
    names = (["spin_chain", "oscillator_bath"] if args.system == "all"
             else [args.system])
    for name in names:
        doc = load_data(f"isocost_vs_dim_{name}.json")
        print(f"[{name}]  target RMSE = {args.target}")
        figure(name, derive(doc, args.target), args.target)


if __name__ == "__main__":
    main()
