"""
benchmark_isocost_vs_dim.py
===========================

Iso-accuracy cost of SLB vs `mcsolve`, as a function of Hilbert-space dimension.

Result 3 (the frontier) races SLB against `mcsolve` at a *fixed* size. Result 1
scales *cost vs dimension*, but only SLB against the *exact* solver -- `mcsolve`
never appears there. This figure fills the gap: it runs the SLB-vs-`mcsolve`
comparison **as a function of dimension**, so you can see whether SLB's advantage
over `mcsolve` widens, holds, or shrinks as the system grows.

Method
------
At each dimension we fix a target accuracy (TARGET_RMSE, the time-averaged RMSE
of <H(t)> against the exact solve) and ask each method: what is the smallest
setting -- and hence the wall-clock cost -- that reaches it?

    * SLB:     sweep the bundle size M and take the smallest M whose estimate
               hits the target; cost is the wall-clock of that estimate. This is
               shown at several averaging levels N (independent runs). With fewer
               runs the statistical floor S/sqrt(N) is larger, so a larger M is
               needed to still reach the target -- the levels re-optimize M, they
               are not mere shifts. The speedup panel uses the cheapest level.
    * mcsolve: the trajectory average is an *unbiased* estimator, so its error is
               exactly S/sqrt(ntraj) (no bias floor). We sample a few small ntraj
               (averaging repeats to smooth run-to-run noise), estimate S, and
               solve ntraj* = (S/target)^2 -- more reliable and far cheaper than
               brute-forcing a noisy threshold crossing. cost = ntraj* x per-traj
               time (single-threaded, to match SLB).

Both are scored with the *same* time-averaged RMSE against the *same* exact
reference. Like Result 1's iso-accuracy curve, this is computable only up to the
reference wall -- tuning either knob to a target needs the exact answer, which is
exactly what becomes intractable at large N.

Panels (sharing the dimension axis):
    (top)    cost to reach TARGET_RMSE vs N: SLB at each averaging level N, and
             mcsolve (with the exact-solver cost as a faint reference).
    (bottom) speedup = mcsolve_cost / cheapest-SLB_cost vs N. If this rises,
             SLB's advantage widens with system size.

Produces:  benchmark_isocost_vs_dim_<system>.png

Run (this is a slow benchmark):  python benchmark_isocost_vs_dim.py
"""

from __future__ import annotations

import time
import numpy as np
import qutip

from benchmark_vs_mcsolve import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST, MC_OPTIONS,
)
from benchmark_scaling import (
    add_settings_footer, FULL_TIME_BUDGET, MAX_FULL_DIM,
)
from qutip_bundling import davies_operators, mesolve_ensemble

# ===========================================================================
# CONFIG
# ===========================================================================
TARGET_RMSE = 0.02              # accuracy both methods must reach (matches Result 1)
N_RUNS_LIST = [4, 8, 16]        # SLB averaging levels shown (each re-optimizes M)
N_RUNS_MAX = max(N_RUNS_LIST)   # draw this many runs once per M, subsample for the rest
SUBSTEPS = 4                    # RK4 substeps per TLIST interval for SLB
M_GRID = [1, 2, 4, 8, 16, 32, 64, 128]          # SLB knob searched
MC_FIT_GRID = [100, 200, 400]                   # mcsolve ntraj sampled to fit S (RMSE = S/sqrt(ntraj))
MC_REPEATS = 4                                  # repeats per point -> smooth out mcsolve noise
NTRAJ_EXTRAP_MAX = 20000                        # beyond this, "reaching the target" is impractical

SYSTEMS = [
    ("spin_chain",      build_spin_chain,      [2, 3, 4, 5]),   # dims 4..32
    ("oscillator_bath", build_oscillator_bath, [4, 8, 16]),     # dims 8..32
]


def tavg_rmse(samples, n_eff, reference):
    """Time-averaged RMSE = mean_t sqrt(bias(t)^2 + (std/sqrt(n_eff))^2)."""
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    bias = np.abs(mean - reference)
    sem = std / np.sqrt(n_eff)
    return float(np.mean(np.sqrt(bias ** 2 + sem ** 2)))


def slb_isocost(H, rho0, c_ops, reference, n_l):
    """For each averaging level in N_RUNS_LIST, the smallest M reaching TARGET_RMSE
    and the cost of that estimate. The M-sweep is done once at N_RUNS_MAX runs and
    subsampled, so all levels come from a single sweep.
    Returns dict: n_runs -> (m_star, cost_seconds, reached)."""
    per_run, samples, m_values = {}, {}, []
    reached_at = {n: None for n in N_RUNS_LIST}
    for m in M_GRID:
        m = min(m, n_l)
        if m not in per_run:
            t0 = time.perf_counter()
            ens = mesolve_ensemble(H, rho0, TLIST, c_ops, M=m, e_ops=[H],
                                   n_realizations=N_RUNS_MAX, rng=100, backend="native",
                                   substeps=SUBSTEPS)
            per_run[m] = (time.perf_counter() - t0) / N_RUNS_MAX
            samples[m] = np.real(ens.samples[:, 0, :])
            m_values.append(m)
            for n in N_RUNS_LIST:
                if reached_at[n] is None and tavg_rmse(samples[m][:n], n, reference) <= TARGET_RMSE:
                    reached_at[n] = m
        if all(v is not None for v in reached_at.values()) or m >= n_l:
            break

    result = {}
    for n in N_RUNS_LIST:
        if reached_at[n] is not None:
            m_star, reached = reached_at[n], True
        else:
            m_star, reached = m_values[-1], False
        result[n] = (m_star, n * per_run[m_star], reached)
    return result


def _mcsolve_runs(H, psi0, c_ops, ntraj):
    try:
        res = qutip.mcsolve(H, psi0, TLIST, c_ops, e_ops=[H], ntraj=ntraj,
                            options=MC_OPTIONS)
    except (TypeError, KeyError):
        res = qutip.mcsolve(H, psi0, TLIST, c_ops, e_ops=[H], ntraj=ntraj,
                            options={"progress_bar": False, "keep_runs_results": True})
    return np.real(np.array([res.runs_expect[0][k] for k in range(ntraj)]))


def mcsolve_isocost(H, psi0, c_ops, reference):
    """ntraj needed to reach TARGET_RMSE, from its statistical form. mcsolve's
    trajectory average is UNBIASED, so RMSE = S/sqrt(ntraj) (no bias floor); we
    sample a few ntraj (averaging repeats), estimate S, and solve
    ntraj* = (S/target)^2. Returns (ntraj_star, cost_seconds, reachable) where
    reachable is False when ntraj* exceeds a practical budget."""
    s2_est, per_traj = [], []
    for nt in MC_FIT_GRID:
        rs = []
        t0 = time.perf_counter()
        for _ in range(MC_REPEATS):
            rs.append(tavg_rmse(_mcsolve_runs(H, psi0, c_ops, nt), nt, reference))
        dt = time.perf_counter() - t0
        s2_est.append(float(np.mean(np.square(rs))) * nt)     # rmse^2 * ntraj ~ S^2
        per_traj.append(dt / (MC_REPEATS * nt))
    S2 = float(np.mean(s2_est))                               # effective trajectory variance
    t_per_traj = float(np.mean(per_traj))
    ntraj_star = S2 / (TARGET_RMSE ** 2)                      # unbiased: RMSE = S/sqrt(ntraj)
    reachable = ntraj_star <= NTRAJ_EXTRAP_MAX
    cost = t_per_traj * min(ntraj_star, NTRAJ_EXTRAP_MAX)
    return ntraj_star, cost, reachable


def run(name, build, sizes):
    dims, full_cost = [], []
    slb = {n: {"cost": [], "mstar": [], "ok": []} for n in N_RUNS_LIST}
    mc_cost, mc_star, mc_ok = [], [], []
    full_feasible = True
    print(f"[{name}]  target RMSE = {TARGET_RMSE}")
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        c_ops = davies_operators(H, X, gamma)
        n_l = len(c_ops)

        if not (full_feasible and dim <= MAX_FULL_DIM):
            print(f"  dim={dim:4d}  reference beyond wall — stopping (iso-accuracy needs the exact solve)")
            break
        try:
            t0 = time.perf_counter()
            reference = np.real(qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H]).expect[0])
            tf = time.perf_counter() - t0
        except MemoryError:
            print(f"  dim={dim:4d}  reference OOM — stopping")
            break
        if tf > FULL_TIME_BUDGET:
            full_feasible = False

        slb_res = slb_isocost(H, rho0, c_ops, reference, n_l)
        nt, mc, within = mcsolve_isocost(H, psi0, c_ops, reference)

        dims.append(dim); full_cost.append(tf)
        for n in N_RUNS_LIST:
            m_, c_, ok_ = slb_res[n]
            slb[n]["cost"].append(c_); slb[n]["mstar"].append(m_); slb[n]["ok"].append(ok_)
        mc_cost.append(mc); mc_star.append(nt); mc_ok.append(within)

        reached = [slb_res[n][1] for n in N_RUNS_LIST if slb_res[n][2]]
        best = min(reached) if reached else min(slb_res[n][1] for n in N_RUNS_LIST)
        speed = (mc / best) if best > 0 else float("nan")
        nt_txt = f"{int(round(nt)):,}" if np.isfinite(nt) else "inf"
        per_n = " ".join(f"N{n}:M*={slb_res[n][0]}/{slb_res[n][1]:.2f}s" for n in N_RUNS_LIST)
        print(f"  dim={dim:4d}  N_L={n_l:4d}  full={tf:7.2f}s | {per_n} | "
              f"mcsolve ntraj*~{nt_txt}{'' if within else ' (impractical)'} {mc:.2f}s | "
              f"speedup x{speed:.1f}{'' if within else '+'}")
    return {
        "dims": np.array(dims), "full_cost": np.array(full_cost),
        "slb": {n: {k: np.array(v) for k, v in slb[n].items()} for n in N_RUNS_LIST},
        "mc_cost": np.array(mc_cost), "mc_star": mc_star, "mc_ok": np.array(mc_ok),
    }


def figure(name, out):
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
    for x, y, m, ok in zip(d, out["slb"][n_max]["cost"], out["slb"][n_max]["mstar"], out["slb"][n_max]["ok"]):
        ax.annotate(f"M*={m}" + ("" if ok else "≥"), (x, y), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color=greens[n_max])
    for x, y, nt, ok in zip(d, out["mc_cost"], out["mc_star"], out["mc_ok"]):
        label = f"ntraj≈{int(round(nt)):,}" if (ok and np.isfinite(nt)) else f"ntraj≳{int(NTRAJ_EXTRAP_MAX):,}"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=8, color="tab:purple")
    ax.set_ylabel("wall-clock cost to reach target (s)")
    ax.set_title(f"{name}: cost to reach RMSE={TARGET_RMSE} — SLB vs mcsolve, vs dimension")
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
        f"time-avg RMSE={TARGET_RMSE} vs exact; {SUBSTEPS} RK4 substep(s)/step, single-thread",
        "one speedup line per SLB run count; computable only to the exact-reference wall; "
        "'≳' = mcsolve needs an impractical trajectory count",
        fontsize=10,
    )
    fig.savefig(f"benchmark_isocost_vs_dim_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benchmark_isocost_vs_dim_{name}.png")


def main():
    for name, build, sizes in SYSTEMS:
        figure(name, run(name, build, sizes))


if __name__ == "__main__":
    main()
