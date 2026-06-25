"""
benchmark_cost_scaling.py
=========================

Cost scaling against the *exact* solver: how does the per-solve wall-clock grow
with Hilbert-space dimension N for the full Lindblad solver versus SLB?

  * full mesolve -- evolves the full density matrix with all N_L collapse
    operators. Cost grows steeply (~O(N^5) in operation count) and the solve
    becomes intractable past a wall, so it is run only up to that point.
  * SLB          -- evolves the density matrix with M (<< N_L) bundled
    operators. Cost grows like the underlying dense propagation (~O(N^3)), so
    it continues cheaply well past the full-solver wall.

The figure plots one timed solve per method per size and reports the *measured*
log-log slope for each (not an idealized guide line), annotated as being
consistent with the O(N^3)/O(N^5) expectation. This isolates the cost-scaling
advantage against the exact solver.

It deliberately does NOT include mcsolve: mcsolve's per-trajectory cost has a
shallower raw slope, so a raw cost-vs-size axis is the wrong place to compare to
it. The meaningful SLB-vs-mcsolve comparison is accuracy-per-cost -- see the
accuracy-cost frontier (Result 3).

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

M_REP = 8   # representative SLB bundle size (matches the other figures)

# Extended sweeps: SLB is cheap, so push well past the full-solver wall.
SYSTEMS = [
    ("spin_chain",      build_spin_chain,      [2, 3, 4, 5, 6, 7, 8]),     # dims 4..256
    ("oscillator_bath", build_oscillator_bath, [4, 8, 16, 32]),            # dims 8..64 (largest stable at 4 substeps)
]


def run(name, build, sizes):
    dims, t_full, t_slb = [], [], []
    full_feasible, wall_dim = True, None
    print(f"[{name}]")
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        c_ops = davies_operators(H, X, gamma)
        dims.append(dim)

        # one exact mesolve, until the wall
        if full_feasible and dim <= MAX_FULL_DIM:
            try:
                t0 = time.perf_counter()
                qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H])
                tf = time.perf_counter() - t0
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

        # one SLB realization at M=M_REP
        m = min(M_REP, len(c_ops))
        t0 = time.perf_counter()
        mesolve_ensemble(H, rho0, TLIST, c_ops, M=m, e_ops=[H],
                         n_realizations=1, rng=0, backend="native", substeps=SUBSTEPS)
        t_slb.append(time.perf_counter() - t0)

        ff = t_full[-1]
        print(f"  dim={dim:4d}  full={('%.3g s' % ff) if np.isfinite(ff) else '   (wall) '}"
              f"   SLB(M={m})={t_slb[-1]:.3g} s")
    return np.array(dims), np.array(t_full), np.array(t_slb), wall_dim


def fit_slope(dims, times):
    """Measured large-N log-log slope. Small dimensions are dominated by fixed
    overhead (object setup, Python), not the asymptotic operation count, so the
    slope is fit from the upper (large-N) part of the range when enough points
    are available."""
    m = np.isfinite(times)
    d, t = dims[m], times[m]
    if len(d) < 2:
        return None
    if len(d) >= 4:                       # use the asymptotic regime
        sel = d >= d.max() / 8.0
        if sel.sum() >= 3:
            d, t = d[sel], t[sel]
    return float(np.polyfit(np.log(d), np.log(t), 1)[0])


def figure(name, dims, t_full, t_slb, wall_dim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e_full = fit_slope(dims, t_full)
    e_slb = fit_slope(dims, t_slb)
    lab_full = "full mesolve (exact)" + (rf"  — large-$N$ slope $\propto N^{{{e_full:.1f}}}$" if e_full else "")
    lab_slb = f"SLB, M={M_REP}" + (rf"  — large-$N$ slope $\propto N^{{{e_slb:.1f}}}$" if e_slb else "")

    fig, ax = plt.subplots(figsize=(8, 5))
    ff = np.isfinite(t_full)
    ax.loglog(dims[ff], t_full[ff], "o-", color="tab:red", lw=2, ms=7, label=lab_full)
    ax.loglog(dims, t_slb, "s-", color="tab:green", lw=2, ms=7, label=lab_slb)

    if wall_dim is not None:
        ax.axvline(wall_dim, color="tab:red", ls="--", alpha=0.5)
        ax.text(wall_dim, ax.get_ylim()[0] * 1.6, "full mesolve\nimpractical past here",
                color="tab:red", fontsize=8, ha="center", va="bottom")

    ax.set_xlabel(r"Hilbert-space dimension $N$")
    ax.set_ylabel("wall-clock time for one solve (s)")
    ax.set_title(f"{name}: cost scaling — SLB stays cheap where full mesolve cannot")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    add_settings_footer(
        fig,
        f"one exact mesolve vs one SLB realization (M={M_REP}, {SUBSTEPS} RK4 substep(s)/step)",
        "operation-count expectation: O(N^3) for SLB vs O(N^5) for full mesolve",
        "for the accuracy-per-cost comparison with mcsolve, see the frontier (Result 3)",
    )

    fig.savefig(f"benchmark_cost_scaling_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  measured slopes:  full N^{e_full},  SLB N^{e_slb}")


def main():
    for name, build, sizes in SYSTEMS:
        figure(name, *run(name, build, sizes))


if __name__ == "__main__":
    main()
