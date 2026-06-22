"""
benchmark_seed_robustness.py
============================

Seed-robustness check for the headline SLB-vs-mcsolve comparison. Both methods
are stochastic, and the main benchmarks use a fixed seed for reproducibility.
A skeptic will ask: "did you get lucky with that one seed?" This script answers
it by recomputing the accuracy-vs-cost frontier for several independent master
seeds and overlaying them. If the SLB points stay clustered below the mcsolve
points for *every* seed, the conclusion is not a seed artifact.

For each seed it sweeps:
    * SLB:      bundle size M
    * mcsolve:  trajectory count ntraj
and plots wall-clock time vs error in <H> at t=2.5 (lower-left is better),
drawing one faint frontier per seed plus the seed-averaged frontier in bold.

Produces, per system:
    benchmark_seed_robustness_<system>.png

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_seed_robustness.py
"""

from __future__ import annotations

import time
import numpy as np
import qutip

from benchmark_vs_mcsolve import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST, err_at_plot_time,
    MC_OPTIONS,
)
from benchmark_scaling import (
    format_slb_settings, format_mcsolve_settings, add_settings_footer,
)
from qutip_bundling import davies_operators, mesolve_ensemble

# ===========================================================================
# CONFIG
# ===========================================================================
SEEDS = [0, 1, 2, 3]                 # master seeds; the conclusion should hold for all
M_VALUES = [2, 8, 32]                # SLB knob (subset, for speed)
NTRAJ_VALUES = [50, 200, 1000]       # mcsolve knob
N_REALIZATIONS = 12                  # SLB realizations averaged per point
SUBSTEPS = 4

SYSTEMS = [
    ("spin_chain",      build_spin_chain,      4),   # dim 16
    ("oscillator_bath", build_oscillator_bath, 8),   # dim 16
]


def frontier_for_seed(H, rho0, psi0, c_ops, reference, seed):
    """Return (slb_times, slb_errs, mc_times, mc_errs) for one master seed."""
    slb_t, slb_e = [], []
    for M in M_VALUES:
        t0 = time.perf_counter()
        ens = mesolve_ensemble(H, rho0, TLIST, c_ops, M=M, e_ops=[H],
                               n_realizations=N_REALIZATIONS, rng=seed,
                               backend="native", substeps=SUBSTEPS)
        slb_t.append(time.perf_counter() - t0)
        slb_e.append(err_at_plot_time(ens.expect[0], reference))

    mc_t, mc_e = [], []
    for nt in NTRAJ_VALUES:
        t0 = time.perf_counter()
        try:
            mc = qutip.mcsolve(H, psi0, TLIST, c_ops, e_ops=[H], ntraj=nt,
                               options=MC_OPTIONS, seeds=seed)
        except TypeError:
            mc = qutip.mcsolve(H, psi0, TLIST, c_ops, e_ops=[H], ntraj=nt,
                               options=MC_OPTIONS)
        mc_t.append(time.perf_counter() - t0)
        mc_e.append(err_at_plot_time(mc.expect[0], reference))
    return (np.array(slb_t), np.array(slb_e), np.array(mc_t), np.array(mc_e))


def run(name, build, size):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    c_ops = davies_operators(H, X, gamma)
    dim = H.shape[0]
    reference = np.real(qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H]).expect[0])

    print(f"\n[{name}] dim={dim}, N_L={len(c_ops)}  seeds={SEEDS}")
    per_seed = []
    for s in SEEDS:
        res = frontier_for_seed(H, rho0, psi0, c_ops, reference, s)
        per_seed.append(res)
        print(f"  seed {s}: SLB err {res[1][0]:.2e}->{res[1][-1]:.2e} | "
              f"mcsolve err {res[3][0]:.2e}->{res[3][-1]:.2e}")
    return per_seed, dim, len(c_ops)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, build, size in SYSTEMS:
        per_seed, dim, n_l = run(name, build, size)

        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        # faint per-seed frontiers
        for (st, se, mt, me) in per_seed:
            ax.plot(st, se, "-", color="tab:green", alpha=0.30, lw=1.0)
            ax.plot(mt, me, "-", color="tab:purple", alpha=0.30, lw=1.0)
        # seed-averaged frontiers in bold (average error & time at each knob)
        slb_t = np.mean([p[0] for p in per_seed], axis=0)
        slb_e = np.mean([p[1] for p in per_seed], axis=0)
        mc_t = np.mean([p[2] for p in per_seed], axis=0)
        mc_e = np.mean([p[3] for p in per_seed], axis=0)
        ax.plot(slb_t, slb_e, "s-", color="tab:green", lw=2.2, ms=7,
                label=f"SLB (seed-averaged, {len(SEEDS)} seeds)")
        ax.plot(mc_t, mc_e, "o-", color="tab:purple", lw=2.2, ms=7,
                label=f"mcsolve (seed-averaged, {len(SEEDS)} seeds)")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("wall-clock time (s)  (lower is better)")
        ax.set_ylabel(r"error in $\langle H\rangle$ at $t=2.5$  (lower is better)")
        ax.set_title(f"{name} (dim {dim}, $N_L$={n_l}): frontier across {len(SEEDS)} seeds")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        add_settings_footer(
            fig,
            format_slb_settings(M=M_VALUES, substeps=SUBSTEPS,
                                n_realizations=N_REALIZATIONS, swept=True),
            format_mcsolve_settings(ntraj=NTRAJ_VALUES, atol=MC_OPTIONS["atol"],
                                    rtol=MC_OPTIONS["rtol"], swept=True),
            f"seed-averaged over {len(SEEDS)} master seeds; "
            "faint lines = individual seeds",
        )
        fig.savefig(f"benchmark_seed_robustness_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved benchmark_seed_robustness_{name}.png")


if __name__ == "__main__":
    main()
