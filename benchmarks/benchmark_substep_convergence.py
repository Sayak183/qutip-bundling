"""
benchmark_substep_convergence.py
================================

Integration-robustness check for Stochastic Lindblad Bundling (SLB). This
answers the skeptic's question "is SLB only fast because it integrates the
master equation more crudely than the adaptive `mesolve`/`mcsolve` reference?"

The check separates SLB's two error sources by holding the *bundling* fixed
(same seed -> identical bundles) and sweeping only the RK4 substep count:

  * pure integration error -- the full, *unbundled* operators run through the
    same fixed-step RK4 backend. This isolates the integrator: it should fall
    quickly (RK4 is ~O(h^4)) and bottom out at the reference's own tolerance.
  * SLB total error        -- the bundled run at a fixed ``M`` and fixed seed.
    Its error is set by the bundling, so it stays flat as substeps grow.

The gap between the two curves is the point: SLB's accuracy is bundling-limited,
and the integrator contributes orders of magnitude less error. SLB therefore is
not "winning by integrating loosely" -- one could integrate far more crudely
without moving the SLB error at all.

Produces, per system:
    benchmark_substep_convergence_<system>.png

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_substep_convergence.py
"""

from __future__ import annotations

import numpy as np
import qutip

from benchmark_scaling import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST,
    format_slb_settings, add_settings_footer,
)
from qutip_bundling import davies_operators, mesolve_ensemble
from qutip_bundling.native_solver import rk4_mesolve

# ===========================================================================
# CONFIG
# ===========================================================================
M = 8                          # fixed bundle size; SLB error is set by this
N_REALIZATIONS = 32            # fixed; with SEED below the bundles are identical
SEED = 0                       # across every substep value, so only the
SUBSTEPS = [1, 2, 4, 8, 16]    # integrator changes along the sweep
ATOL, RTOL = 1e-8, 1e-6        # reference ODE tolerances (match other benchmarks)

# One representative size per system (kept modest so the exact reference is cheap).
SYSTEMS = [
    ("spin_chain",      build_spin_chain,      4),    # dim 16
    ("oscillator_bath", build_oscillator_bath, 8),    # dim 16
]


def run(name, build, size):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    c_ops = davies_operators(H, X, gamma)
    dim = H.shape[0]
    n_l = len(c_ops)

    # adaptive, tight-tolerance reference (the shared ground truth)
    ref = np.real(
        qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H],
                      options={"atol": ATOL, "rtol": RTOL}).expect[0]
    )

    int_err, slb_err = [], []
    for s in SUBSTEPS:
        # pure integration error: full (unbundled) operators through RK4
        full = np.real(
            rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H], substeps=s).expect[0]
        )
        int_err.append(float(np.max(np.abs(full - ref))))

        # SLB total error: bundled, fixed M and fixed seed -> bundles identical
        # at every substep, so the only thing changing is the integration
        res = mesolve_ensemble(H, rho0, TLIST, c_ops, M=M, e_ops=[H],
                               n_realizations=N_REALIZATIONS, rng=SEED,
                               backend="native", substeps=s)
        slb_err.append(float(np.max(np.abs(np.asarray(res.expect[0]) - ref))))

    print(f"[{name}] dim={dim}, N_L={n_l}")
    print(f"  substeps:        {SUBSTEPS}")
    print(f"  integration err: {[f'{e:.2e}' for e in int_err]}")
    print(f"  SLB total err:   {[f'{e:.2e}' for e in slb_err]}")
    return dim, n_l, int_err, slb_err


def figure(name, dim, n_l, int_err, slb_err):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(SUBSTEPS, slb_err, "s-", color="tab:green", lw=2, ms=8,
              label=f"SLB total error (M={M}, bundles fixed) — bundling-limited")
    ax.loglog(SUBSTEPS, int_err, "o--", color="tab:blue", lw=2, ms=7,
              label="full operators via RK4 — pure integration error")
    ax.set_xlabel("RK4 substeps per output step")
    ax.set_ylabel(r"max-over-time error in $\langle H\rangle$ vs adaptive reference")
    ax.set_title(f"{name} (dim {dim}, $N_L$={n_l}): "
                 "SLB error is set by bundling, not the integrator")
    ax.set_xticks(SUBSTEPS)
    ax.set_xticklabels([str(s) for s in SUBSTEPS])
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    add_settings_footer(
        fig,
        format_slb_settings(M=M, substeps="swept", n_realizations=N_REALIZATIONS),
        f"reference: adaptive mesolve, atol={ATOL:g}/rtol={RTOL:g}",
        "bundles held fixed (one seed) across all substep values",
    )

    fname = f"benchmark_substep_convergence_{name}.png"
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def main():
    for name, build, size in SYSTEMS:
        dim, n_l, int_err, slb_err = run(name, build, size)
        figure(name, dim, n_l, int_err, slb_err)


if __name__ == "__main__":
    main()
