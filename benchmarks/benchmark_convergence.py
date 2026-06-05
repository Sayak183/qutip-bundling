"""
benchmark_convergence.py
========================

Convergence-rate check for Stochastic Lindblad Bundling (SLB), at a fixed system
size, as the bundle size ``M`` grows. This answers the skeptic's question "how
do I know M is converged and not just tuned to look good?" by showing the error
falls at the *rate the theory predicts*, which tuning cannot fake.

Two quantities, two predicted rates (both on a log-log axis with guide lines):

  * statistical spread  -- the standard deviation of the bundled <H>(t) over
    realizations. SLB is a Monte Carlo estimator, so this should fall as the
    classic ``M^(-1/2)``.
  * bias               -- the deviation of the bundled *mean* from the exact
    reference. The finite-M bias is higher order and should fall faster,
    ~ ``M^(-1)``.

Seeing the measured slopes land on -1/2 and -1 is direct evidence the estimator
behaves as derived (eq. 8/11 of the paper), independent of any tuned setting.

Produces, per system:
    benchmark_convergence_<system>.png

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_convergence.py
"""

from __future__ import annotations

import numpy as np
import qutip

from benchmark_scaling import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST,
)
from qutip_bundling import davies_operators, mesolve_ensemble

# ===========================================================================
# CONFIG
# ===========================================================================
N_REALIZATIONS = 128         # enough to estimate the spread and mean cleanly
SUBSTEPS = 4                 # matches the other benchmarks (>=2 for stability)
SEED = 0
M_VALUES = [2, 4, 8, 16, 32, 64]

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
    ref = np.real(qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H]).expect[0])

    Ms, stat, bias = [], [], []
    print(f"\n[{name}] dim={dim}, N_L={len(c_ops)}")
    print(f"{'M':>5}  {'stat spread':>12}  {'bias':>12}")
    for M in M_VALUES:
        if M > len(c_ops):
            continue
        ens = mesolve_ensemble(H, rho0, TLIST, c_ops, M=M, e_ops=[H],
                               n_realizations=N_REALIZATIONS, rng=SEED,
                               backend="native", substeps=SUBSTEPS)
        mean = np.real(ens.expect[0])
        std = np.asarray(ens.std[0], float)
        s = float(np.max(std))
        b = float(np.max(np.abs(mean - ref)))
        Ms.append(M); stat.append(s); bias.append(b)
        print(f"{M:>5}  {s:>12.3e}  {b:>12.3e}")
    return np.array(Ms), np.array(stat), np.array(bias), dim, len(c_ops)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, build, size in SYSTEMS:
        Ms, stat, bias, dim, n_l = run(name, build, size)

        s_slope = np.polyfit(np.log(Ms), np.log(stat), 1)[0]
        b_slope = np.polyfit(np.log(Ms), np.log(bias), 1)[0]

        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        ax.loglog(Ms, stat, "o-", color="tab:blue", lw=1.8,
                  label=fr"statistical spread (fit slope {s_slope:.2f})")
        ax.loglog(Ms, bias, "s-", color="tab:green", lw=1.8,
                  label=fr"bias vs exact (fit slope {b_slope:.2f})")

        # Theoretical guide lines, anchored at the first point.
        ax.loglog(Ms, stat[0] * (Ms / Ms[0]) ** -0.5, "--", color="tab:blue",
                  alpha=0.6, label=r"$M^{-1/2}$ (Monte Carlo)")
        ax.loglog(Ms, bias[0] * (Ms / Ms[0]) ** -1.0, "--", color="tab:green",
                  alpha=0.6, label=r"$M^{-1}$ (finite-$M$ bias)")

        ax.set_xlabel("bundle size $M$")
        ax.set_ylabel(r"max-over-time error in $\langle H\rangle$")
        ax.set_title(f"{name} (dim {dim}, $N_L$={n_l}): SLB convergence in $M$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"benchmark_convergence_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved benchmark_convergence_{name}.png")


if __name__ == "__main__":
    main()
