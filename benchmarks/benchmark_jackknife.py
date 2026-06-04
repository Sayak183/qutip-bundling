"""
benchmark_jackknife.py
======================

Jackknife bias study for `qutip-bundling`, run over the same TWO systems as
`benchmark_scaling.py`:

    * "spin_chain"        a dissipative transverse-field Ising chain
                          (size = number of spins; Hilbert dim = 2**n_spins)
    * "oscillator_bath"   an anharmonic oscillator + spin
                          (size = Fock truncation; Hilbert dim = n_fock * 2)

At a *fixed* bundle size M, finite-M Stochastic Lindblad Bundling (SLB) carries
an O(1/M) bias because the dissipator noise enters the density matrix
nonlinearly. The jackknife-2 estimator (eq. 16, implemented in
`qutip_bundling.mesolve_jackknife`) cancels the leading bias by combining the
full bundle with its two halves. This script asks a single question:

    Holding M fixed, how does the SLB bias grow with system size for each
    model, and how much of it does the jackknife remove?

For each system it produces:

    benchmark_jackknife_<system>.png   bias in <H(t)> vs Hilbert dimension,
                                       uncorrected SLB vs jackknife-corrected.

The bias is measured against the full Lindblad reference (`qutip.mesolve`), so
sizes are capped where that reference is still affordable. The jackknife is a
property of the bundling estimator only; it does not apply to the full solve
(exact) or to the trajectory solver `mcsolve` (which is unbiased in M and
carries a different, trajectory-count error).

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_jackknife.py
"""

from __future__ import annotations

import math
import numpy as np
import qutip
from qutip_bundling import davies_operators, mesolve_jackknife

# Reuse the exact bath, builders, and time grid from the scaling benchmark so
# the two scripts describe the same physics.
from benchmark_scaling import (
    gamma,
    build_spin_chain,
    build_oscillator_bath,
    TLIST,
)

# ===========================================================================
# CONFIG
# ===========================================================================
M = 4                        # fixed bundle size for the whole study (must be even)
N_REALIZATIONS = 32          # enough that the SEM on the mean is well below the
                             # bias we are trying to resolve (bias does not shrink
                             # with realizations; the noise on its estimate does)
SEED = 12345

# Sizes capped so the full-mesolve reference is affordable. The reference is a
# full Lindblad solve with all N_L operators, so the largest size is limited by
# how long *that* solve takes on your machine (roughly Hilbert dim <= 64; at the
# top end a single reference solve can take ~a minute, as in benchmark_scaling).
#   spin_chain:      knob is the number of spins   -> dim = 2 ** n_spins
#   oscillator_bath: knob is the Fock truncation   -> dim = n_fock * 2
# Override with env vars for a quick smoke test, e.g.
#   JK_SPIN_SIZES=2,3,4  JK_OSC_SIZES=4,8  python benchmark_jackknife.py
import os


def _sizes(env_name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env_name)
    return [int(x) for x in raw.split(",")] if raw else default


SYSTEMS = [
    ("spin_chain",      build_spin_chain,      _sizes("JK_SPIN_SIZES", [2, 3, 4, 5])),
    # Oscillator: Fock truncations chosen to fill the mid-range and give a
    # denser curve where the bias is actually resolvable. n_fock -> dim is
    # n_fock*2, so 4,6,8,12,16 -> dims 8,12,16,24,32. Stops at dim 32; the
    # full reference solve at dim 64 (n_fock=32) builds a dense D**4
    # Liouvillian and can exhaust memory. Push higher only with plenty of RAM.
    ("oscillator_bath", build_oscillator_bath, _sizes("JK_OSC_SIZES", [4, 6, 8, 12, 16])),
]


# ===========================================================================
# CORE
# ===========================================================================
def bias_at_size(name, build, size, rng):
    """Return (dim, n_L, bias_direct, bias_jack, sem_direct, sem_jack).

    bias_* is the maximum absolute deviation of <H(t)> from the full Lindblad
    reference over the time grid; sem_* is the standard error of the mean at
    that worst-deviation time, so a bias below its sem is noise-limited.
    """
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    dim = H.shape[0]

    c_ops = davies_operators(H, X, gamma)
    n_L = len(c_ops)

    reference = np.real(
        qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H]).expect[0]
    )

    res = mesolve_jackknife(
        H, rho0, TLIST, c_ops, M=M, e_ops=[H],
        n_realizations=N_REALIZATIONS, rng=rng,
    )
    corrected = np.real(res.expect[0])            # jackknife-corrected SLB
    direct = np.real(res.extra["direct"][0])      # uncorrected SLB

    dev_direct = np.abs(direct - reference)
    dev_jack = np.abs(corrected - reference)
    i_d = int(np.argmax(dev_direct))
    i_j = int(np.argmax(dev_jack))
    # SEM of the mean at the worst-deviation time: a point whose bias is below
    # this is statistically consistent with zero, i.e. noise-limited not bias.
    sem_direct = float(np.real(res.extra["direct_samples"])[:, 0, i_d].std(ddof=1)
                       / math.sqrt(N_REALIZATIONS))
    sem_jack = float(np.real(res.samples)[:, 0, i_j].std(ddof=1)
                     / math.sqrt(N_REALIZATIONS))
    return (dim, n_L,
            float(dev_direct[i_d]), float(dev_jack[i_j]),
            sem_direct, sem_jack)


def run_system(name, build, sizes, rng):
    dims, n_Ls, bd, bj, sd, sj = [], [], [], [], [], []
    print(f"\n=== {name} (M={M}, n_realizations={N_REALIZATIONS}) ===")
    print(f"{'dim':>5}  {'N_L':>6}  {'bias direct':>12}  {'bias jackknife':>14}  {'ratio':>6}")
    for s in sizes:
        dim, n_L, b_direct, b_jack, sem_d, sem_j = bias_at_size(name, build, s, rng)
        dims.append(dim); n_Ls.append(n_L)
        bd.append(b_direct); bj.append(b_jack); sd.append(sem_d); sj.append(sem_j)
        ratio = b_direct / b_jack if b_jack > 0 else float("inf")
        print(f"{dim:>5}  {n_L:>6}  {b_direct:>12.3e}  {b_jack:>14.3e}  {ratio:>6.1f}")
    return (np.array(dims), np.array(n_Ls),
            np.array(bd), np.array(bj), np.array(sd), np.array(sj))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    for name, build, sizes in SYSTEMS:
        dims, n_Ls, bd, bj, sd, sj = run_system(name, build, sizes, rng)

        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.errorbar(dims, bd, yerr=sd, fmt="o-", color="#c1442e", lw=1.8,
                    capsize=3, label="SLB, uncorrected")
        ax.errorbar(dims, bj, yerr=sj, fmt="s-", color="#2e7d32", lw=1.8,
                    capsize=3, label="SLB, jackknife-corrected")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Hilbert-space dimension")
        ax.set_ylabel(r"max$_t\,|\langle H\rangle_{\rm SLB}-\langle H\rangle_{\rm ref}|$")
        ax.set_title(f"{name}: SLB bias vs system size (M={M})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

        # Secondary x-axis: number of Lindblad operators N_L at each dimension.
        secax = ax.secondary_xaxis("top")
        secax.set_xticks(dims)
        secax.set_xticklabels([str(int(n)) for n in n_Ls])
        secax.minorticks_off()
        secax.set_xlabel(r"number of Lindblad operators $N_L$")

        fig.savefig(f"benchmark_jackknife_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote benchmark_jackknife_{name}.png")


if __name__ == "__main__":
    main()
