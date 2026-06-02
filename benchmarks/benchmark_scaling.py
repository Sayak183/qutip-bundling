"""
benchmark_scaling_updated.py
============================

Speed/scaling + accuracy benchmark for `qutip-bundling`, run over TWO systems:

    * "spin_chain"        a dissipative transverse-field Ising chain
                          (many Davies/Lindblad operators; bundling-friendly)
    * "oscillator_bath"   an anharmonic oscillator + spin, bath coupled to the
                          oscillator position (close to the StochLind physics;
                          scales by Fock truncation)

For each system it produces:

    benchmark_scaling_<system>.png    wall-clock time vs Hilbert dimension
                                      (log-log), full mesolve vs bundled.
    benchmark_accuracy_<system>.png   <H(t)> versus time, full mesolve reference
                                      vs bundled curves at selected M values.

Updates in this version:
    * Prints the original full number of Lindblad operators N_L explicitly.
    * Adds N_L directly to the accuracy-plot title.
    * Uses only M = 2 and M = 8 in the accuracy plots by default, because larger
      M curves may be visually redundant when they already sit on top of the
      full-mesolve reference.
    * Avoids calling full mesolve "exact"; it is labeled as the full Lindblad
      reference.
    * Softens the scaling-plot wall label to "stopped by time/RAM budget".

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_scaling_updated.py
"""

from __future__ import annotations

import math
import time
import numpy as np
import qutip
from qutip_bundling import davies_operators, mesolve_ensemble

# ===========================================================================
# CONFIG
# ===========================================================================
M = 4                       # number of bundled operators for the timing sweep
N_REALIZATIONS = 8          # stochastic repeats averaged for the bundled mean

# M ladder for the accuracy figure.  These are system-specific so the plots do
# not become visually cluttered when M=8 is already essentially on the reference.
ACCURACY_M_BY_SYSTEM = {
    "oscillator_bath": [2, 8],
    "spin_chain": [2, 8, 16, 32, 64],
}
DEFAULT_ACCURACY_M = [2, 8]

FULL_TIME_BUDGET = 60.0     # stop full mesolve once one solve exceeds this
MAX_FULL_DIM = 64           # never attempt full mesolve above this dimension
TLIST = np.linspace(0.0, 5.0, 40)

# Shared detailed-balance ohmic bath.
ALPHA, KT, OMEGA_C = 0.3, 0.5, 8.0


def gamma(omega: float) -> float:
    if abs(omega) < 1e-10:
        return ALPHA * KT
    return ALPHA * omega * math.exp(-abs(omega) / OMEGA_C) / (1.0 - math.exp(-omega / KT))


# ===========================================================================
# SYSTEM BUILDERS
# ===========================================================================
def build_spin_chain(n_sites: int, J: float = 1.0, h: float = 0.6):
    """Dissipative transverse-field Ising chain."""
    sx, sz, I = qutip.sigmax(), qutip.sigmaz(), qutip.qeye(2)

    def op(o, i):
        return qutip.tensor([o if k == i else I for k in range(n_sites)])

    H = 0
    for i in range(n_sites - 1):
        H += -J * op(sz, i) * op(sz, i + 1)
    for i in range(n_sites):
        H += -h * op(sx, i)

    X = sum(op(sx, i) for i in range(n_sites))
    psi0 = qutip.tensor([qutip.basis(2, 0)] * n_sites)
    return H, X, psi0


def build_oscillator_bath(n_fock: int, omega0=1.0, anh=0.1, spin_gap=1.0, coupling=0.3):
    """Anharmonic oscillator + spin, with bath coupled to oscillator position."""
    a = qutip.destroy(n_fock)
    num = a.dag() * a
    x = (a + a.dag()) / math.sqrt(2.0)

    sz, sx = qutip.sigmaz(), qutip.sigmax()
    Io, Is = qutip.qeye(n_fock), qutip.qeye(2)

    H = (
        omega0 * qutip.tensor(num + 0.5, Is)
        + anh * qutip.tensor(num * num, Is)
        + 0.5 * spin_gap * qutip.tensor(Io, sz)
        + coupling * qutip.tensor(x, sx)
    )
    X = qutip.tensor(x, Is)
    psi0 = qutip.tensor(qutip.basis(n_fock, n_fock - 1), qutip.basis(2, 0))
    return H, X, psi0


#   name              builder               size knob (-> dim)        accuracy size
SYSTEMS = [
    ("spin_chain",      build_spin_chain,      [2, 3, 4, 5, 6, 7], 4),
    ("oscillator_bath", build_oscillator_bath, [4, 8, 16, 32],    8),
]


# ===========================================================================
# BENCHMARK ROUTINES
# ===========================================================================
def capped_unique_m_values(name: str, n_lindblad: int) -> list[int]:
    """Return selected M values, capped by the available number of operators."""
    requested = ACCURACY_M_BY_SYSTEM.get(name, DEFAULT_ACCURACY_M)
    values: list[int] = []
    for m in requested:
        m_eff = min(int(m), n_lindblad)
        if m_eff > 0 and m_eff not in values:
            values.append(m_eff)
    return values


def run_scaling(name, build, sizes):
    dims, ops, t_full, t_bundled = [], [], [], []
    full_feasible, wall_dim = True, None

    print(f"\n[{name}]  {'dim':>5} {'N_L':>6} {'full mesolve':>14} {'bundled':>12}")
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        c_ops = davies_operators(H, X, gamma)
        n_l = len(c_ops)

        dims.append(dim)
        ops.append(n_l)

        m_timing = min(M, n_l)
        t0 = time.perf_counter()
        mesolve_ensemble(
            H, rho0, TLIST, c_ops, M=m_timing, e_ops=[H],
            n_realizations=N_REALIZATIONS, rng=0, backend="native",
        )
        tb = time.perf_counter() - t0
        t_bundled.append(tb)

        if full_feasible and dim <= MAX_FULL_DIM:
            try:
                t0 = time.perf_counter()
                qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H])
                tf = time.perf_counter() - t0
                t_full.append(tf)
                if tf > FULL_TIME_BUDGET:
                    full_feasible = False
                    wall_dim = dim
            except MemoryError:
                t_full.append(np.nan)
                full_feasible = False
                wall_dim = dim
                tf = float("nan")
        else:
            t_full.append(np.nan)
            tf = float("nan")

        fstr = f"{tf:11.2f}s" if np.isfinite(tf) else "   (wall)   "
        print(f"           {dim:>5} {n_l:>6} {fstr:>14} {tb:10.2f}s")

    t_full = np.array(t_full)
    if wall_dim is None and any(~np.isfinite(t_full)):
        wall_dim = dims[int(np.argmax(~np.isfinite(t_full)))]

    return np.array(dims), np.array(ops), t_full, np.array(t_bundled), wall_dim


def run_accuracy(name, build, size):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    dim = H.shape[0]
    c_ops = davies_operators(H, X, gamma)
    n_l = len(c_ops)
    tlist = np.linspace(0.0, 5.0, 80)

    print(f"[{name}] accuracy plot: dim={dim}, original Lindblad operators N_L={n_l}")

    reference = np.real(qutip.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[H]).expect[0])

    curves = {}
    m_values = capped_unique_m_values(name, n_l)
    print(f"[{name}] accuracy M values shown: {m_values}")

    for m_eff in m_values:
        ens = mesolve_ensemble(
            H, rho0, tlist, c_ops, M=m_eff, e_ops=[H],
            n_realizations=32, rng=0, backend="native",
        )
        mean = np.real(ens.expect[0])
        std = (
            np.asarray(ens.std[0], float)
            if getattr(ens, "std", None) is not None
            else np.asarray(ens.sem[0], float) * math.sqrt(32)
        )
        curves[m_eff] = (mean, std)

    return tlist, reference, curves, dim, n_l


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, build, sizes, acc_size in SYSTEMS:
        dims, ops, t_full, t_bundled, wall_dim = run_scaling(name, build, sizes)
        tlist, reference, curves, acc_dim, acc_n_l = run_accuracy(name, build, acc_size)

        # Scaling figure.
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        fin = np.isfinite(t_full)
        ax.loglog(
            dims[fin], t_full[fin], "o-", color="tab:red", lw=2, ms=7,
            label=r"full mesolve (all $N_L$ Lindblad ops)",
        )
        ax.loglog(
            dims, t_bundled, "s-", color="tab:green", lw=2, ms=7,
            label=fr"bundled ($M$={M} of $N_L$, {N_REALIZATIONS}x)  [this work]",
        )
        if wall_dim is not None:
            ax.axvline(wall_dim, color="tab:red", ls="--", lw=1.2, alpha=0.7)
            ax.text(
                wall_dim, ax.get_ylim()[1],
                " full mesolve\n stopped past here\n (time/RAM budget)",
                color="tab:red", va="top", ha="left", fontsize=8,
            )
        for d, n, tb in zip(dims, ops, t_bundled):
            ax.annotate(
                rf"$N_L$={n}", (d, tb), textcoords="offset points",
                xytext=(0, -15), fontsize=8, ha="center", color="dimgray",
            )
        ax.set_xlabel("Hilbert-space dimension")
        ax.set_ylabel("wall-clock time (s)")
        ax.set_title(f"{name}: cost vs system size")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"benchmark_scaling_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved benchmark_scaling_{name}.png")

        # Accuracy figure.
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.plot(tlist, reference, "k-", lw=2.6, label="full Lindblad reference (mesolve)", zorder=5)
        palette = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]
        for (m_eff, (mean, std)), col in zip(sorted(curves.items()), palette):
            ax.plot(tlist, mean, "-", color=col, lw=1.6, label=f"bundled M={m_eff}")
            ax.fill_between(tlist, mean - std, mean + std, color=col, alpha=0.16)
        ax.set_xlabel("time")
        ax.set_ylabel(r"$\langle H\rangle$")
        ax.set_title(
            rf"{name} (dim {acc_dim}, $N_L$={acc_n_l}): bundled vs full Lindblad reference"
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"benchmark_accuracy_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved benchmark_accuracy_{name}.png")


if __name__ == "__main__":
    main()
