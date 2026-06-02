"""
benchmark_vs_mcsolve.py
=========================================

Accuracy-vs-cost benchmark for `qutip-bundling` against QuTiP's Monte-Carlo
trajectory solver (`qutip.mcsolve`), run over the same two systems as
benchmark_scaling.py.

Both methods are stochastic and have different accuracy knobs:

    * bundling:       bundle size M
    * qutip.mcsolve:  number of quantum-jump trajectories ntraj

The fair comparison is therefore an accuracy-vs-cost frontier: sweep each
method's knob and plot max-over-time error against wall-clock time. Lower-left
means lower error and lower cost.

This version also supports Davies-operator sparsity controls:

    * DAVIES_THRESHOLD       -> original davies_operators threshold
    * COUPLING_THRESHOLD     -> prunes weak Davies coupling blocks while building c_ops
    * LAMB_SHIFT_THRESHOLD   -> optional independent Lamb-shift threshold

For thresholded runs, the full mesolve reference, bundled solver, and mcsolve
all use the same retained/pruned c_ops. The script prints both the retained
operator count and the unpruned count, e.g. N_L = 42/128.

Environment-variable examples:

    PowerShell:
        $env:COUPLING_THRESHOLD="1e-6"
        python benchmark_vs_mcsolve.py

    Bash/Linux:
        COUPLING_THRESHOLD=1e-6 python benchmark_vs_mcsolve.py

Produces, per system:
    benchmark_frontier_<system>.png
or, for sparse runs:
    benchmark_frontier_<system>_cthresh_1em6.png

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_vs_mcsolve.py
"""

from __future__ import annotations

import math
import os
import time
import numpy as np
import qutip
from qutip_bundling import davies_operators, mesolve_ensemble

# ===========================================================================
# CONFIG
# ===========================================================================
M_VALUES = [1, 2, 4, 8, 16]            # bundling knob
NTRAJ_VALUES = [10, 50, 200, 1000]     # mcsolve knob
N_REALIZATIONS = 16                    # repeats averaged for bundled mean
N_REPEATS = 4                          # independent repeats -> SEM bars
TLIST = np.linspace(0.0, 5.0, 80)

# Davies operator sparsity controls.  Defaults reproduce the full, unpruned
# operator set.  Set from the shell to avoid editing the file for each run.
DAVIES_THRESHOLD = float(os.environ.get("DAVIES_THRESHOLD", "0.0"))
COUPLING_THRESHOLD = float(os.environ.get("COUPLING_THRESHOLD", "0.0"))


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "" or raw.lower() in {"none", "null"}:
        return None
    return float(raw)


LAMB_SHIFT_THRESHOLD = _env_float_or_none("LAMB_SHIFT_THRESHOLD")

ALPHA, KT, OMEGA_C = 0.3, 0.5, 8.0


def gamma(omega: float) -> float:
    if abs(omega) < 1e-10:
        return ALPHA * KT
    return ALPHA * omega * math.exp(-abs(omega) / OMEGA_C) / (1.0 - math.exp(-omega / KT))


# ===========================================================================
# SYSTEM BUILDERS
# ===========================================================================
def build_spin_chain(n_sites: int, J: float = 1.0, h: float = 0.6):
    sx, sz, I = qutip.sigmax(), qutip.sigmaz(), qutip.qeye(2)

    def op(o, i):
        return qutip.tensor([o if k == i else I for k in range(n_sites)])

    H = 0
    for i in range(n_sites - 1):
        H += -J * op(sz, i) * op(sz, i + 1)
    for i in range(n_sites):
        H += -h * op(sx, i)

    X = sum(op(sx, i) for i in range(n_sites))
    return H, X, qutip.tensor([qutip.basis(2, 0)] * n_sites)


def build_oscillator_bath(n_fock: int, omega0=1.0, anh=0.1, spin_gap=1.0, coupling=0.3):
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
    return H, qutip.tensor(x, Is), qutip.tensor(qutip.basis(n_fock, n_fock - 1), qutip.basis(2, 0))


#   name              builder               fixed size, small enough for full mesolve reference
SYSTEMS = [
    ("spin_chain",      build_spin_chain,      4),   # dim 16, expected unpruned N_L=64
    ("oscillator_bath", build_oscillator_bath, 8),   # dim 16, expected unpruned N_L=128
]


# ===========================================================================
# HELPERS
# ===========================================================================
def max_err(curve, reference):
    return float(np.max(np.abs(np.real(curve) - reference)))


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, 0.0
    sem = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return mean, sem


def capped_unique_m_values(n_lindblad: int) -> list[int]:
    values: list[int] = []
    for m in M_VALUES:
        m_eff = min(int(m), n_lindblad)
        if m_eff > 0 and m_eff not in values:
            values.append(m_eff)
    return values


def make_davies_ops(H: qutip.Qobj, X: qutip.Qobj):
    """Build c_ops using the configured sparsity controls."""
    return davies_operators(
        H,
        X,
        gamma,
        threshold=DAVIES_THRESHOLD,
        coupling_threshold=COUPLING_THRESHOLD,
        lamb_shift_threshold=LAMB_SHIFT_THRESHOLD,
    )


def make_unpruned_davies_ops(H: qutip.Qobj, X: qutip.Qobj):
    """Build the baseline full c_ops for counting only."""
    return davies_operators(
        H,
        X,
        gamma,
        threshold=0.0,
        coupling_threshold=0.0,
        lamb_shift_threshold=None,
    )


def sparsity_active() -> bool:
    return (
        DAVIES_THRESHOLD != 0.0
        or COUPLING_THRESHOLD != 0.0
        or LAMB_SHIFT_THRESHOLD is not None
    )


def nl_label(n_l: int, n_l_full: int) -> str:
    if n_l == n_l_full and not sparsity_active():
        return str(n_l)
    return f"{n_l}/{n_l_full}"


def format_float_for_filename(x: float | None) -> str:
    if x is None:
        return "none"
    return f"{x:g}".replace("+", "").replace("-", "m").replace(".", "p")


def output_suffix() -> str:
    parts: list[str] = []
    if DAVIES_THRESHOLD != 0.0:
        parts.append(f"thresh_{format_float_for_filename(DAVIES_THRESHOLD)}")
    if COUPLING_THRESHOLD != 0.0:
        parts.append(f"cthresh_{format_float_for_filename(COUPLING_THRESHOLD)}")
    if LAMB_SHIFT_THRESHOLD is not None:
        parts.append(f"lamb_{format_float_for_filename(LAMB_SHIFT_THRESHOLD)}")
    return "" if not parts else "_" + "_".join(parts)


# ===========================================================================
# FRONTIER BENCHMARK
# ===========================================================================
def frontier(name, build, size):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)

    c_ops_full = make_unpruned_davies_ops(H, X)
    n_l_full = len(c_ops_full)

    if sparsity_active():
        c_ops = make_davies_ops(H, X)
    else:
        c_ops = c_ops_full
    n_l = len(c_ops)

    if n_l == 0:
        raise ValueError(
            f"[{name}] No Lindblad operators survived the configured thresholds. "
            "Lower DAVIES_THRESHOLD/COUPLING_THRESHOLD."
        )

    label = nl_label(n_l, n_l_full)
    print(f"\n[{name}] dim={H.shape[0]}, Lindblad operators N_L={label}")
    print(
        f"    DAVIES_THRESHOLD={DAVIES_THRESHOLD:g}, "
        f"COUPLING_THRESHOLD={COUPLING_THRESHOLD:g}, "
        f"LAMB_SHIFT_THRESHOLD={LAMB_SHIFT_THRESHOLD}"
    )
    if n_l != n_l_full:
        print(f"    retained {n_l} of {n_l_full} unpruned Davies/Lindblad operators")

    # Reference is the deterministic Lindblad solution for the same retained
    # c_ops.  Thus, for thresholded runs, the frontier compares bundling and
    # mcsolve on the pruned Lindblad equation.
    reference = np.real(qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H]).expect[0])

    m_values = capped_unique_m_values(n_l)

    b_time_mean, b_time_sem, b_err_mean, b_err_sem = [], [], [], []
    print("  bundling (sweep M):")
    for m_eff in m_values:
        times, errors = [], []
        for r in range(N_REPEATS):
            t0 = time.perf_counter()
            ens = mesolve_ensemble(
                H, rho0, TLIST, c_ops, M=m_eff, e_ops=[H],
                n_realizations=N_REALIZATIONS, rng=1000 + r, backend="native",
            )
            times.append(time.perf_counter() - t0)
            errors.append(max_err(ens.expect[0], reference))

        tm, ts = mean_sem(times)
        em, es = mean_sem(errors)
        b_time_mean.append(tm)
        b_time_sem.append(ts)
        b_err_mean.append(em)
        b_err_sem.append(es)
        print(f"    M={m_eff:3d}  time={tm:7.3f} +/- {ts:.3f}s  err={em:.3e} +/- {es:.1e}")

    m_time_mean, m_time_sem, m_err_mean, m_err_sem = [], [], [], []
    print("  mcsolve (sweep ntraj):")
    for nt in NTRAJ_VALUES:
        times, errors = [], []
        for r in range(N_REPEATS):
            t0 = time.perf_counter()
            mc = qutip.mcsolve(
                H,
                psi0,
                TLIST,
                c_ops,
                e_ops=[H],
                ntraj=nt,
                seeds=2000 + r,
                options={"progress_bar": False},
            )
            times.append(time.perf_counter() - t0)
            errors.append(max_err(mc.expect[0], reference))

        tm, ts = mean_sem(times)
        em, es = mean_sem(errors)
        m_time_mean.append(tm)
        m_time_sem.append(ts)
        m_err_mean.append(em)
        m_err_sem.append(es)
        print(f"    ntraj={nt:5d}  time={tm:7.3f} +/- {ts:.3f}s  err={em:.3e} +/- {es:.1e}")

    return {
        "dim": H.shape[0],
        "n_l": n_l,
        "n_l_full": n_l_full,
        "n_l_label": label,
        "m_values": m_values,
        "b_time_mean": np.asarray(b_time_mean),
        "b_time_sem": np.asarray(b_time_sem),
        "b_err_mean": np.asarray(b_err_mean),
        "b_err_sem": np.asarray(b_err_sem),
        "ntraj_values": list(NTRAJ_VALUES),
        "m_time_mean": np.asarray(m_time_mean),
        "m_time_sem": np.asarray(m_time_sem),
        "m_err_mean": np.asarray(m_err_mean),
        "m_err_sem": np.asarray(m_err_sem),
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = output_suffix()
    for name, build, size in SYSTEMS:
        out = frontier(name, build, size)

        fig, ax = plt.subplots(figsize=(6.8, 5.0))

        ax.errorbar(
            out["b_time_mean"], out["b_err_mean"],
            xerr=out["b_time_sem"], yerr=out["b_err_sem"],
            fmt="s-", color="tab:green", lw=2, ms=8, capsize=3,
            label="bundling (this work)",
        )
        for x, y, m_eff in zip(out["b_time_mean"], out["b_err_mean"], out["m_values"]):
            ax.annotate(
                f"M={m_eff}", (x, y), textcoords="offset points", xytext=(6, 4),
                fontsize=8, color="tab:green",
            )

        ax.errorbar(
            out["m_time_mean"], out["m_err_mean"],
            xerr=out["m_time_sem"], yerr=out["m_err_sem"],
            fmt="o-", color="tab:purple", lw=2, ms=8, capsize=3,
            label="qutip.mcsolve",
        )
        for x, y, nt in zip(out["m_time_mean"], out["m_err_mean"], out["ntraj_values"]):
            ax.annotate(
                f"{nt}", (x, y), textcoords="offset points", xytext=(6, -10),
                fontsize=8, color="tab:purple",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock time (s)  ->  cheaper to the left")
        ax.set_ylabel("max-over-time error vs full mesolve reference  ->  better at the bottom")
        ax.set_title(
            rf"{name} (dim {out['dim']}, $N_L$={out['n_l_label']}): accuracy-vs-cost frontier"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        filename = f"benchmark_frontier_{name}{suffix}.png"
        fig.savefig(filename, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {filename}")


if __name__ == "__main__":
    main()
