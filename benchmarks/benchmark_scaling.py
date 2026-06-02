"""
benchmark_scaling.py
=====================================

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

Sparsity control:

    The Davies/Lindblad operator construction can now be pruned through
    COUPLING_THRESHOLD.  This is separate from the bundling parameter M.

        COUPLING_THRESHOLD = 0.0      -> keep the full Davies operator set
        COUPLING_THRESHOLD > 0.0      -> drop weak coupling blocks while building
                                        the original Lindblad operator set

    You may set the value directly below, or override it from the shell:

        PowerShell:
            $env:COUPLING_THRESHOLD="1e-6"
            python benchmark_scaling.py

        bash:
            COUPLING_THRESHOLD=1e-6 python benchmark_scaling.py

    The figures and terminal output report both the retained operator count and,
    when pruning is active, the unpruned count:

        N_L = retained/full

Updates in this version:
    * Prints the original/full and retained Lindblad-operator counts explicitly.
    * Adds N_L and any active threshold values directly to plot titles.
    * Uses system-specific M ladders for accuracy plots.
    * Avoids calling full mesolve "exact"; it is labeled as a Lindblad reference.
    * Softens the scaling-plot wall label to "stopped by time/RAM budget".
    * Makes output names threshold-aware so sparse runs do not overwrite the
      threshold-zero figures unless OUTPUT_SUFFIX is explicitly set.

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_scaling.py
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
M = 4                       # number of bundled operators for the timing sweep
N_REALIZATIONS = 8          # stochastic repeats averaged for the bundled mean

# M ladder for the accuracy figure. These are system-specific so the plots do
# not become visually cluttered when M=8 is already essentially on the reference.
ACCURACY_M_BY_SYSTEM = {
    "oscillator_bath": [2, 8],
    "spin_chain": [2, 8, 16, 32, 64],
}
DEFAULT_ACCURACY_M = [2, 8]

FULL_TIME_BUDGET = 60.0     # stop full mesolve once one solve exceeds this
MAX_FULL_DIM = 64           # never attempt full mesolve above this dimension
TLIST = np.linspace(0.0, 5.0, 40)

# ---------------------------------------------------------------------------
# Davies/Lindblad sparsity controls
# ---------------------------------------------------------------------------
# Legacy threshold passed to davies_operators/build_collapse_ops. Keep this at
# zero for the main benchmark unless you intentionally want the old threshold.
DAVIES_THRESHOLD = float(os.environ.get("DAVIES_THRESHOLD", "0.0"))

# New pruning threshold for weak system-bath coupling blocks during the Davies
# operator build. This controls the sparsity/original operator count N_L.
COUPLING_THRESHOLD = float(os.environ.get("COUPLING_THRESHOLD", "0.0"))

# Lamb-shift pruning threshold. None means inherit DAVIES_THRESHOLD, matching the
# implementation's default. This benchmark does not use imag_gamma, but the
# option is exposed for completeness.
_lamb_shift_env = os.environ.get("LAMB_SHIFT_THRESHOLD", "None").strip()
LAMB_SHIFT_THRESHOLD: float | None
if _lamb_shift_env.lower() in {"", "none", "null"}:
    LAMB_SHIFT_THRESHOLD = None
else:
    LAMB_SHIFT_THRESHOLD = float(_lamb_shift_env)

# If True and COUPLING_THRESHOLD > 0, build the unpruned operator set once too,
# only to report the original N_L. This is useful for captions/diagnostics.
REPORT_UNPRUNED_N_L = True

# Optional manual output suffix. If empty, sparse runs get an automatic suffix.
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "")

# Shared detailed-balance ohmic bath.
ALPHA, KT, OMEGA_C = 0.3, 0.5, 8.0


def gamma(omega: float) -> float:
    if abs(omega) < 1e-10:
        return ALPHA * KT
    return ALPHA * omega * math.exp(-abs(omega) / OMEGA_C) / (1.0 - math.exp(-omega / KT))


# ===========================================================================
# SPARSITY / OUTPUT HELPERS
# ===========================================================================
def make_davies_ops(H: qutip.Qobj, X: qutip.Qobj, *, coupling_threshold: float | None = None):
    """Build Davies operators using the benchmark's sparsity settings."""
    if coupling_threshold is None:
        coupling_threshold = COUPLING_THRESHOLD
    return davies_operators(
        H, X, gamma,
        threshold=DAVIES_THRESHOLD,
        coupling_threshold=float(coupling_threshold),
        lamb_shift_threshold=LAMB_SHIFT_THRESHOLD,
    )


def count_unpruned_ops(H: qutip.Qobj, X: qutip.Qobj, retained_count: int) -> int:
    """Return the fully threshold-zero operator count for reporting only."""
    if not REPORT_UNPRUNED_N_L or not sparsity_active():
        return retained_count
    return len(
        davies_operators(
            H, X, gamma,
            threshold=0.0,
            coupling_threshold=0.0,
            lamb_shift_threshold=None,
        )
    )


def n_l_label(retained: int, full: int) -> str:
    """Human-readable operator count label."""
    if retained == full:
        return f"{retained}"
    return f"{retained}/{full}"


def safe_float_label(x: float) -> str:
    """Compact float string safe for filenames."""
    return f"{x:g}".replace("-", "m").replace("+", "").replace(".", "p")


def sparsity_active() -> bool:
    """Whether this run changes the Davies/Lindblad operator construction."""
    return (
        DAVIES_THRESHOLD != 0.0
        or COUPLING_THRESHOLD != 0.0
        or LAMB_SHIFT_THRESHOLD is not None
    )


def sparsity_label() -> str:
    """Short label for plot titles."""
    parts: list[str] = []
    if DAVIES_THRESHOLD != 0.0:
        parts.append(rf"$\tau_w$={DAVIES_THRESHOLD:g}")
    if COUPLING_THRESHOLD != 0.0:
        parts.append(rf"$\tau_c$={COUPLING_THRESHOLD:g}")
    if LAMB_SHIFT_THRESHOLD is not None:
        parts.append(rf"$\tau_{{LS}}$={LAMB_SHIFT_THRESHOLD:g}")
    return "" if not parts else ", " + ", ".join(parts)


def output_filename(kind: str, system_name: str) -> str:
    """Figure filename, with automatic sparsity suffix for nonzero thresholds."""
    suffix = OUTPUT_SUFFIX
    if not suffix:
        parts: list[str] = []
        if DAVIES_THRESHOLD != 0.0:
            parts.append(f"thresh_{safe_float_label(DAVIES_THRESHOLD)}")
        if COUPLING_THRESHOLD != 0.0:
            parts.append(f"cthresh_{safe_float_label(COUPLING_THRESHOLD)}")
        if LAMB_SHIFT_THRESHOLD is not None:
            parts.append(f"lamb_{safe_float_label(LAMB_SHIFT_THRESHOLD)}")
        suffix = "" if not parts else "_" + "_".join(parts)
    return f"benchmark_{kind}_{system_name}{suffix}.png"


def print_sparsity_config() -> None:
    lamb = "None" if LAMB_SHIFT_THRESHOLD is None else f"{LAMB_SHIFT_THRESHOLD:g}"
    print("\nDavies/Lindblad construction settings:")
    print(f"  DAVIES_THRESHOLD       = {DAVIES_THRESHOLD:g}")
    print(f"  COUPLING_THRESHOLD     = {COUPLING_THRESHOLD:g}")
    print(f"  LAMB_SHIFT_THRESHOLD   = {lamb}")
    print(f"  REPORT_UNPRUNED_N_L    = {REPORT_UNPRUNED_N_L}")


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


def require_nonempty_c_ops(name: str, dim: int, c_ops: list[qutip.Qobj]) -> None:
    """Fail clearly if the sparsity threshold removed every dissipator."""
    if len(c_ops) == 0:
        raise ValueError(
            f"[{name}] dim={dim}: COUPLING_THRESHOLD={COUPLING_THRESHOLD:g} "
            "removed all Davies/Lindblad operators. Reduce COUPLING_THRESHOLD."
        )


def run_scaling(name, build, sizes):
    dims, ops, full_ops, t_full, t_bundled = [], [], [], [], []
    full_feasible, wall_dim = True, None

    print(
        f"\n[{name}]  {'dim':>5} {'N_L kept':>9} {'N_L full':>9} "
        f"{'full mesolve':>14} {'bundled':>12}"
    )
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]

        c_ops = make_davies_ops(H, X)
        require_nonempty_c_ops(name, dim, c_ops)
        n_l = len(c_ops)
        n_l_full = count_unpruned_ops(H, X, n_l)

        dims.append(dim)
        ops.append(n_l)
        full_ops.append(n_l_full)

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
        print(f"           {dim:>5} {n_l:>9} {n_l_full:>9} {fstr:>14} {tb:10.2f}s")

    t_full = np.array(t_full)
    if wall_dim is None and any(~np.isfinite(t_full)):
        wall_dim = dims[int(np.argmax(~np.isfinite(t_full)))]

    return (
        np.array(dims), np.array(ops), np.array(full_ops),
        t_full, np.array(t_bundled), wall_dim,
    )


def run_accuracy(name, build, size):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    dim = H.shape[0]

    c_ops = make_davies_ops(H, X)
    require_nonempty_c_ops(name, dim, c_ops)
    n_l = len(c_ops)
    n_l_full = count_unpruned_ops(H, X, n_l)
    tlist = np.linspace(0.0, 5.0, 80)

    print(
        f"[{name}] accuracy plot: dim={dim}, "
        f"retained/full Lindblad operators N_L={n_l_label(n_l, n_l_full)}, "
        f"coupling_threshold={COUPLING_THRESHOLD:g}"
    )

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

    return tlist, reference, curves, dim, n_l, n_l_full


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print_sparsity_config()

    for name, build, sizes, acc_size in SYSTEMS:
        dims, ops, full_ops, t_full, t_bundled, wall_dim = run_scaling(name, build, sizes)
        tlist, reference, curves, acc_dim, acc_n_l, acc_n_l_full = run_accuracy(name, build, acc_size)

        # Scaling figure.
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        fin = np.isfinite(t_full)
        ax.loglog(
            dims[fin], t_full[fin], "o-", color="tab:red", lw=2, ms=7,
            label=r"mesolve reference (retained $N_L$ ops)",
        )
        ax.loglog(
            dims, t_bundled, "s-", color="tab:green", lw=2, ms=7,
            label=fr"bundled ($M$={M} of retained $N_L$, {N_REALIZATIONS}x)  [this work]",
        )
        if wall_dim is not None:
            ax.axvline(wall_dim, color="tab:red", ls="--", lw=1.2, alpha=0.7)
            ax.text(
                wall_dim, ax.get_ylim()[1],
                " full mesolve\n stopped past here\n (time/RAM budget)",
                color="tab:red", va="top", ha="left", fontsize=8,
            )
        for d, n, nf, tb in zip(dims, ops, full_ops, t_bundled):
            label = rf"$N_L$={n}" if n == nf else rf"$N_L$={n}/{nf}"
            ax.annotate(
                label, (d, tb), textcoords="offset points",
                xytext=(0, -15), fontsize=8, ha="center", color="dimgray",
            )
        ax.set_xlabel("Hilbert-space dimension")
        ax.set_ylabel("wall-clock time (s)")
        ax.set_title(f"{name}: cost vs system size{sparsity_label()}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        scaling_file = output_filename("scaling", name)
        fig.savefig(scaling_file, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {scaling_file}")

        # Accuracy figure.
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.plot(tlist, reference, "k-", lw=2.6, label="Lindblad reference (mesolve)", zorder=5)
        palette = ["tab:orange", "tab:blue", "tab:green", "tab:purple", "tab:brown"]
        for (m_eff, (mean, std)), col in zip(sorted(curves.items()), palette):
            ax.plot(tlist, mean, "-", color=col, lw=1.6, label=f"bundled M={m_eff}")
            ax.fill_between(tlist, mean - std, mean + std, color=col, alpha=0.16)
        ax.set_xlabel("time")
        ax.set_ylabel(r"$\langle H\rangle$")
        ax.set_title(
            rf"{name} (dim {acc_dim}, $N_L$={n_l_label(acc_n_l, acc_n_l_full)}"
            rf"{sparsity_label()}): bundled vs mesolve"
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        accuracy_file = output_filename("accuracy", name)
        fig.savefig(accuracy_file, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {accuracy_file}")


if __name__ == "__main__":
    main()
