"""
benchmark_scaling.py
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
Run:           python benchmark_scaling.py
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
ACCURACY_N_REALIZATIONS = 32  # realizations for the accuracy & coherence figures
                              # (more than the timing sweep so the +/-1 std band
                              # is well resolved); lifted from a literal so the
                              # figure caption reads it, not a magic number.
SUBSTEPS = 4                # RK4 substeps per TLIST step for SLB (native backend).
                            # >=2 is required for stability on the stiffer
                            # oscillator at larger sizes (substeps=1 diverges
                            # there); the result is already converged by 2, so 4
                            # carries margin at negligible cost. Stated on the
                            # plot for a fair comparison against mcsolve's ntraj.

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
# SHARED FIGURE-CAPTION HELPERS
# ===========================================================================
# Single source of truth for the settings footer printed under every benchmark
# figure. Each figure builds its caption from the SAME constants that drive its
# run via these helpers, so a caption can never silently disagree with the code
# that produced the figure. Imported by the other benchmark scripts.
def format_slb_settings(*, M, substeps, n_realizations, n_repeats=None,
                        swept=False, jackknife=False):
    m = M if isinstance(M, int) else list(M)
    head = "SLB: sweep M=" if swept else "SLB: M="
    s = f"{head}{m}"
    if jackknife:
        s += " (jackknife-2)"
    s += f", {substeps} RK4 substep(s)/step, {n_realizations} realizations"
    if n_repeats:
        s += f" \u00d7 {n_repeats} repeats"
    return s


def format_mcsolve_settings(*, ntraj, atol=None, rtol=None,
                            single_thread=True, swept=False):
    head = "mcsolve: sweep ntraj=" if swept else "mcsolve: ntraj="
    s = f"{head}{list(ntraj)}"
    if single_thread:
        s += ", single-thread"
    if atol is not None:
        s += f", atol={atol:g}/rtol={rtol:g}"
    return s


def add_settings_footer(fig, *segments, y=-0.02, fontsize=9, wrap_chars=170):
    """Place one uniform settings caption centred below the whole figure.

    Built from the run's own constants by the format_* helpers, so the caption
    cannot disagree with the settings that produced the figure.  A long caption
    (e.g. the frontier figure, which carries both the SLB and mcsolve settings)
    is split across two centred lines so it does not overflow the figure width;
    short captions stay on a single line.  Segments are never broken mid-text.
    """
    sep = "   |   "
    segs = [seg for seg in segments if seg]
    text = sep.join(segs)
    if len(text) <= wrap_chars or len(segs) < 2:
        fig.text(0.5, y, text, ha="center", va="top", fontsize=fontsize,
                 color="dimgray")
        return text
    # balance the segments across two lines (whole segments only)
    split = min(range(1, len(segs)),
                key=lambda i: abs(len(sep.join(segs[:i])) - len(sep.join(segs[i:]))))
    line1, line2 = sep.join(segs[:split]), sep.join(segs[split:])
    fig.text(0.5, y, line1, ha="center", va="top", fontsize=fontsize,
             color="dimgray")
    fig.text(0.5, y - 0.038, line2, ha="center", va="top", fontsize=fontsize,
             color="dimgray")
    return line1 + "\n" + line2


# ===========================================================================
# ERROR METRIC (shared; imported by benchmark_vs_mcsolve.py)
# ===========================================================================
# The mcsolve frontier (Result 3) and the substep integrator check report the
# error at a single mid-relaxation time. A fixed representative time is used
# there rather than max-over-time because, for a head-to-head comparison, the
# maximum of noisy samples is biased upward and would inflate the noisier
# method's apparent error. (The convergence and jackknife figures, which
# characterize one method's own scaling, use max-over-time instead.)
ERR_PLOT_TIME = 2.5                # the point plotted in error-vs-X figures


def plot_time_index(tlist):
    """Index of ERR_PLOT_TIME in tlist (the value shown in error-vs-X plots)."""
    return int(np.argmin(np.abs(np.asarray(tlist) - ERR_PLOT_TIME)))


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


def _populated_coherence_op(H, ref_states):
    """Hermitian coherence operator |a><b| + h.c. for the energy-eigenstate pair
    (a, b) whose coherence is *most populated by the actual dynamics* (largest
    |<a|rho(t)|b>| over the reference trajectory). This guarantees we track a
    coherence the system genuinely develops -- picking by coupling strength can
    land on a pair the dynamics never populates (value ~ machine zero), which is
    uninformative. <H> is essentially diagonal, so this off-diagonal is exactly
    what energy cannot see."""
    Ha = 0.5 * (np.asarray(H.full()) + np.asarray(H.full()).conj().T)
    evals, evecs = np.linalg.eigh(Ha)
    R = evecs.conj().T  # rows are eigenvectors
    best = (0, 1, -1.0)
    for s in ref_states:
        rho_e = R @ np.asarray(s.full()) @ R.conj().T
        ab = np.abs(rho_e)
        np.fill_diagonal(ab, 0.0)
        i, j = np.unravel_index(int(np.argmax(ab)), ab.shape)
        if ab[i, j] > best[2]:
            best = (int(i), int(j), float(ab[i, j]))
    a, b = best[0], best[1]
    P = np.outer(evecs[:, a], evecs[:, b].conj())
    C = qutip.Qobj(P + P.conj().T, dims=H.dims)
    return C, (a, b), best[2]


def run_accuracy(name, build, size):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    dim = H.shape[0]
    c_ops = davies_operators(H, X, gamma)
    n_l = len(c_ops)
    tlist = np.linspace(0.0, 5.0, 80)

    # Reference states once: used both to choose the coherence and to get the
    # exact <H>, <C> curves.
    ref_states = qutip.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[]).states
    C, (ia, ib), peak = _populated_coherence_op(H, ref_states)
    e_ops = [H, C]   # energy (diagonal-dominated) and a genuinely populated coherence

    print(f"[{name}] accuracy plot: dim={dim}, N_L={n_l}; coherence on eigenstate "
          f"pair ({ia},{ib}), peak |rho_ab|={peak:.2e}")

    references = [np.real(qutip.expect(H, ref_states)),
                  np.real(qutip.expect(C, ref_states))]

    # curves[obs_index][M] = (mean, std)
    curves = [{}, {}]
    m_values = capped_unique_m_values(name, n_l)
    print(f"[{name}] accuracy M values shown: {m_values}")

    for m_eff in m_values:
        ens = mesolve_ensemble(
            H, rho0, tlist, c_ops, M=m_eff, e_ops=e_ops,
            n_realizations=ACCURACY_N_REALIZATIONS, rng=0, backend="native",
            substeps=SUBSTEPS,
        )
        for oi in (0, 1):
            mean = np.real(ens.expect[oi])
            std = (
                np.asarray(ens.std[oi], float)
                if getattr(ens, "std", None) is not None
                else np.asarray(ens.sem[oi], float) * math.sqrt(ACCURACY_N_REALIZATIONS)
            )
            curves[oi][m_eff] = (mean, std)

    return tlist, references, curves, dim, n_l, (ia, ib)


# ===========================================================================
# MAIN
# ===========================================================================
def _accuracy_figure(plt, name, tlist, reference, curves, acc_dim, acc_n_l,
                     obs_math, fname_suffix, subtitle):
    """Two panels: observable vs time (top) and residual vs reference (bottom),
    one SLB curve per M. Used for both <H> and the coherence observable."""
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.6, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    ax.plot(tlist, reference, "k-", lw=1.4, alpha=0.7,
            label="full Lindblad reference (mesolve)", zorder=1)
    palette = ["tab:orange", "tab:blue", "tab:green", "tab:purple",
               "tab:red", "tab:brown", "tab:pink", "tab:olive"]
    for (m_eff, (mean, std)), col in zip(sorted(curves.items()), palette):
        ax.plot(tlist, mean, "-", color=col, lw=1.6, label=f"SLB, M={m_eff}",
                zorder=3)
        ax.fill_between(tlist, mean - std, mean + std, color=col, alpha=0.16)
        axr.plot(tlist, mean - reference, "-", color=col, lw=1.4)
    axr.axhline(0.0, color="k", lw=1.0, alpha=0.7)
    axr.set_xlabel("time")
    axr.set_ylabel(rf"${obs_math}_{{\rm SLB}}-{obs_math}_{{\rm ref}}$")
    ax.set_ylabel(rf"${obs_math}$")
    if name == "spin_chain":
        size_str = f"{int(round(math.log2(acc_dim)))} spins, dim {acc_dim}"
    elif name == "oscillator_bath":
        size_str = f"Fock cutoff {acc_dim // 2}, dim {acc_dim}"
    else:
        size_str = f"dim {acc_dim}"
    ax.set_title(rf"{name} ({size_str}, $N_L$={acc_n_l}): {subtitle}")
    ax.legend(frameon=False)
    fig.tight_layout()
    add_settings_footer(
        fig,
        format_slb_settings(M=sorted(curves), substeps=SUBSTEPS,
                            n_realizations=ACCURACY_N_REALIZATIONS),
        "shaded band = \u00b11 std over realizations; "
        "full-Lindblad reference (mesolve)",
    )
    fig.savefig(f"benchmark_{fname_suffix}_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benchmark_{fname_suffix}_{name}.png")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, build, sizes, acc_size in SYSTEMS:
        tlist, references, curves, acc_dim, acc_n_l, (ia, ib) = run_accuracy(name, build, acc_size)

        # Accuracy figures: energy (diagonal-dominated) and the dominant
        # coherence (off-diagonal). Tracking both shows SLB reproduces
        # populations AND coherences, not just the easy energy expectation.
        _accuracy_figure(
            plt, name, tlist, references[0], curves[0], acc_dim, acc_n_l,
            obs_math=r"\langle H\rangle", fname_suffix="accuracy",
            subtitle="SLB vs full Lindblad reference (energy)",
        )
        _accuracy_figure(
            plt, name, tlist, references[1], curves[1], acc_dim, acc_n_l,
            obs_math=rf"\langle C_{{{ia},{ib}}}\rangle", fname_suffix="coherence",
            subtitle=rf"SLB vs reference (coherence on eigenstates {ia},{ib})",
        )


if __name__ == "__main__":
    main()
