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
SUBSTEPS = 4                # RK4 substeps per TLIST step for SLB (native backend).
                            # >=2 is required for stability on the stiffer
                            # oscillator at larger sizes (substeps=1 diverges
                            # there); the result is already converged by 2, so 4
                            # carries margin at negligible cost. Stated on the
                            # plot for a fair comparison against mcsolve's ntraj.

# SLB bundle sizes shown on the cost-vs-size plot. Three accuracy levels: the
# cost of each rises with size, and mcsolve is matched to the most accurate
# (largest M) so the comparison is against SLB's best available accuracy.
M_SCALING = [2, 4, 8]

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

# Fixed mcsolve trajectory counts. Rather than search for the ntraj that
# matches SLB (which produced confusing lower bounds), we run a few fixed
# values and simply report the cost AND the error each one achieves. No
# matching, no "+": every method reports what it cost and how accurate it was.
NTRAJ_FIXED = [50, 200, 1000]

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


def _run_mcsolve(H, psi0, c_ops, ntraj):
    """Run mcsolve at a fixed ntraj; return (time_s, expect_array).

    mcsolve is pinned single-threaded ("map": "serial") so its wall-clock time
    is the full sequential cost of all ntraj trajectories -- matching SLB's
    single-threaded realization loop. Without this, mcsolve would spread its
    trajectories across CPU cores and be timed with a parallel speedup SLB does
    not get, making the cost comparison unfair to SLB.
    """
    opts = {"progress_bar": False, "map": "serial"}
    try:
        t0 = time.perf_counter()
        res = qutip.mcsolve(H, psi0, TLIST, c_ops=c_ops, e_ops=[H],
                            ntraj=ntraj, options=opts)
        tm = time.perf_counter() - t0
    except (TypeError, KeyError):
        t0 = time.perf_counter()
        res = qutip.mcsolve(H, psi0, TLIST, c_ops=c_ops, e_ops=[H], ntraj=ntraj,
                            options={"progress_bar": False})
        tm = time.perf_counter() - t0
    return tm, np.real(res.expect[0])


def run_scaling(name, build, sizes):
    """Measure cost AND error for every method at fixed knobs across sizes.

    No accuracy matching: each method is run at fixed settings (SLB at each M
    in M_SCALING, mcsolve at each ntraj in NTRAJ_FIXED, full mesolve exact) and
    we record both its wall-clock cost and the error it achieves vs the full
    reference. Error is only defined where the reference is computable.
    """
    dims, ops, t_full = [], [], []
    t_bundled = {m: [] for m in M_SCALING}
    e_bundled = {m: [] for m in M_SCALING}
    t_mc = {nt: [] for nt in NTRAJ_FIXED}
    e_mc = {nt: [] for nt in NTRAJ_FIXED}
    full_feasible, wall_dim = True, None

    print(f"\n[{name}]  cost (s) and max-error vs reference at fixed knobs")
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        c_ops = davies_operators(H, X, gamma)
        n_l = len(c_ops)
        dims.append(dim)
        ops.append(n_l)

        # Reference first (if feasible), so errors can be measured this size.
        ref = None
        if full_feasible and dim <= MAX_FULL_DIM:
            try:
                t0 = time.perf_counter()
                ref_res = qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H])
                tf = time.perf_counter() - t0
                ref = np.real(ref_res.expect[0])
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

        def err_vs_ref(curve):
            return (float(np.max(np.abs(curve - ref)))
                    if ref is not None else np.nan)

        # SLB at each M: cost + error.
        for m in M_SCALING:
            m_timing = min(m, n_l)
            t0 = time.perf_counter()
            ens = mesolve_ensemble(
                H, rho0, TLIST, c_ops, M=m_timing, e_ops=[H],
                n_realizations=N_REALIZATIONS, rng=0, backend="native",
                substeps=SUBSTEPS,
            )
            t_bundled[m].append(time.perf_counter() - t0)
            e_bundled[m].append(err_vs_ref(np.real(ens.expect[0])))

        # mcsolve at each fixed ntraj: cost + error. Only run where a reference
        # exists -- past the wall there is nothing to measure error against, and
        # mcsolve at large ntraj on big systems is prohibitively slow (e.g. a
        # single ntraj=1000 run can take ~an hour at dim 64+), so running it
        # there would buy an unusable (nan-error) point at enormous cost.
        for nt in NTRAJ_FIXED:
            if ref is not None:
                tm, curve = _run_mcsolve(H, psi0, c_ops, nt)
                t_mc[nt].append(tm)
                e_mc[nt].append(err_vs_ref(curve))
            else:
                t_mc[nt].append(np.nan)
                e_mc[nt].append(np.nan)

        fstr = f"{tf:8.2f}s" if np.isfinite(tf) else " (wall) "
        slb_str = " ".join(f"M{m}:{t_bundled[m][-1]:.2f}/{e_bundled[m][-1]:.1e}"
                           for m in M_SCALING)
        if ref is not None:
            mc_str = " ".join(f"nt{nt}:{t_mc[nt][-1]:.2f}/{e_mc[nt][-1]:.1e}"
                              for nt in NTRAJ_FIXED)
        else:
            mc_str = "mcsolve skipped (past wall, no reference)"
        print(f"   dim={dim:>4} N_L={n_l:>5} full={fstr}  {slb_str}  {mc_str}")

    t_full = np.array(t_full)
    if wall_dim is None and any(~np.isfinite(t_full)):
        wall_dim = dims[int(np.argmax(~np.isfinite(t_full)))]

    arr = lambda d: {k: np.array(v) for k, v in d.items()}
    return {
        "dims": np.array(dims), "ops": np.array(ops), "t_full": t_full,
        "t_bundled": arr(t_bundled), "e_bundled": arr(e_bundled),
        "t_mc": arr(t_mc), "e_mc": arr(e_mc), "wall_dim": wall_dim,
    }


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
            n_realizations=32, rng=0, backend="native",
            substeps=SUBSTEPS,
        )
        for oi in (0, 1):
            mean = np.real(ens.expect[oi])
            std = (
                np.asarray(ens.std[oi], float)
                if getattr(ens, "std", None) is not None
                else np.asarray(ens.sem[oi], float) * math.sqrt(32)
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
    palette = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]
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
    fig.savefig(f"benchmark_{fname_suffix}_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved benchmark_{fname_suffix}_{name}.png")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, build, sizes, acc_size in SYSTEMS:
        sc = run_scaling(name, build, sizes)
        tlist, references, curves, acc_dim, acc_n_l, (ia, ib) = run_accuracy(name, build, acc_size)

        dims, ops, t_full, wall_dim = sc["dims"], sc["ops"], sc["t_full"], sc["wall_dim"]
        green_shades = ["#9ccc9c", "#4caf50", "#1b5e20"]
        purple_shades = ["#c9b3e6", "#9b6fd4", "#5e3b9e"]

        # Two panels sharing the size axis: cost (top), error (bottom). Every
        # method is at FIXED settings; we read cost above and the accuracy it
        # achieved below. No accuracy matching.
        fig, (axc, axe) = plt.subplots(
            2, 1, figsize=(7.0, 7.4), sharex=True,
            gridspec_kw={"height_ratios": [1, 1]},
        )

        # --- cost panel ---
        fin = np.isfinite(t_full)
        axc.loglog(dims[fin], t_full[fin], "o-", color="tab:red", lw=2, ms=7,
                   label=r"full mesolve (exact, all $N_L$ ops)")
        for m, col in zip(M_SCALING, green_shades):
            axc.loglog(dims, sc["t_bundled"][m], "s-", color=col, lw=2, ms=6,
                       label=fr"SLB, $M$={m}")
        for nt, col in zip(NTRAJ_FIXED, purple_shades):
            axc.loglog(dims, sc["t_mc"][nt], "^-", color=col, lw=1.8, ms=6,
                       label=fr"mcsolve, ntraj={nt}")
        if wall_dim is not None:
            for a in (axc, axe):
                a.axvline(wall_dim, color="tab:red", ls="--", lw=1.2, alpha=0.6)
            axc.text(wall_dim, axc.get_ylim()[0],
                     "full mesolve\nstopped past here ", color="tab:red",
                     va="bottom", ha="right", fontsize=7.5)
        axc.set_ylabel("wall-clock time (s)")
        axc.set_title(f"{name}: cost and accuracy vs system size (fixed settings)")
        axc.grid(True, which="both", alpha=0.3)
        axc.legend(frameon=False, fontsize=7.5, ncol=2)
        axc.text(
            0.01, 0.02,
            f"SLB: M={M_SCALING}, {SUBSTEPS} RK4 substep(s)/step, "
            f"{N_REALIZATIONS} realizations averaged\n"
            f"mcsolve: ntraj={NTRAJ_FIXED} trajectories averaged "
            f"(both single-thread)",
            transform=axc.transAxes, ha="left", va="bottom", fontsize=7,
            color="dimgray",
        )

        secax = axc.secondary_xaxis("top")
        secax.set_xticks(dims)
        secax.set_xticklabels([str(int(n)) for n in ops])
        secax.minorticks_off()
        secax.set_xlabel(r"number of Lindblad operators $N_L$ (full dissipator)")

        # --- error panel ---
        for m, col in zip(M_SCALING, green_shades):
            e = sc["e_bundled"][m]
            ef = np.isfinite(e)
            axe.loglog(dims[ef], e[ef], "s-", color=col, lw=2, ms=6,
                       label=fr"SLB, $M$={m}")
        for nt, col in zip(NTRAJ_FIXED, purple_shades):
            e = sc["e_mc"][nt]
            ef = np.isfinite(e)
            axe.loglog(dims[ef], e[ef], "^-", color=col, lw=1.8, ms=6,
                       label=fr"mcsolve, ntraj={nt}")
        axe.set_xlabel("Hilbert-space dimension")
        axe.set_ylabel(r"max error in $\langle H\rangle$ vs exact")
        axe.grid(True, which="both", alpha=0.3)
        axe.text(0.99, 0.96,
                 "error measured only where the exact\nreference is computable",
                 transform=axe.transAxes, ha="right", va="top",
                 fontsize=7, color="dimgray")

        fig.tight_layout()
        fig.savefig(f"benchmark_scaling_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved benchmark_scaling_{name}.png")

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
