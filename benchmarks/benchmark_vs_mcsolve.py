"""
benchmark_vs_mcsolve.py
===============================

Accuracy-vs-cost benchmark for `qutip-bundling` against QuTiP's Monte-Carlo
trajectory solver (`qutip.mcsolve`), over the same two systems as
benchmark_scaling.py.

Both methods are stochastic with different accuracy knobs:

    * bundling:       bundle size M
    * qutip.mcsolve:  number of quantum-jump trajectories ntraj

Error metric (see BENCHMARKS.md section 3.1)
--------------------------------------------
A stochastic estimate has two error parts, and one number that hides either is
misleading. At every output time t we form, from the estimate's own samples,

    BIAS(t)  = | mean(t) - reference(t) |          (systematic error)
    SEM(t)   = std(t) / sqrt(N)                     (statistical error, S/sqrt(N))
    RMSE(t)  = sqrt( BIAS(t)**2 + SEM(t)**2 )       (total error)

and report the **time-averaged RMSE**, with the time-averaged SEM as the error
bar. Both methods are treated identically: the estimate is the average of N
samples, and the error bar is that estimate's own sample spread S/sqrt(N) --
for SLB the N samples are independent bundled runs; for mcsolve they are the
ntraj trajectories. There are no extra "repeats" of one method but not the
other, and no bootstrap: a single estimate per point, so the comparison is
symmetric and the big systems stay affordable (one estimate, not R x).

The fair comparison is the accuracy-vs-cost frontier: lower-left is better.

Substeps guard
--------------
SLB integrates with fixed RK4 (SUBSTEPS substeps per output interval). Before
the frontier, the script verifies the SLB error floor is flat in substeps --
i.e. it is the genuine O(1/M) bundling bias, not an unconverged timestep. If
doubling the substeps changes the bias by more than SUBSTEPS_TOL, it prints a
warning so you raise SUBSTEPS (important for larger / stiffer systems).

Produces, per system:
    benchmark_frontier_<system>.png

Requirements:  pip install qutip-bundling matplotlib
Run:           python benchmark_vs_mcsolve.py
"""

from __future__ import annotations

import math
import time
import numpy as np
import qutip
from qutip_bundling import davies_operators, mesolve_ensemble
from benchmark_scaling import add_settings_footer

# ===========================================================================
# CONFIG
# ===========================================================================
M_VALUES = [1, 2, 4, 8, 16, 32]        # bundling knob
NTRAJ_VALUES = [10, 50, 200, 1000]     # mcsolve knob
N_RUNS_SWEEP = [16, 32, 64]            # independent SLB runs averaged per estimate
N_RUNS_MAX = max(N_RUNS_SWEEP)         # draw this many once, subsample for the sweep
SUBSTEPS = 4                           # RK4 substeps per TLIST interval for SLB
SUBSTEPS_TOL = 0.05                    # warn if doubling substeps moves the bias > 5%
SUBSTEPS_PROBE_M = 16                  # M used for the substeps convergence guard

# Fairness controls for mcsolve:
#  - single-threaded ("map": "serial") so wall-clock matches SLB's serial loop.
#  - keep per-trajectory results so the trajectory spread (-> S/sqrt(ntraj)) is
#    available for the SEM error bar, the same quantity SLB gets from its runs.
MC_ATOL = 1e-8
MC_RTOL = 1e-6
MC_OPTIONS = {"progress_bar": False, "map": "serial",
              "atol": MC_ATOL, "rtol": MC_RTOL, "keep_runs_results": True}
TLIST = np.linspace(0.0, 5.0, 80)

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
    ("spin_chain",      build_spin_chain,      4),   # dim 16
    ("oscillator_bath", build_oscillator_bath, 8),   # dim 16
]


# ===========================================================================
# ERROR METRIC: time-averaged BIAS / SEM / RMSE from an estimate's samples
# ===========================================================================
def bias_sem_rmse(samples: np.ndarray, n_eff: int, reference: np.ndarray):
    """Return (time-avg BIAS, time-avg SEM, time-avg RMSE).

    samples: (n_samples, n_times). The estimate is the sample mean; SEM uses
    n_eff = number of averaged samples (S / sqrt(n_eff)).
    """
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    bias = np.abs(mean - reference)
    sem = std / math.sqrt(n_eff)
    rmse = np.sqrt(bias ** 2 + sem ** 2)
    return float(np.mean(bias)), float(np.mean(sem)), float(np.mean(rmse))


def capped_unique_m_values(n_lindblad: int) -> list[int]:
    values: list[int] = []
    for m in M_VALUES:
        m_eff = min(int(m), n_lindblad)
        if m_eff > 0 and m_eff not in values:
            values.append(m_eff)
    return values


def slb_samples(H, rho0, c_ops, m_eff, n_runs, substeps):
    """Return (samples (n_runs, n_times), per_run_seconds) for one M."""
    t0 = time.perf_counter()
    ens = mesolve_ensemble(H, rho0, TLIST, c_ops, M=m_eff, e_ops=[H],
                           n_realizations=n_runs, rng=1000, backend="native",
                           substeps=substeps)
    per_run = (time.perf_counter() - t0) / n_runs
    return np.real(ens.samples[:, 0, :]), per_run


def substeps_guard(H, rho0, c_ops, reference, m_eff):
    """Check the SLB bias is flat under doubling SUBSTEPS (integrator converged)."""
    s_lo, s_hi = SUBSTEPS, 2 * SUBSTEPS
    samp_lo, _ = slb_samples(H, rho0, c_ops, m_eff, N_RUNS_MAX, s_lo)
    samp_hi, _ = slb_samples(H, rho0, c_ops, m_eff, N_RUNS_MAX, s_hi)
    bias_lo = bias_sem_rmse(samp_lo, N_RUNS_MAX, reference)[0]
    bias_hi = bias_sem_rmse(samp_hi, N_RUNS_MAX, reference)[0]
    rel = abs(bias_hi - bias_lo) / max(bias_lo, 1e-12)
    ok = rel <= SUBSTEPS_TOL
    flag = "OK (floor is bundling bias, not timestep)" if ok else \
        f"WARNING: bias moved {rel:.0%} -- RAISE SUBSTEPS"
    print(f"  substeps guard (M={m_eff}): bias {bias_lo:.3e} -> {bias_hi:.3e} "
          f"on {s_lo}->{s_hi} substeps  [{flag}]")
    return ok


def _mcsolve_samples(H, psi0, c_ops, ntraj):
    """One mcsolve call; return per-trajectory <H> array (ntraj, n_times)."""
    try:
        res = qutip.mcsolve(H, psi0, TLIST, c_ops, e_ops=[H], ntraj=ntraj,
                            options=MC_OPTIONS)
    except (TypeError, KeyError):
        res = qutip.mcsolve(H, psi0, TLIST, c_ops, e_ops=[H], ntraj=ntraj,
                            options={"progress_bar": False, "keep_runs_results": True})
    return np.real(np.array([res.runs_expect[0][k] for k in range(ntraj)]))


# ===========================================================================
# FRONTIER BENCHMARK
# ===========================================================================
def frontier(name, build, size):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    c_ops = davies_operators(H, X, gamma)
    n_l = len(c_ops)

    print(f"\n[{name}] dim={H.shape[0]}, original Lindblad operators N_L={n_l}")
    reference = np.real(qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H]).expect[0])

    m_values = capped_unique_m_values(n_l)
    substeps_guard(H, rho0, c_ops, reference, min(SUBSTEPS_PROBE_M, n_l))

    # ---- SLB: draw N_RUNS_MAX runs per M, subsample for the N sweep ----
    slb = {n: {"cost": [], "rmse": [], "sem": []} for n in N_RUNS_SWEEP}
    per_run_times = []
    print("  bundling (sweep M):")
    for m_eff in m_values:
        samples, per_run = slb_samples(H, rho0, c_ops, m_eff, N_RUNS_MAX, SUBSTEPS)
        per_run_times.append(per_run)
        line = f"    M={m_eff:3d}  one-run={per_run*1000:6.1f}ms |"
        for n in N_RUNS_SWEEP:
            bias, sem, rmse = bias_sem_rmse(samples[:n], n, reference)
            slb[n]["cost"].append(n * per_run)
            slb[n]["rmse"].append(rmse)
            slb[n]["sem"].append(sem)
            line += f"  N={n}: rmse={rmse:.2e}"
        print(line)

    # ---- mcsolve: one call per ntraj; error bar = S/sqrt(ntraj) ----
    mc = {"cost": [], "rmse": [], "sem": []}
    print("  mcsolve (sweep ntraj):")
    for nt in NTRAJ_VALUES:
        t0 = time.perf_counter()
        runs = _mcsolve_samples(H, psi0, c_ops, nt)
        dt = time.perf_counter() - t0
        bias, sem, rmse = bias_sem_rmse(runs, nt, reference)
        mc["cost"].append(dt)
        mc["rmse"].append(rmse)
        mc["sem"].append(sem)
        print(f"    ntraj={nt:5d}  time={dt:7.3f}s  rmse={rmse:.3e} (bias {bias:.2e}, sem {sem:.2e})")

    return {
        "dim": H.shape[0], "n_l": n_l, "m_values": m_values,
        "per_run_times": per_run_times, "slb": slb,
        "ntraj_values": list(NTRAJ_VALUES), "mc": mc,
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # blue gradient generated from N_RUNS_SWEEP, so any choice of run counts
    # works (lighter = fewer runs, darker = more)
    _blues = plt.cm.Blues(np.linspace(0.45, 0.9, len(N_RUNS_SWEEP)))
    slb_blues = {n: _blues[i] for i, n in enumerate(sorted(N_RUNS_SWEEP))}

    for name, build, size in SYSTEMS:
        out = frontier(name, build, size)

        fig, ax = plt.subplots(figsize=(7.2, 5.2))

        for n in N_RUNS_SWEEP:
            d = out["slb"][n]
            ax.errorbar(d["cost"], d["rmse"], yerr=d["sem"],
                        fmt="s-", color=slb_blues[n], lw=1.8, ms=6, capsize=3,
                        label=f"SLB (N={n} runs)")
        n_lo = min(N_RUNS_SWEEP)
        d_lo = out["slb"][n_lo]
        for x, y, m_eff in zip(d_lo["cost"], d_lo["rmse"], out["m_values"]):
            ax.annotate(f"M={m_eff}", (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=slb_blues[n_lo])

        ax.errorbar(out["mc"]["cost"], out["mc"]["rmse"], yerr=out["mc"]["sem"],
                    fmt="o--", color="tab:purple", lw=1.8, ms=7, capsize=3,
                    label="qutip.mcsolve")
        for x, y, nt in zip(out["mc"]["cost"], out["mc"]["rmse"], out["ntraj_values"]):
            ax.annotate(f"{nt}", (x, y), textcoords="offset points",
                        xytext=(5, -11), fontsize=7, color="tab:purple")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock cost (s)   (lower is better)")
        ax.set_ylabel(r"time-averaged RMSE in $\langle H\rangle$   (lower is better)")
        ax.set_title(
            rf"{name} (dim {out['dim']}, $N_L$={out['n_l']}): RMSE-vs-cost frontier"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()

        tmin = min(out["per_run_times"]) * 1000
        tmax = max(out["per_run_times"]) * 1000
        slb_caption = (f"SLB: sweep M={out['m_values']}, {SUBSTEPS} RK4 substep(s)/step, "
                       f"N in {N_RUNS_SWEEP} runs (one run {tmin:.0f}-{tmax:.0f} ms)")
        mc_caption = (f"mcsolve: sweep ntraj={out['ntraj_values']}, single-thread, "
                      f"atol={MC_ATOL:g}/rtol={MC_RTOL:g}")
        add_settings_footer(
            fig, slb_caption, mc_caption,
            "metric = time-averaged RMSE; error bar = S/sqrt(N) (each method's "
            "own sample spread); one estimate per point; full-Lindblad reference",
        )
        fig.savefig(f"benchmark_frontier_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved benchmark_frontier_{name}.png")


if __name__ == "__main__":
    main()
