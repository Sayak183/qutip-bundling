"""
common.py
=========

Shared physics, configuration, figure helpers, and data I/O for the benchmark
suite. Single source of truth: every benchmark builds the SAME two systems with
the SAME bath and integrator settings, so results across figures are directly
comparable and no two scripts can silently drift apart.

Contents:
  * bath spectral density `gamma` and its constants
  * the two model systems: `build_spin_chain`, `build_oscillator_bath`
  * global run settings: TLIST, SUBSTEPS, FULL_TIME_BUDGET, MAX_FULL_DIM
  * error metric: `tavg_rmse` and its jackknife uncertainty `tavg_rmse_jackknife`
  * figure caption helpers: `format_slb_settings`, `format_mcsolve_settings`,
    `add_settings_footer`
  * data I/O for the run/plot split: `save_data`, `load_data`, `as_array`.
    Run scripts (run_*.py) do the compute and write JSON into data/;
    plot scripts (plot_*.py) read the JSON and draw figures. Every data file
    carries a metadata block (package versions, timestamp, seeds, parameter
    grid) so any figure is traceable to the exact run that produced it.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import platform
import socket
from pathlib import Path

import numpy as np
import qutip

from qutip_bundling import (
    __version__ as QUTIP_BUNDLING_VERSION,
    davies_operators,
)

# ===========================================================================
# GLOBAL RUN SETTINGS
# ===========================================================================
SUBSTEPS = 4                # RK4 substeps per TLIST step for SLB (native backend).
                            # >=2 is required for stability on the stiffer
                            # oscillator at larger sizes (substeps=1 diverges
                            # there); the result is already converged by 2, so 4
                            # carries margin at negligible cost. Stated on the
                            # plot for a fair comparison against mcsolve's ntraj.

FULL_TIME_BUDGET = 60.0     # stop full mesolve once one solve exceeds this
MAX_FULL_DIM = 32           # never attempt full mesolve above this dimension;
                            # corrected dim-64 spin took 11,160 s, so the
                            # elapsed-time budget cannot protect that call
                            # after it has already started
MAX_NATIVE_REF_DIM = 128    # safe dense full-dissipator RK4 default; dim 128
                            # is certified, dim 256 is an explicit long run,
                            # and dim 512 exceeds this laptop's RAM
TLIST = np.linspace(0.0, 5.0, 40)        # cost-scaling grid (Result 2)
TLIST_FINE = np.linspace(0.0, 5.0, 80)   # finer grid used by the accuracy-style
                                         # comparisons (Results 1, 3, 4)

# Fairness controls for mcsolve, shared by every figure that races it:
#  - single-threaded ("map": "serial") so wall-clock matches SLB's serial loop.
#  - keep per-trajectory results so the trajectory spread (-> S/sqrt(ntraj)) is
#    available for the SEM error bar, the same quantity SLB gets from its runs.
MC_ATOL = 1e-8
MC_RTOL = 1e-6
MC_OPTIONS = {"progress_bar": False, "map": "serial",
              "atol": MC_ATOL, "rtol": MC_RTOL, "keep_runs_results": True}

# Shared detailed-balance ohmic bath.
ALPHA, KT, OMEGA_C = 0.3, 0.5, 8.0
# Absolute energy tolerance used both for degenerate eigenspaces and
# equal Bohr-frequency sectors. Keep this central so every benchmark
# uses and records the same strict Davies construction.
DAVIES_DEGENERACY_TOL = 1e-10


def gamma(omega: float) -> float:
    if abs(omega) < 1e-10:
        return ALPHA * KT
    return ALPHA * omega * math.exp(-abs(omega) / OMEGA_C) / (1.0 - math.exp(-omega / KT))


def build_davies_operators(H, X):
    """Strict grouped-frequency Davies operators for every benchmark."""
    return davies_operators(
        H,
        X,
        gamma,
        degeneracy_tol=DAVIES_DEGENERACY_TOL,
    )


# ===========================================================================
# SYSTEM BUILDERS
# ===========================================================================
MIXED_FIELD_G = 0.4        # longitudinal field of the non-integrable chain


def build_spin_chain(n_sites: int, J: float = 1.0, h: float = 0.6,
                     g: float = 0.0):
    """Dissipative Ising chain in a transverse (and optionally longitudinal) field.

        H = -J sum_i Z_i Z_i+1  -  h sum_i X_i  -  g sum_i Z_i

    ``g=0`` is the transverse-field Ising model: **exactly solvable**. It maps
    to free fermions, so its energies are sums of n independent mode energies
    and enormous numbers of transitions share the same Bohr frequency. The
    Davies construction groups those together, and the operator count collapses
    to N_L = n^2 - n + 1 -- only 31 at dimension 64. There is almost nothing
    left for bundling to compress, which is why this is the *control* system.

    ``g>0`` breaks integrability. Bohr frequencies stop coinciding, grouping
    merges almost nothing, and N_L climbs to about N^2/2 -- 2,017 at dimension
    64 and 8,193 at 128. This is the generic case, and the one the method is
    for. Measured counts at dimension 64: 31 at g=0 against 2,017 at g=0.4,
    with the count saturating by g~0.01.

    Do not use very small nonzero ``g``. Between roughly 1e-8 and 1e-4 the count
    depends on where the splittings fall relative to ``degeneracy_tol``, and
    more importantly the secular approximation behind the Davies construction
    assumes splittings large compared with the bath coupling. Use ``g=0`` or
    ``g >= 0.01``.
    """
    sx, sz, I = qutip.sigmax(), qutip.sigmaz(), qutip.qeye(2)

    def op(o, i):
        return qutip.tensor([o if k == i else I for k in range(n_sites)])

    H = 0
    for i in range(n_sites - 1):
        H += -J * op(sz, i) * op(sz, i + 1)
    for i in range(n_sites):
        H += -h * op(sx, i)
    if g:
        for i in range(n_sites):
            H += -g * op(sz, i)

    X = sum(op(sx, i) for i in range(n_sites))
    psi0 = qutip.tensor([qutip.basis(2, 0)] * n_sites)
    return H, X, psi0


def build_mixed_field_chain(n_sites: int):
    """The same chain with the integrability-breaking longitudinal field on."""
    return build_spin_chain(n_sites, g=MIXED_FIELD_G)


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


# ===========================================================================
# OBSERVABLES
# ===========================================================================
# Every method in a comparison must be scored on the SAME operators, so they
# are defined once here rather than per runner.
#
# <H> alone is not enough. It is built almost entirely from the diagonal of
# rho, so a method can reproduce it while getting the coherences wrong, and on
# System B the spin contributes under 4% of the total energy -- a large error
# there is invisible in <H>. Each system therefore also carries an off-diagonal
# probe and the individual terms of its own Hamiltonian.
OSCILLATOR_PARAMS = dict(omega0=1.0, anh=0.1, spin_gap=1.0, coupling=0.3)
SPIN_CHAIN_PARAMS = dict(J=1.0, h=0.6)


def populated_coherence_op(H, ref_states):
    """Hermitian coherence operator |a><b| + h.c. for the energy-eigenstate pair
    (a, b) whose coherence is *most populated by the actual dynamics* (largest
    |<a|rho(t)|b>| over the reference trajectory). This guarantees we track a
    coherence the system genuinely develops -- picking by coupling strength can
    land on a pair the dynamics never populates (value ~ machine zero), which is
    uninformative. <H> is essentially diagonal, so this off-diagonal is exactly
    what energy cannot see.

    The pair is chosen from the REFERENCE and the resulting operator is then
    handed to every method: if each method picked its own "largest" pair they
    would be scored on different observables and their errors would not be
    comparable.
    """
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


def spin_chain_observables(H):
    """(labels, ops) for System A, excluding the coherence.

    ``zz`` and ``sx`` are the two terms of H itself:
        <H> = -J * <zz> - h * <sx>
    Measuring them separately splits the energy into its interaction and field
    halves. They move non-monotonically and partly cancel, so that structure is
    invisible in <H>; their sum also reconstructs it as a free correctness
    check. ``zz_per_bond`` is the same quantity normalised by the bond count,
    which stays comparable across dimensions.
    """
    n_sites = len(H.dims[0])
    sx, sz, identity = qutip.sigmax(), qutip.sigmaz(), qutip.qeye(2)

    def op(single, site):
        return qutip.tensor([single if k == site else identity
                             for k in range(n_sites)])

    zz = sum(op(sz, i) * op(sz, i + 1) for i in range(n_sites - 1))
    sx_total = sum(op(sx, i) for i in range(n_sites))
    sz_total = sum(op(sz, i) for i in range(n_sites))
    n_bonds = max(n_sites - 1, 1)
    # sz is the longitudinal magnetisation: the third term of H when g>0, and a
    # physically meaningful order parameter either way.
    return (
        ["energy", "zz", "sx", "sz", "zz_per_bond"],
        [H, zz, sx_total, sz_total, zz / n_bonds],
    )


def oscillator_observables(H):
    """(labels, ops) for System B, excluding the coherence.

    The bath couples only through x (x) I, so it drains the oscillator
    directly; ``sz`` moves solely through the internal coupling g and is the
    delicate, second-hand quantity. ``n2`` and ``x_sx`` are the remaining terms
    of H, so that
        <H> = w0*(<n>+1/2) + chi*<n2> + (Delta/2)*<sz> + g*<x_sx>
    reconstructs the energy exactly.
    """
    n_fock = H.dims[0][0]
    a = qutip.destroy(n_fock)
    num = a.dag() * a
    x = (a + a.dag()) / math.sqrt(2.0)
    Io, Is = qutip.qeye(n_fock), qutip.qeye(2)
    sz, sx = qutip.sigmaz(), qutip.sigmax()
    return (
        ["energy", "n", "sz", "n2", "x_sx"],
        [H, qutip.tensor(num, Is), qutip.tensor(Io, sz),
         qutip.tensor(num * num, Is), qutip.tensor(x, sx)],
    )


def observable_set(system, H, ref_states=None):
    """(labels, ops) for ``system``; adds the coherence when states are given.

    Pass the reference states so the coherence operator is chosen once, from
    the reference, and shared by every method.
    """
    if system in ("spin_chain", "mixed_chain"):
        labels, ops = spin_chain_observables(H)
    elif system == "oscillator_bath":
        labels, ops = oscillator_observables(H)
    else:
        raise ValueError(f"unknown system {system!r}")

    coherence_pair = None
    if ref_states is not None:
        C, pair, peak = populated_coherence_op(H, ref_states)
        labels = labels + ["coherence"]
        ops = ops + [C]
        coherence_pair = {"levels": list(pair), "peak_abs_rho": peak}
    return labels, ops, coherence_pair


def reconstruct_energy(system, curves):
    """Rebuild <H> from the individual Hamiltonian terms in ``curves``.

    Returns None when the required terms are absent. Used as a self-check: the
    residual against the measured energy must be at machine precision.
    """
    if system in ("spin_chain", "mixed_chain"):
        if not {"zz", "sx"} <= curves.keys():
            return None
        p = SPIN_CHAIN_PARAMS
        total = (-p["J"] * np.asarray(curves["zz"])
                 - p["h"] * np.asarray(curves["sx"]))
        if system == "mixed_chain":
            if "sz" not in curves:
                return None
            total = total - MIXED_FIELD_G * np.asarray(curves["sz"])
        return total
    if system == "oscillator_bath":
        if not {"n", "n2", "sz", "x_sx"} <= curves.keys():
            return None
        p = OSCILLATOR_PARAMS
        return (p["omega0"] * (np.asarray(curves["n"]) + 0.5)
                + p["anh"] * np.asarray(curves["n2"])
                + 0.5 * p["spin_gap"] * np.asarray(curves["sz"])
                + p["coupling"] * np.asarray(curves["x_sx"]))
    return None


# ===========================================================================
# ERROR METRIC (time-averaged RMSE and its jackknife uncertainty)
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


def tavg_rmse(samples, reference, n_eff=None):
    """Time-averaged RMSE of an n-run SLB estimate against a reference curve.

    samples : (n_runs, n_times) real array of per-run expectation curves.
    Combines the bias of the run-mean and the SEM of the estimate at each time,
    then averages over the trajectory:  mean_t sqrt(bias^2 + sem^2).
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0] if n_eff is None else n_eff
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    bias = np.abs(mean - np.asarray(reference, dtype=float))
    sem = std / np.sqrt(n)
    return float(np.mean(np.sqrt(bias ** 2 + sem ** 2)))


def tavg_bias_sem_rmse(samples, reference, n_eff=None):
    """(time-avg BIAS, time-avg SEM, time-avg RMSE) of an n-run estimate.

    Same statistic family as tavg_rmse, reported in parts: the frontier
    (Result 3) plots the RMSE with the SEM as its error bar, and the substeps
    guard watches the BIAS alone.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0] if n_eff is None else n_eff
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    bias = np.abs(mean - np.asarray(reference, dtype=float))
    sem = std / np.sqrt(n)
    rmse = np.sqrt(bias ** 2 + sem ** 2)
    return float(np.mean(bias)), float(np.mean(sem)), float(np.mean(rmse))


def tavg_rmse_jackknife(samples, reference):
    """(rmse, jackknife std of rmse) by delete-one resampling over the runs.

    Matches the jackknife methodology of the validation appendix: the statistic
    is recomputed on each leave-one-out subset and the spread is scaled by the
    usual (n-1)/n jackknife factor. Gives the error bar for RMSE points.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0]
    full = tavg_rmse(samples, reference)
    if n < 3:
        return full, float("nan")
    loo = np.array([
        tavg_rmse(np.delete(samples, i, axis=0), reference) for i in range(n)
    ])
    var = (n - 1) / n * np.sum((loo - loo.mean()) ** 2)
    return full, float(np.sqrt(var))


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
# DATA I/O (run/plot split)
# ===========================================================================
DATA_DIR = Path(__file__).resolve().parent / "data"


def _execution_context():
    """Host, scheduler, and thread settings that wall-clock times depend on.

    Timings are only comparable when they come from the same machine under the
    same threading. Recording that in the file itself means a table of times
    can be checked for comparability instead of taken on trust: a shared-node
    sweep and a standalone run of the same dimension have differed by more than
    an order of magnitude here purely through node load.
    """
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS")
    context = {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "threads": {name: os.environ[name]
                    for name in thread_vars if name in os.environ},
    }
    # Slurm exports these only inside a batch allocation; their absence is
    # itself informative (the run was not scheduled).
    scheduler = {key: os.environ[env] for key, env in (
        ("job_id", "SLURM_JOB_ID"),
        ("job_name", "SLURM_JOB_NAME"),
        ("nodelist", "SLURM_JOB_NODELIST"),
        ("cpus_per_task", "SLURM_CPUS_PER_TASK"),
        ("partition", "SLURM_JOB_PARTITION"),
    ) if env in os.environ}
    if scheduler:
        context["slurm"] = scheduler
    return context


def run_metadata(tlist=TLIST, substeps=SUBSTEPS, max_full_dim=None, **params):
    """Metadata block stamped into every data file: enough to trace a figure
    back to the exact run (and machine state) that produced its numbers.

    ``max_full_dim`` records the cap actually in force for this run. Runners
    that expose ``--max-full-dim`` must pass their effective value, otherwise
    the metadata would advertise the module default while the run used
    something else.
    """
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "qutip": qutip.__version__,
        "qutip_bundling": QUTIP_BUNDLING_VERSION,
        "platform": platform.platform(),
        "execution": _execution_context(),
        "tlist": {"t0": float(tlist[0]), "t1": float(tlist[-1]),
                  "n": int(len(tlist))},
        "substeps": substeps,
        "full_time_budget_s": FULL_TIME_BUDGET,
        "max_full_dim": MAX_FULL_DIM if max_full_dim is None else int(max_full_dim),
        "max_native_ref_dim": MAX_NATIVE_REF_DIM,
        "davies": {
            "construction": "grouped_frequency_sectors",
            "degeneracy_tol": DAVIES_DEGENERACY_TOL,
            "coupling_block_floor": "512 * eps * dim * frobenius_norm(X_in_energy_basis)",
        },
        "params": params,
    }


def _sanitize(obj):
    """Convert numpy scalars/arrays to plain Python and NaN/inf to None, so the
    output is strict JSON (NaN is not valid JSON and breaks other parsers)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def save_data(filename, meta, compact=False, **payload):
    """Write data/<filename> as strict JSON: {"meta": ..., <payload>}.

    compact=True writes without indentation - use it for files carrying raw
    sample arrays, where one-number-per-line indentation inflates the file
    several-fold for no readability gain."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    doc = _sanitize({"meta": meta, **payload})
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if compact:
            json.dump(doc, fh, separators=(",", ":"), allow_nan=False)
        else:
            json.dump(doc, fh, indent=1, allow_nan=False)
        fh.write("\n")
    print(f"  wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return path


def load_data(filename):
    """Read data/<filename>; raises with a pointer to the run script if absent."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - generate it first with the matching run_*.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def as_array(seq):
    """List from a data file -> float array, mapping JSON null back to NaN."""
    return np.array([np.nan if v is None else v for v in seq], dtype=float)
