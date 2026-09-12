"""
run_accuracy_vs_M.py
====================

DATA-GENERATION HALF of the accuracy-versus-bundle-size benchmark (Result 1).
All the compute, none of the plotting; the figures are drawn from the saved
data by plot_accuracy_vs_M.py.

For each system, at one fixed reference-feasible size, it records:

  * construction vs dynamics, timed separately: how long building the N_L
    Davies/Lindblad operators takes, versus how long the reference solve and
    each SLB ensemble propagation take. The two are different costs with
    different scalings, and the saved numbers keep them distinct instead of
    blurring them into one.
  * the exact reference dynamics for ALL observables defined by
    ``common.observable_set`` for that system -- typically energy, every
    Hamiltonian sub-term, and the dominant coherence. Saving the full set
    means the error-decomposition plots can cover every observable without a
    re-run.
  * SLB, for each bundle size M on the ladder: the raw per-realization curves
    of every observable (N_REALIZATIONS x n_times each) and the wall-clock of
    that ensemble solve. Saving raw realizations is the point of the split:
    the mean curve, the +/-1 std band, and the peak-error bias/fluctuation
    decomposition are all derived at analysis time, so new views of this data
    (different bands, different error anatomies) need no re-run.

Uses the fine 80-point time grid (TLIST_FINE), like the published Result 1.

The data file is rewritten after EVERY M, not once at the end. These runs
reach tens of hours (the chain at dim 512 is ~41 h; the mixed chain's dim-256
reference alone is 25-35 h), and a job that dies before its single final write
leaves nothing behind -- run_frontier_spins lost 38 hours of compute that way.
``sweep_complete`` in the payload distinguishes a finished ladder from a file
written mid-run.

Writes, per system and dimension:  data/accuracy_vs_M_<system>_dim<D>.json
``--realizations`` sets the ensemble size per M (default 200). Bias
resolution -- the measured bias divided by its own standard error -- grows as
``sqrt(R)``, and at dim 128 the oscillator's falls to ~1.8 from M=16 upward,
which is why Result 1 reports its fitted slope as untrustworthy there. A
non-default count writes ``accuracy_vs_M_<system>_dim<D>_r<R>.json`` rather
than sharing the default file, since every error bar scales as ``1/sqrt(R)``.

Run:                 python run_accuracy_vs_M.py (--system ... | --all)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    build_davies_operators,
    build_spin_chain, build_oscillator_bath, build_mixed_field_chain,
    TLIST_FINE, SUBSTEPS,
    DATA_DIR, MAX_FULL_DIM, observable_set, run_metadata, save_data,
)
from benchmark_cli import (
    add_max_full_dim_argument, add_safety_arguments, preflight_run,
    selected_systems,
)
from qutip_bundling import mesolve_ensemble
from qutip_bundling.native_solver import rk4_mesolve, SolverInstabilityError

DEFAULT_REALIZATIONS = 200
N_REALIZATIONS = DEFAULT_REALIZATIONS
                        # realizations per M (fixed across the ladder: nothing
                        # about the sampling is tuned, so any trend with M is
                        # purely the effect of M). --realizations overrides it;
                        # see output_name for why a different count does not
                        # share a file with the default one.
RNG = 0                 # seed (matches prior runs)
ROUND = 8               # decimals kept for saved curves

# M ladder per system: system-specific so the plots do not become visually
# cluttered when M=8 already sits essentially on the reference.
# Each system sweeps a LIST of size-points (size, m_ladder, substeps). run_*
# computes every point ONCE, saving one file per dim; plot_* picks a dim to
# draw. substeps>4 flags a disclosed higher-resolution run (the oscillator's
# stiff dim-64 needs 16).
NATIVE_REF_SUBSTEPS_FACTOR = 2

SYSTEMS = {
    "spin_chain": (build_spin_chain, [
        (4, [2, 4, 8, 16, 32, 64], 4),
        (5, [2, 4, 8, 16, 32, 64], 4),
        (6, [2, 4, 8, 16, 32, 64], 4),
        # dim 128. The size-invariance claim rested on three dimensions; this
        # makes it four. M is capped at N_L = 43 here.
        (7, [2, 4, 8, 16, 32, 64], 4),
        # dim 256, making five. M is capped at N_L = 57. This is the system
        # where the invariance claim is cheapest to extend, because its N_L
        # grows slowly and the exact reference stays affordable.
        (8, [2, 4, 8, 16, 32, 64], 4),
        # dim 512, making six, and the first size where the ladder is NOT
        # capped: N_L = 73 here, so all six M values run as written. Roughly
        # 41 h, and essentially all of it is the 1,200 SLB solves -- job
        # 19604735 measured the native reference at 413 s on the 40-point
        # grid, so ~830 s on this 80-point one, under 1% of the run.
        (9, [2, 4, 8, 16, 32, 64], 4),
        # dim 1024, making seven, and System A's LAST Result 1 size: dim 2048
        # costs ~60 days at 200 realizations. About 9 days here, of which the
        # native reference is 3.7 h.
        #
        # Priced from the frontier's own points rather than extrapolated. Its
        # dim-1024 timings decompose into a fixed Hamiltonian term and a term
        # linear in M (0.0162 and 0.0937 s per RK4 step); on this section's
        # 80-point, 4-substep grid that is 3,808 s per realization across the
        # whole ladder, so 761,600 s for 200. The same decomposition at dim 512
        # predicted 35.0 h against 36.1 h measured by job 19604740 -- 3%.
        #
        # Worth doing because two trends now need testing, not one. The bias
        # slope drifts shallower after dim 64 (-0.98, -0.98, -0.97, -0.94) and
        # the height exponent falls on every extension (+0.64, +0.61, +0.58).
        # A seventh point tests both; before dim 512 landed there was only a
        # constant to confirm, and confirming a constant is worth much less.
        (10, [2, 4, 8, 16, 32, 64], 4),
    ]),
    # Same sizes as the TFIM chain so the two are directly comparable: they
    # differ only by the longitudinal field.
    "mixed_chain": (build_mixed_field_chain, [
        (4, [2, 4, 8, 16, 32, 64], 4),
        (5, [2, 4, 8, 16, 32, 64], 4),
        (6, [2, 4, 8, 16, 32, 64], 4),
        # dim 128, making four. Expensive: N_L = 8,193 here, and this system's
        # dim-128 exact reference cost 24.6 h in Result 2.
        (7, [2, 4, 8, 16, 32, 64], 4),
        # dim 256, making five. N_L = 32,637 here, and the exact reference --
        # not the SLB sweep -- is what costs: it grew 19x from dim 64 to 128
        # (600 s -> 11,354 s) because N_L quadruples on top of the dimension.
        # Measured by job 19604858: reference 9.4 h, whole job 2.8 d.
        (8, [2, 4, 8, 16, 32, 64], 4),
        # dim 512, making six, and System B's LAST Result 1 size: dim 1024 has
        # an 8.8 TB operator list, more than all four nodes hold.
        #
        # This size was first written off as unreachable on memory grounds.
        # The operator list is 549 GB nominal (N_L = 131,001), and B's dim-256
        # job had peaked at 2.9x its nominal list, which scales to ~1.6 TB
        # against a 1.55 TB node. Two things changed that. probe_memory.py
        # measured the construction at this size directly: 1.88x, a 1.1 TB
        # peak (job 19607139, 25 min). And native_solver.py stopped caching a
        # second dense copy of every operator (the adjoint list), which is
        # where the third 1x had been going. The reference now holds the Qobj
        # list plus ONE dense copy, ~1.1 TB, the same as the construction
        # peak. The sweep streams operators one at a time and never holds a
        # second copy at all. Submit with --mem=1450G --exclusive.
        #
        # Time: ~3 weeks serial on one node. The reference is ~12.5 days --
        # the dim-256 solve took 9.4 h and each spin has cost ~13x, times the
        # self-check's extra 1.5 solves. The ladder is ~9 days: bundle
        # construction dominates at ~7 min per realization and is roughly
        # independent of M, so the six rungs cost about an hour each per
        # realization. Saved after every M, so a partial ladder is kept.
        (9, [2, 4, 8, 16, 32, 64], 4),
    ]),
    "oscillator_bath": (build_oscillator_bath, [
        (8,  [2, 4, 8, 16, 32, 64], 4),
        (16, [2, 4, 8, 16, 32, 64], 4),
        (32, [2, 4, 8, 16, 32, 64], 16),
        # dim 128, making four. 32 substeps, not 16: the anharmonic ladder needs
        # roughly double per octave and 16 is what diverged here in Result 2.
        # Its 64-substep reference was measured at 16,441 s by job 19597388.
        (64, [2, 4, 8, 16, 32, 64], 32),
    ]),
}


def output_name(system, dim):
    """Data file for one sweep, carrying the realization count when it is not
    the default.

    A run at a different R gets its own file instead of overwriting the
    200-realization one, because the two are not interchangeable: every error
    bar in this data scales as 1/sqrt(R), so a figure that assumes 200 would
    silently misreport a run of 800.

    The consequence is deliberate and worth knowing: the plotters read the
    unsuffixed name, so a non-default run does NOT appear in any figure until
    something is written to ask for it. Better that than a plot quietly mixing
    two ensemble sizes on one axis. (This was only half true until 2026-09-12:
    plot_accuracy_vs_M's auto-pick globbed _dim*.json and could have chosen
    the suffixed file by glob order. It now matches the canonical name only.)
    """
    suffix = "" if N_REALIZATIONS == DEFAULT_REALIZATIONS else f"_r{N_REALIZATIONS}"
    return f"accuracy_vs_M_{system}_dim{dim}{suffix}.json"


def capped_unique_m_values(requested, n_lindblad):
    values = []
    for m in requested:
        m_eff = min(int(m), n_lindblad)
        if m_eff > 0 and m_eff not in values:
            values.append(m_eff)
    return values


def run(name, build, size, m_ladder, substeps):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    dim = H.shape[0]

    # --- construction, timed on its own (this is NOT dynamics) ---
    t0 = time.perf_counter()
    c_ops = build_davies_operators(H, X)
    t_davies = time.perf_counter() - t0
    n_l = len(c_ops)

    # --- exact reference: mesolve while feasible, else certified native RK4 ---
    ref_substeps = NATIVE_REF_SUBSTEPS_FACTOR * substeps
    ref_method, ref_selfcheck, ref_states = None, None, None
    if dim <= MAX_FULL_DIM:
        try:
            t0 = time.perf_counter()
            ref_states = qutip.mesolve(H, rho0, TLIST_FINE, c_ops=c_ops,
                                       e_ops=[]).states
            t_reference = time.perf_counter() - t0
            ref_method = "mesolve"
        except MemoryError:
            ref_states = None
    if ref_states is None:
        t0 = time.perf_counter()
        res = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[],
                          substeps=ref_substeps, store_states=True)
        t_reference = time.perf_counter() - t0
        ref_states = res.states
        lo = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[H],
                         substeps=ref_substeps // 2)
        hi = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[H],
                         substeps=ref_substeps)
        dev = float(np.max(np.abs(np.real(hi.expect[0]) - np.real(lo.expect[0]))))
        ref_selfcheck = {"substeps": ref_substeps, "max_abs_dev": dev,
                         "passed": bool(np.isfinite(dev) and dev <= 1e-4)}
        ref_method = f"native_rk4_substeps{ref_substeps}"
        if not ref_selfcheck["passed"]:
            print(f"[{name}] dim={dim}: native reference self-check FAILED "
                  f"(dev {dev:.1e}) -- skipping (uncertifiable).")
            return

    # --- observables: full set from common.observable_set, same as Result 3 ---
    labels, ops, coherence_info = observable_set(name, H, ref_states)
    e_ops = ops
    reference = {label: np.round(np.real(qutip.expect(op, ref_states)), ROUND)
                 for label, op in zip(labels, ops)}

    print(f"[{name}] dim={dim}, N_L={n_l}; Davies {t_davies*1e3:.1f} ms, "
          f"reference ({ref_method}) {t_reference:.2f} s, {substeps} substeps")
    print(f"  observables: {labels}")
    if coherence_info is not None:
        ia, ib = coherence_info["levels"]
        print(f"  coherence on eigenstate pair ({ia},{ib}), "
              f"peak |rho_ab|={coherence_info['peak_abs_rho']:.2e}")

    # --- SLB ensemble per M: raw realizations of every observable ---
    m_values = capped_unique_m_values(m_ladder, n_l)
    ref_energy = np.asarray(reference["energy"], dtype=float)
    guard = 100.0 * (1.0 + float(np.max(np.abs(ref_energy))))
    sweep = []

    def persist():
        """Write everything measured so far, reference included.

        Called before the ladder and again after every M, rather than once at
        the end. These runs are long -- ~41 h for the chain at dim 512, and on
        the mixed chain the reference ALONE costs 25-35 h -- and a single
        write at the end keeps nothing when a job does not reach it.
        run_frontier_spins lost 38 hours of compute to exactly that.

        Rewriting the file costs a few MB against ensemble solves of minutes
        to hours, so it is free in practice. ``sweep_complete`` says whether
        the ladder finished, because a file written mid-run is otherwise
        indistinguishable from one whose later M values soft-diverged.
        """
        meta = run_metadata(
            tlist=TLIST_FINE, max_full_dim=MAX_FULL_DIM,
            system=name, size=size, M_LADDER=m_ladder, substeps=substeps,
            N_REALIZATIONS=N_REALIZATIONS, rng=RNG,
        )
        save_data(output_name(name, dim), meta, compact=True,
                  dim=dim, n_l=n_l, substeps=substeps,
                  reference_method=ref_method,
                  reference_selfcheck=ref_selfcheck,
                  t_davies=t_davies, t_reference=t_reference,
                  observables=labels, coherence=coherence_info,
                  reference=reference,
                  m_values_planned=m_values,
                  sweep_complete=len(sweep) == len(m_values),
                  slb_sweep=sweep)

    persist()
    for m_eff in m_values:
        t0 = time.perf_counter()
        ens = mesolve_ensemble(H, rho0, TLIST_FINE, c_ops, M=m_eff, e_ops=e_ops,
                               n_realizations=N_REALIZATIONS, rng=RNG,
                               backend="native", substeps=substeps)
        dt = time.perf_counter() - t0
        # Divergence guard on the energy channel
        se = np.real(ens.samples[:, 0, :])
        if not np.isfinite(se).all() or float(np.max(np.abs(se))) > guard:
            print(f"    M={m_eff:3d}  SOFT DIVERGENCE at {substeps} substeps -- skipped")
            continue
        # Save per-observable samples keyed by label
        samples = {}
        for idx, label in enumerate(labels):
            samples[label] = np.round(np.real(ens.samples[:, idx, :]), ROUND)
        sweep.append({"M": m_eff, "cost": dt, "samples": samples})
        print(f"    M={m_eff:3d}  ensemble ({N_REALIZATIONS} realizations) "
              f"= {dt:.2f} s")
        persist()

    print(f"  -> {output_name(name, dim)} complete "
          f"({len(sweep)}/{len(m_values)} bundle sizes)")


def main():
    global MAX_FULL_DIM, N_REALIZATIONS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_safety_arguments(ap, SYSTEMS)
    add_max_full_dim_argument(ap, MAX_FULL_DIM)
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="only these Hilbert dims (default: all configured "
                         "sizes; each saved to its own file).")
    ap.add_argument("--realizations", type=int, default=DEFAULT_REALIZATIONS,
                    help=f"realizations per M (default {DEFAULT_REALIZATIONS}). "
                         "Bias resolution grows as sqrt(R), so 4x the "
                         "realizations doubles it. A non-default value writes "
                         "to accuracy_vs_M_<system>_dim<D>_r<R>.json and is "
                         "not read by the plotters.")
    args = ap.parse_args()
    if args.realizations != DEFAULT_REALIZATIONS:
        N_REALIZATIONS = args.realizations
        print(f"[config] {N_REALIZATIONS} realizations per M "
              f"({N_REALIZATIONS / DEFAULT_REALIZATIONS:.1f}x the default cost, "
              f"{(N_REALIZATIONS / DEFAULT_REALIZATIONS) ** 0.5:.1f}x the bias "
              f"resolution); output carries _r{N_REALIZATIONS}")
    if args.max_full_dim != MAX_FULL_DIM:
        MAX_FULL_DIM = args.max_full_dim
        print(f"[config] exact-mesolve dimension cap raised to {MAX_FULL_DIM}")
    names = selected_systems(args, SYSTEMS)
    work = []
    plans = []
    for name in names:
        build, points = SYSTEMS[name]
        available_dims = set()
        for size, m_ladder, substeps in points:
            probe_dim = build(size)[0].shape[0]
            available_dims.add(probe_dim)
            if args.dims and probe_dim not in args.dims:
                continue
            work.append((name, build, size, m_ladder, substeps))
            plans.append((
                f"Result 1: {name}, dim {probe_dim}, M={m_ladder}",
                DATA_DIR / output_name(name, probe_dim),
            ))
        if args.dims:
            missing = sorted(set(args.dims) - available_dims)
            if missing:
                ap.error(
                    f"{name} has no configured Result 1 dimensions {missing}; "
                    f"available dimensions are {sorted(available_dims)}"
                )

    if not preflight_run(
        plans, overwrite=args.overwrite, dry_run=args.dry_run
    ):
        return
    for item in work:
        run(*item)


if __name__ == "__main__":
    main()
