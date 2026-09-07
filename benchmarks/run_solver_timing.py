"""
run_solver_timing.py
====================

TIMING ONLY: what one solve costs, per method, per size. No accuracy sweep.

WHY THIS EXISTS
---------------
``run_cost_scaling.py`` answers "how accurate, at what cost", and spends almost
all of its time on the accuracy half: at every dimension it runs ``N_ACC=16``
realizations at each of 8 bundle sizes -- 128 solves. Measured on System A at 9
and 10 spins (job 19603810, 17 h 54 m in total):

    size 10, dim 1024   native reference     2,594 s     4%
                        SLB at M=8             130 s     0.2%
                        accuracy sweep      51,706 s    93%

When the question is only *how far can this method go before it is too slow*,
that sweep is pure cost. This script drops it. The same System A point takes
about 45 minutes here instead of 14 hours.

**This is not a replacement for run_cost_scaling.py.** It produces no accuracy
data, no RMSE, no iso-accuracy curve, and nothing in Results 1-5 can be plotted
from it. It answers exactly one question: what does one solve cost.

COMPARABILITY -- READ BEFORE QUOTING A NUMBER
---------------------------------------------
Uses ``common.TLIST`` and the same substep conventions as
``run_cost_scaling.py``, so a figure from here sits on the same curve as one
from there -- *provided the thread count matches*. It does not always. Job
19603810 ran 32 threads where Result 2's committed data used 4, and its 9-spin
numbers came out 2.3x faster for that reason alone, on both the reference and
the SLB solve. ``meta.execution.threads`` records the setting; check it before
comparing anything.

Substeps must match too, and they are NOT the same across systems. Result 2's
committed data uses 4 (SLB) and 8 (reference) on both chains, but **32 and 64 on
the oscillator**, which diverges at 4. Pass ``--slb-substeps 32`` for the
oscillator or the numbers will not sit on the same curve. This script does not
invent its own ladder; there are already two conventions in the codebase and a
third would help nobody.

The same applies to node exclusivity. Wall-clocks from a job submitted without
``--exclusive`` have been measured up to 10x slow, in proportion to problem
size. Use ``--exclusive`` for anything you intend to quote.

``mesolve`` IS CAPPED, and the cap is the point. Its superoperator has dimension
``N^2``, so memory grows as ``N^4``: 268 MB at dim 64, 4.3 GB at 128, **68.7 GB
at 256**, 1.1 TB at 512. Jobs 19604455 and 19604456 were OOM-killed at dim 256
because an earlier version of this script had no cap. ``--max-full-dim``
defaults to ``common.MAX_FULL_DIM`` exactly as in ``run_cost_scaling.py``; the
projected size is printed so raising it is an informed choice. An OOM-kill is
SIGKILL and cannot be caught, so the guard has to prevent the attempt.

Writes ``data/solver_timing_<system>.json`` **after every method**, so a crash
keeps whatever completed. That is not a hypothetical: an earlier frontier runner
built its results in memory and discarded them, losing 38 hours of compute.

Run:  python run_solver_timing.py --system spin_chain --sizes 9 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    build_spin_chain, build_oscillator_bath, build_mixed_field_chain,
    build_davies_operators, TLIST, SUBSTEPS, MAX_FULL_DIM,
    run_metadata, save_data, DATA_DIR,
)
from benchmark_cli import (
    add_max_full_dim_argument, add_safety_arguments, preflight_run,
    selected_systems,
)
from qutip_bundling import mesolve_ensemble
try:
    from qutip_bundling import SolverInstabilityError
except ImportError:
    from qutip_bundling.native_solver import SolverInstabilityError
from qutip_bundling.native_solver import rk4_mesolve

NATIVE_REF_SUBSTEPS = 2 * SUBSTEPS   # matches run_cost_scaling.py exactly
M_REP = 8                            # representative bundle size, as elsewhere
MC_NTRAJ_PROBE = 8                   # trajectories timed to get a per-trajectory
                                     # cost; mcsolve's real cost is that number
                                     # times ntraj*, which only the accuracy
                                     # side can determine -- so this reports the
                                     # unit, not a total.
RNG_TIMING = 0                       # seed for the timed SLB solve

# (builder, default sizes). The size lists match run_cost_scaling.py exactly,
# so running this with no --sizes reproduces Result 2's grid rather than some
# grid of its own. Pass --sizes to go past it; that is the point of the script.
SYSTEMS = {
    "spin_chain":      (build_spin_chain,        [2, 3, 4, 5, 6, 7, 8, 9]),
    "oscillator_bath": (build_oscillator_bath,   [4, 8, 16, 32, 64, 128]),
    "mixed_chain":     (build_mixed_field_chain, [2, 3, 4, 5, 6, 7]),
}

METHODS = ("native", "mesolve", "slb", "mcsolve")


def liouvillian_bytes(dim):
    """Dense superoperator size for qutip.mesolve: (N^2)^2 complex128.

    qutip stores it sparsely, but with a large collapse-operator list it fills
    in, and this is the number that decides whether the job survives. Jobs
    19604455 and 19604456 were OOM-killed at dim 256, where this is 68.7 GB.
    """
    return (dim ** 4) * 16


def timed(fn, repeats):
    """Run fn `repeats` times; return (median, all samples).

    The median rather than the minimum: a fastest-of-N hides exactly the
    contention this project keeps being bitten by, where the interesting signal
    is that some runs are slow.
    """
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)), samples


def measure(system, size, methods, repeats, sub, native_sub,
            max_full_dim, out_name, meta, points):
    build = SYSTEMS[system][0]
    H, X, psi0 = build(size)
    dim = H.shape[0]
    rho0 = psi0 * psi0.dag()

    t0 = time.perf_counter()
    c_ops = build_davies_operators(H, X)
    t_build = time.perf_counter() - t0
    n_l = len(c_ops)

    row = {"size": size, "dim": dim, "n_l": n_l,
           "t_operator_build": t_build,
           "slb_substeps": sub, "native_substeps": native_sub,
           "timings": {}}

    print(f"\n--- {system}  size {size}  dim {dim}  N_L {n_l:,} ---")
    print(f"  operator list built in {t_build:.1f}s")

    def record(name, fn, note=None):
        if name not in methods:
            return
        try:
            median, samples = timed(fn, repeats)
        except SolverInstabilityError as exc:
            row["timings"][name] = {"diverged": True, "error": str(exc)[:200]}
            print(f"  {name:>8}: DIVERGED -- {str(exc)[:80]}")
            return
        except MemoryError:
            row["timings"][name] = {"out_of_memory": True}
            print(f"  {name:>8}: OUT OF MEMORY")
            return
        entry = {"median_s": median, "samples_s": samples}
        if note:
            entry.update(note)
        row["timings"][name] = entry
        print(f"  {name:>8}: {median:9.2f}s" + (f"   ({note})" if note else ""))
        # After each METHOD, not each size: an OOM-kill is SIGKILL and cannot
        # be caught, so both jobs that died lost a completed native solve too.
        save_data(out_name, meta, points=points + [row])

    record("native", lambda: rk4_mesolve(
        H, rho0, TLIST, c_ops=c_ops, e_ops=[H], substeps=native_sub))

    liou = liouvillian_bytes(dim)
    if "mesolve" in methods and dim > max_full_dim:
        row["timings"]["mesolve"] = {
            "skipped": True,
            "reason": f"dim {dim} > --max-full-dim {max_full_dim}",
            "projected_liouvillian_bytes": liou,
        }
        print(f"  {'mesolve':>8}: SKIPPED -- dim {dim} exceeds --max-full-dim "
              f"{max_full_dim}; its superoperator alone would need "
              f"{liou / 1e9:.1f} GB")
    else:
        if "mesolve" in methods:
            print(f"           (mesolve superoperator ~{liou / 1e9:.2f} GB)")
        record("mesolve", lambda: qutip.mesolve(
            H, rho0, TLIST, c_ops=c_ops, e_ops=[H]))

    # M cannot exceed the number of operators there are to bundle.
    m_rep = min(M_REP, n_l)
    row["m_rep"] = m_rep
    record("slb", lambda: mesolve_ensemble(
        H, rho0, TLIST, c_ops, M=m_rep, e_ops=[H], n_realizations=1,
        rng=RNG_TIMING, backend="native", substeps=sub))

    if "mcsolve" in methods:
        try:
            median, samples = timed(lambda: qutip.mcsolve(
                H, psi0, TLIST, c_ops, e_ops=[H], ntraj=MC_NTRAJ_PROBE,
                options={"progress_bar": False}), repeats)
            row["timings"]["mcsolve"] = {
                "median_s": median, "samples_s": samples,
                "ntraj_probe": MC_NTRAJ_PROBE,
                "per_trajectory_s": median / MC_NTRAJ_PROBE,
            }
            print(f"  {'mcsolve':>8}: {median:9.2f}s for {MC_NTRAJ_PROBE} "
                  f"trajectories = {median / MC_NTRAJ_PROBE:.3g}s each")
        except Exception as exc:                       # noqa: BLE001
            row["timings"]["mcsolve"] = {"failed": True, "error": str(exc)[:200]}
            print(f"  {'mcsolve':>8}: FAILED -- {str(exc)[:80]}")

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[4])
    add_safety_arguments(ap, SYSTEMS)
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="model sizes (spin counts, or Fock cutoffs for the "
                         "oscillator) -- NOT Hilbert dimensions. Defaults to "
                         "the same grid run_cost_scaling.py uses for that "
                         "system; pass this to go past it.")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=METHODS,
                    help="which solvers to time (default: all four)")
    add_max_full_dim_argument(ap, MAX_FULL_DIM)
    ap.add_argument("--slb-substeps", type=int, default=SUBSTEPS,
                    help="RK4 substeps for the SLB solve. Default %(default)s "
                         "matches Result 2's spin and mixed chains. THE "
                         "OSCILLATOR NEEDS 32: its committed data was taken "
                         "that way, and 4 diverges at large Fock cutoff.")
    ap.add_argument("--native-ref-substeps", type=int, default=None,
                    help="substeps for the native full-dissipator reference "
                         "(default: 2x --slb-substeps, as in "
                         "run_cost_scaling.py -- so 8 for the chains, and 64 "
                         "for the oscillator once --slb-substeps is 32)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="wall-clock samples per solve; the median is reported "
                         "and every sample is kept")
    ap.add_argument("--out", default=None,
                    help="output filename under data/ "
                         "(default: solver_timing_<system>.json)")
    args = ap.parse_args()

    if args.sizes and any(s <= 0 for s in args.sizes):
        ap.error("--sizes values must be positive")

    sub = args.slb_substeps
    native_sub = (args.native_ref_substeps if args.native_ref_substeps
                  else 2 * sub)
    if "oscillator_bath" in (
            [args.system] if args.system else list(SYSTEMS)) and sub < 32:
        print(f"WARNING: --slb-substeps {sub} on the oscillator. Result 2's "
              f"committed data uses 32, and 4 diverges at large Fock cutoff. "
              f"These numbers will not be comparable with it.")

    names = selected_systems(args, SYSTEMS)
    if args.out and len(names) > 1:
        ap.error("--out names a single file; use it with --system, not --all")

    sizes_for = {name: args.sizes or SYSTEMS[name][1] for name in names}
    plans = [(f"{name}: sizes {sizes_for[name]}, methods {args.methods}, "
              f"SLB substeps {sub}, reference substeps {native_sub}, "
              f"mesolve capped at dim {args.max_full_dim}",
              DATA_DIR / (args.out or f"solver_timing_{name}.json"))
             for name in names]
    if not preflight_run(plans, overwrite=args.overwrite, dry_run=args.dry_run):
        return

    meta = run_metadata(
        tlist=TLIST, substeps=sub,
        systems=names, sizes={k: list(v) for k, v in sizes_for.items()},
        methods=list(args.methods), repeats=args.repeats,
        m_rep=M_REP, native_substeps=native_sub,
        max_full_dim=args.max_full_dim,
        mc_ntraj_probe=MC_NTRAJ_PROBE,
        purpose="wall-clock only; no accuracy data -- see module docstring",
    )

    print("NOTE: numbers are comparable only at matching thread counts and "
          "node exclusivity -- meta.execution records both.")
    print(f"substeps: SLB {sub}, native reference {native_sub}")

    for name in names:
        out_name = args.out or f"solver_timing_{name}.json"
        print(f"\ntiming {name}: sizes {sizes_for[name]}, "
              f"methods {args.methods}, {args.repeats} sample(s) each")
        points = []
        for size in sizes_for[name]:
            points.append(measure(name, size, set(args.methods),
                                  args.repeats, sub, native_sub,
                                  args.max_full_dim, out_name, meta, points))
            save_data(out_name, meta, points=points)

    print("\ndone.")


if __name__ == "__main__":
    main()
