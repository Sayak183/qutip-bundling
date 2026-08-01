"""
run_method_comparison.py
========================

DATA-GENERATION HALF of the four-method comparison. All the compute, none of
the plotting.

Four ways to solve the same Lindblad problem are put on one footing:

  * native   -- full-dissipator RK4 on the density matrix (all N_L operators).
                Also supplies the certified reference.
  * mesolve  -- QuTiP's exact solver. Builds an N^2 x N^2 Liouvillian, so it is
                the first to hit a memory wall; skipped above --max-full-dim.
  * mcsolve  -- QuTiP's trajectory solver at a FIXED ntraj budget. Its accuracy
                is whatever that budget buys; it is never given more.
  * slb      -- this package, swept over a grid of bundle sizes M.

Every method sees the same Hamiltonian, Davies construction, initial state,
time grid, and observables (common.observable_set), and is scored against the
same certified reference. The observables are fixed before any method runs,
because mcsolve cannot store full states and must be told what to measure.

Timing is only meaningful within ONE job on ONE node: the same certified
dimension-256 reference has taken 2,744 s standalone and 86.6 s inside a
sequential sweep on the same machine. Run every dimension you intend to
compare in a single allocation, and treat ratios below ~1.5x as unresolved.

No error is computed here. Raw per-realization SLB samples and per-method
curves are saved, and the accuracy target is applied at plot time, exactly as
in the other Result scripts.

Writes, per system and dimension:  data/method_comparison_<system>_dim<D>.json
Run:  python run_method_comparison.py (--system SYSTEM | --all) [--dims ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import qutip

from common import (
    build_davies_operators, build_spin_chain, build_oscillator_bath,
    DATA_DIR, MAX_FULL_DIM, MC_OPTIONS, SUBSTEPS, TLIST,
    observable_set, reconstruct_energy, run_metadata, save_data,
)
from benchmark_cli import (
    add_max_full_dim_argument, add_safety_arguments, preflight_run,
    selected_systems,
)
from qutip_bundling import mesolve_ensemble, mesolve_jackknife
from qutip_bundling.native_solver import rk4_mesolve, SolverInstabilityError

# Model sizes, not Hilbert dimensions. The spin chain doubles per site; the
# oscillator's dimension is 2 * n_fock.
SYSTEMS = {
    "spin_chain":      (build_spin_chain,      [2, 3, 4, 5, 6]),
    "oscillator_bath": (build_oscillator_bath, [4, 8, 16]),
}

M_GRID = [1, 2, 4, 8, 16, 32]   # bundle sizes swept for SLB
N_RUNS = 16                     # SLB realizations averaged per M
NTRAJ = 500                     # fixed mcsolve budget (see module docstring)
RNG_SLB = 100
REF_SELFCHECK_TOL = 1e-4        # a reference is used only if halving its
                                # substeps barely moves it


def curves_dict(labels, expect):
    return {label: np.real(np.asarray(values))
            for label, values in zip(labels, expect)}


def certified_reference(H, rho0, c_ops, ref_substeps):
    """Native full-dissipator RK4 states plus a halved-substep certification.

    Returns (states, t_reference, selfcheck) or raises if it cannot be
    certified -- an uncertified reference is worse than none, because every
    error in the comparison is measured against it.
    """
    t0 = time.perf_counter()
    primary = rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H],
                          substeps=ref_substeps, store_states=True)
    t_reference = time.perf_counter() - t0

    partner = rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H],
                          substeps=max(ref_substeps // 2, 1))
    deviation = float(np.max(np.abs(np.real(primary.expect[0])
                                    - np.real(partner.expect[0]))))
    selfcheck = {
        "substeps_pair": [max(ref_substeps // 2, 1), ref_substeps],
        "max_abs_dev": deviation,
        "tol": REF_SELFCHECK_TOL,
        "passed": bool(np.isfinite(deviation) and deviation <= REF_SELFCHECK_TOL),
    }
    return primary.states, t_reference, selfcheck


def stored_reference(system, dim, n_sites):
    """Load an already-certified reference instead of recomputing it.

    ``run_high_dim_spin_reference.py`` writes the full density-matrix
    trajectory to an NPZ alongside its certification, so every observable is
    derivable from it. Recomputing that costs ~4 h at dimension 1024 and
    reproduces numbers we already have -- reuse is what makes the comparison
    affordable at the dimensions the reference already reaches.

    Returns (states, t_reference, selfcheck) or None when no usable archive
    exists. Raises if an archive exists but cannot be trusted: a silently
    mismatched reference would poison every error in the comparison.
    """
    if system != "spin_chain":
        return None                      # only the chain has these archives
    json_path = DATA_DIR / f"high_dim_reference_spin_chain_dim{dim}.json"
    npz_path = json_path.with_suffix(".npz")
    if not json_path.exists() or not npz_path.exists():
        return None

    document = json.loads(json_path.read_text(encoding="utf-8"))
    point, meta = document["point"], document["meta"]
    if not point["selfcheck"].get("passed", False):
        raise ValueError(f"{json_path.name} is not a certified reference")

    grid = meta["tlist"]
    if (grid["n"] != len(TLIST)
            or not np.isclose(grid["t0"], TLIST[0])
            or not np.isclose(grid["t1"], TLIST[-1])):
        raise ValueError(
            f"{json_path.name} was computed on a different time grid "
            f"(t0={grid['t0']}, t1={grid['t1']}, n={grid['n']}) than this "
            f"comparison uses ({TLIST[0]}, {TLIST[-1]}, {len(TLIST)})."
        )

    archive = point.get("state_archive", {})
    expected = archive.get("sha256")
    if expected:
        digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{npz_path.name} does not match the SHA-256 recorded in "
                f"{json_path.name}; the archive and its certification have "
                f"diverged, so it cannot be used as a reference."
            )

    with np.load(npz_path) as data:
        states_array = data["states"]
    dims = [[2] * n_sites, [2] * n_sites]
    states = [qutip.Qobj(states_array[i], dims=dims)
              for i in range(states_array.shape[0])]
    return states, float(point["t_reference"]), point["selfcheck"]


def repeat_timed(call, repeats):
    """Run ``call`` ``repeats`` times; return (median wall, all walls, result).

    Wall-clock on a shared machine varies by tens of percent between runs even
    on the same node, so a single measurement cannot resolve a ratio below
    roughly 1.5x. The median is reported and every measurement kept, so a
    figure can show the spread instead of implying a precision the timing does
    not have. The computation is identical each time (fixed seeds), so only the
    timing varies.
    """
    walls, result = [], None
    for _ in range(max(int(repeats), 1)):
        start = time.perf_counter()
        result = call()
        walls.append(time.perf_counter() - start)
    return float(np.median(walls)), walls, result


def run_native(H, rho0, c_ops, ops, substeps):
    return rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=ops,
                       substeps=substeps).expect


def run_mesolve(H, rho0, c_ops, ops):
    return qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=ops).expect


def run_mcsolve(H, psi0, c_ops, ops, ntraj):
    """Fixed-budget mcsolve. Returns (mean curves, per-trajectory std).

    Trajectories are averaged here rather than saved: keeping every trajectory
    for every observable is what the fixed-budget design exists to avoid.
    """
    res = qutip.mcsolve(H, psi0, TLIST, c_ops, e_ops=ops, ntraj=ntraj,
                        options=MC_OPTIONS)
    # MC_OPTIONS sets keep_runs_results, so `expect` is per-trajectory with
    # shape (ntraj, n_times) rather than an average -- averaging it here is
    # what makes the saved curve a curve. run_frontier.py reads runs_expect
    # the same way.
    runs = getattr(res, "runs_expect", None)
    if runs is not None:
        per_traj = [np.real(np.asarray(r)) for r in runs]
        mean = [p.mean(axis=0) for p in per_traj]
        std = [p.std(axis=0, ddof=1) for p in per_traj]
    else:
        mean = [np.real(np.asarray(e)) for e in res.expect]
        std = [np.full_like(m, np.nan) for m in mean]
    return mean, std


def run_slb(H, rho0, c_ops, ops, m_values, n_runs, substeps, n_l, repeats):
    """Sweep M. Saves raw per-realization samples so the plot side can pick
    any accuracy target later without recomputing."""
    rows, seen = [], set()
    for m in m_values:
        m_eff = min(int(m), n_l)          # a bundle cannot exceed the operators
        if m_eff in seen or m_eff < 1:
            continue
        seen.add(m_eff)
        try:
            wall, walls, ens = repeat_timed(
                lambda: mesolve_ensemble(
                    H, rho0, TLIST, c_ops, M=m_eff, e_ops=ops,
                    n_realizations=n_runs, rng=RNG_SLB,
                    backend="native", substeps=substeps),
                repeats)
        except SolverInstabilityError as err:
            print(f"      M={m_eff:3d}  diverged: {err}")
            rows.append({"M": m_eff, "diverged": True})
            continue
        # samples: (n_runs, n_obs, n_times)
        rows.append({
            "M": m_eff,
            "n_runs": n_runs,
            "wall_s": wall,
            "wall_s_repeats": walls,
            "samples": np.asarray(ens.samples),
        })
        print(f"      M={m_eff:3d}  {wall:8.2f} s")
    return rows


def run_jackknife(H, rho0, c_ops, ops, m_values, n_runs, substeps, n_l, repeats):
    """Sweep M with the jackknife-2 estimator (eqs. 15-16).

    Finite M leaves an O(1/M) bias because the dissipator noise enters the
    density matrix nonlinearly. The jackknife combines the full bundle with its
    two halves to cancel the leading term. Saving both the corrected and the
    uncorrected samples is the point: the bias reduction is only visible as the
    difference between them, measured against the same reference.

    M must be even, so odd entries in the grid are skipped rather than silently
    rounded.
    """
    rows, seen = [], set()
    for m in m_values:
        m_eff = min(int(m), n_l)
        if m_eff in seen or m_eff < 2 or m_eff % 2:
            continue
        seen.add(m_eff)
        try:
            wall, walls, jack = repeat_timed(
                lambda: mesolve_jackknife(
                    H, rho0, TLIST, c_ops, M=m_eff, e_ops=ops,
                    n_realizations=n_runs, rng=RNG_SLB,
                    backend="native", substeps=substeps),
                repeats)
        except SolverInstabilityError as err:
            print(f"      M={m_eff:3d}  diverged: {err}")
            rows.append({"M": m_eff, "diverged": True})
            continue
        rows.append({
            "M": m_eff,
            "n_runs": n_runs,
            "wall_s": wall,
            "wall_s_repeats": walls,
            "samples": np.asarray(jack.samples),                     # corrected
            "direct_samples": np.asarray(jack.extra["direct_samples"]),
        })
        print(f"      M={m_eff:3d}  {wall:8.2f} s  (jackknife)")
    return rows


def run(name, build, size, args):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    dim = H.shape[0]

    t0 = time.perf_counter()
    c_ops = build_davies_operators(H, X)
    t_davies = time.perf_counter() - t0
    n_l = len(c_ops)
    print(f"[{name}] dim={dim}  N_L={n_l}  Davies {t_davies:.3f} s")

    ref_substeps = args.ref_substeps or 2 * args.slb_substeps
    reused = None
    if args.reuse_reference:
        n_sites = len(H.dims[0]) if name == "spin_chain" else 0
        reused = stored_reference(name, dim, n_sites)
        if reused is None:
            print(f"  no stored reference for {name} dim {dim}; computing one")
    if reused is not None:
        states, t_reference, selfcheck = reused
        print(f"  reference reused from the certified archive "
              f"({t_reference:.1f} s as originally measured)")
    else:
        states, t_reference, selfcheck = certified_reference(
            H, rho0, c_ops, ref_substeps)
    if not selfcheck["passed"]:
        print(f"  reference self-check FAILED (dev {selfcheck['max_abs_dev']:.2e})"
              f" -- skipping this dimension")
        return None
    if reused is None:
        print(f"  reference certified ({t_reference:.2f} s, "
              f"substeps {ref_substeps}, dev {selfcheck['max_abs_dev']:.2e})")

    # Observables are fixed here, from the reference, and shared by every
    # method for the rest of this dimension.
    labels, ops, coherence = observable_set(name, H, states)
    reference = curves_dict(labels, [qutip.expect(op, states) for op in ops])
    print(f"  observables: {labels}")

    residual = reconstruct_energy(name, reference)
    energy_check = (float(np.max(np.abs(residual - reference["energy"])))
                    if residual is not None else None)
    if energy_check is not None:
        print(f"  energy reconstruction residual {energy_check:.2e}")

    methods: dict = {}

    if "native" in args.methods:
        wall, walls, expect = repeat_timed(
            lambda: run_native(H, rho0, c_ops, ops, args.slb_substeps),
            args.repeats)
        methods["native"] = {"wall_s": wall, "wall_s_repeats": walls,
                             "curves": curves_dict(labels, expect)}
        print(f"    native   {wall:8.2f} s")

    if "mesolve" in args.methods:
        if dim <= args.max_full_dim:
            try:
                wall, walls, expect = repeat_timed(
                    lambda: run_mesolve(H, rho0, c_ops, ops), args.repeats)
                methods["mesolve"] = {"wall_s": wall, "wall_s_repeats": walls,
                                      "curves": curves_dict(labels, expect)}
                print(f"    mesolve  {wall:8.2f} s")
            except MemoryError as err:
                methods["mesolve"] = {"skipped": f"MemoryError: {err}"}
                print(f"    mesolve  skipped (MemoryError)")
        else:
            reason = f"dim {dim} exceeds --max-full-dim {args.max_full_dim}"
            methods["mesolve"] = {"skipped": reason}
            print(f"    mesolve  skipped ({reason})")

    if "mcsolve" in args.methods:
        wall, walls, (mean, std) = repeat_timed(
            lambda: run_mcsolve(H, psi0, c_ops, ops, args.ntraj), args.repeats)
        methods["mcsolve"] = {
            "ntraj": args.ntraj,
            "wall_s": wall,
            "wall_s_repeats": walls,
            "curves": dict(zip(labels, mean)),
            "traj_std": dict(zip(labels, std)),
        }
        print(f"    mcsolve  {wall:8.2f} s  (ntraj={args.ntraj})")

    if "slb" in args.methods:
        methods["slb"] = run_slb(H, rho0, c_ops, ops, args.m_grid,
                                 args.n_runs, args.slb_substeps, n_l,
                                 args.repeats)

    if "jackknife" in args.methods:
        methods["jackknife"] = run_jackknife(
            H, rho0, c_ops, ops, args.m_grid, args.n_runs,
            args.slb_substeps, n_l, args.repeats)

    return {
        "dim": dim,
        "size": size,
        "n_l": n_l,
        "t_davies": t_davies,
        "observables": labels,
        "coherence": coherence,
        "energy_reconstruction_residual": energy_check,
        "reference": {
            "method": f"native_rk4_substeps{ref_substeps}",
            "reused_from_archive": bool(reused is not None),
            "wall_s": t_reference,
            "selfcheck": selfcheck,
            "curves": reference,
        },
        "methods": methods,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_safety_arguments(ap, SYSTEMS)
    add_max_full_dim_argument(ap, MAX_FULL_DIM)
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="model sizes to run instead of the configured list "
                         "(these are not Hilbert dimensions)")
    ap.add_argument("--methods", nargs="+", default=["native", "mesolve",
                                                     "mcsolve", "slb"],
                    choices=["native", "mesolve", "mcsolve", "slb",
                             "jackknife"],
                    help="subset of methods to run. 'jackknife' is the "
                         "bias-corrected variant of 'slb' and is not included "
                         "by default, since it costs three solves per "
                         "realization instead of one.")
    ap.add_argument("--ntraj", type=int, default=NTRAJ,
                    help=f"fixed mcsolve trajectory budget (default {NTRAJ}). "
                         f"Accuracy is whatever this buys; it is reported, not "
                         f"tuned to hit a target.")
    ap.add_argument("--m-grid", type=int, nargs="+", default=M_GRID,
                    help="bundle sizes swept for SLB")
    ap.add_argument("--n-runs", type=int, default=N_RUNS,
                    help="SLB realizations averaged per M")
    ap.add_argument("--reuse-reference", action="store_true",
                    help="load the certified reference from "
                         "data/high_dim_reference_*.{json,npz} instead of "
                         "recomputing it (spin chain only; the archive stores "
                         "the full density trajectory, so every observable is "
                         "derivable). Recomputing costs ~4 h at dimension 1024 "
                         "and reproduces numbers already in hand. The archive's "
                         "SHA-256 and time grid are verified before use.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="time every method this many times and record all "
                         "measurements (default 1). Wall-clock varies by tens "
                         "of percent between runs, so ratios below ~1.5x need "
                         "repeats to be resolved. Multiplies the runtime.")
    ap.add_argument("--slb-substeps", type=int, default=SUBSTEPS,
                    help="RK4 substeps for SLB and the native solve")
    ap.add_argument("--ref-substeps", type=int, default=None,
                    help="substeps for the certified reference "
                         "(default: 2x --slb-substeps)")
    args = ap.parse_args()

    names = selected_systems(args, SYSTEMS)
    work, plans = [], []
    for name in names:
        build, configured = SYSTEMS[name]
        sizes = args.sizes if args.sizes else configured
        work.append((name, build, sizes))
        for size in sizes:
            dim = 2 ** size if name == "spin_chain" else 2 * size
            plans.append((
                f"four-method comparison: {name}, size={size} (dim {dim})",
                DATA_DIR / f"method_comparison_{name}_dim{dim}.json",
            ))

    if not preflight_run(plans, overwrite=args.overwrite, dry_run=args.dry_run):
        return

    for name, build, sizes in work:
        for size in sizes:
            point = run(name, build, size, args)
            if point is None:
                continue
            meta = run_metadata(
                max_full_dim=args.max_full_dim,
                system=name, size=size, methods=args.methods,
                M_GRID=args.m_grid, N_RUNS=args.n_runs, NTRAJ=args.ntraj,
                repeats=args.repeats, reused_reference=args.reuse_reference,
                substeps=args.slb_substeps,
                ref_substeps=args.ref_substeps or 2 * args.slb_substeps,
                rng_slb=RNG_SLB,
                mc_atol=MC_OPTIONS["atol"], mc_rtol=MC_OPTIONS["rtol"],
            )
            # compact: these files carry raw per-realization sample arrays,
            # where one-number-per-line indentation inflates them several-fold.
            save_data(f"method_comparison_{name}_dim{point['dim']}.json",
                      meta, compact=True, point=point)


if __name__ == "__main__":
    main()
