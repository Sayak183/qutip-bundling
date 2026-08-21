"""
run_isocost_vs_dim.py
=====================

DATA-GENERATION HALF of the iso-accuracy cost-vs-dimension benchmark (Result 4:
SLB vs mcsolve as the system grows). All the compute, none of the plotting; the
figure is drawn from the saved data by plot_isocost_vs_dim.py.

At each Hilbert dimension (up to the exact-reference wall) it records:

  * the exact reference <H(t)> and its wall-clock cost;
  * SLB: an ascending bundle-size sweep. For each M, enough independent runs
    are drawn ONCE to cover the largest averaging level configured for that
    system in isocost_config.SYSTEM_N_RUNS. Their raw per-run <H(t)> samples
    are saved along with the per-run wall-clock. The plot script derives every
    configured averaging level from those samples.
  * mcsolve: the raw S^2 fit. mcsolve's trajectory average is unbiased, so its
    RMSE is exactly S/sqrt(ntraj); we sample a few small ntraj (with repeats to
    smooth run-to-run noise) and save each repeat's RMSE and the per-trajectory
    time. The plot script solves ntraj* = (S/target)^2 for any target.

The sweep stops early only once the RMSE evaluated at the FEWEST configured
run count (the harshest level, largest statistical floor) falls below
SWEEP_STOP_RMSE. The configured levels are recorded in the metadata, and the
plot script rejects data that cannot support them.

Uses the fine 80-point time grid (TLIST_FINE), matching the published Result 4
and the other accuracy-style comparisons (Results 1 and 3).

Writes, per system:  data/isocost_vs_dim_<system>.json
Run:                 python run_isocost_vs_dim.py (--system ... | --all)
                     [--sizes ...]
                     (this is the slow benchmark)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    observable_set,
    build_davies_operators,
    build_spin_chain, build_oscillator_bath, build_mixed_field_chain,
    TLIST_FINE, SUBSTEPS,
    DATA_DIR, FULL_TIME_BUDGET, MAX_FULL_DIM, MC_OPTIONS, tavg_rmse,
    run_metadata, save_data,
)
from benchmark_cli import (
    add_max_full_dim_argument, add_safety_arguments, preflight_run,
    selected_systems,
)
from isocost_config import run_counts
from qutip_bundling import mesolve_ensemble
from qutip_bundling.native_solver import rk4_mesolve

SWEEP_STOP_RMSE = 0.01  # stop once the smallest configured averaging level
                        # reaches this floor (recorded in metadata; must stay
                        # <= any plot-side target)
M_GRID = [1, 2, 4, 8, 16, 32, 64, 128]          # SLB knob searched
MC_FIT_GRID = [100, 200, 400]                   # mcsolve ntraj sampled to fit S
MC_REPEATS = 4                                  # repeats per point -> smooth noise
RNG_SWEEP = 100         # seed for the SLB estimates (matches prior runs)
ROUND = 8               # decimals kept for saved curves (keeps the JSON compact)

# Size-points: (size, substeps). R4's x-axis IS dimension, so every size lives
# in ONE file/figure -- there is no per-dim split here. "Sweep everything" means
# extending the list and continuing PAST the mesolve wall via the certified
# native reference instead of stopping there.
NATIVE_REF_SUBSTEPS_FACTOR = 2
MC_TIME_BUDGET_S = 3600.0   # per-dimension mcsolve budget (see mc_fit)

SYSTEMS = {
    # Extended to dim 512. This result IS the cost-versus-dimension claim, and
    # it spanned 4-64 while Results 2 and 3 spanned 4-512. Cheap here because
    # mcsolve on this system stays trivial with size -- 0.21 s/trajectory at
    # dim 128, 1.89 s at 256 -- and MC_TIME_BUDGET_S caps the ladder anyway.
    "spin_chain":      (build_spin_chain,      [(2, 4), (3, 4), (4, 4),
                                                (5, 4), (6, 4), (7, 4),
                                                (8, 4), (9, 4)]),   # dims 4..512
    # System B: same sizes as the TFIM chain, so the pair differs only by the
    # longitudinal field, exactly as in Results 1 and 3. Its N_L reaches 2,017
    # at dim 64, which is where the iso-accuracy question actually has teeth --
    # without it Result 4 compares only the two systems where N_L is either
    # tiny (TFIM) or ladder-structured (oscillator).
    "mixed_chain":     (build_mixed_field_chain, [(2, 4), (3, 4), (4, 4),
                                                  (5, 4), (6, 4)]), # dims 4..64
    "oscillator_bath": (build_oscillator_bath, [(4, 4), (8, 4), (16, 4),
                                                (32, 16)]),         # dims 8..64
}


def slb_sweep(H, rho0, c_ops, reference, n_l, n_runs_max, sweep_min_runs,
              substeps=SUBSTEPS, ops=None, labels=None):
    """Ascending M sweep at n_runs_max runs each; raw samples saved per M.

    ``reference`` is (n_obs, n_times) and ``samples`` is stored as
    (n_runs_max, n_obs, n_times), so M* can afterwards be solved per observable
    rather than for the energy alone.

    The stop condition is the WORST observable, not the first. Stopping when
    the energy clears the floor would leave the sweep short of the M that a
    harder observable needs, and M* for that observable would then be
    unreachable rather than merely expensive -- which is the failure mode this
    change exists to remove.
    """
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    n_obs = reference.shape[0]
    rows, seen = [], set()
    for m in M_GRID:
        m_eff = min(m, n_l)
        if m_eff in seen:
            continue
        seen.add(m_eff)
        t0 = time.perf_counter()
        ens = mesolve_ensemble(H, rho0, TLIST_FINE, c_ops, M=m_eff, e_ops=ops,
                               n_realizations=n_runs_max, rng=RNG_SWEEP,
                               backend="native", substeps=substeps)
        per_run = (time.perf_counter() - t0) / n_runs_max
        samples = np.real(ens.samples)            # (n_runs, n_obs, n_times)
        rmse_min = np.array([tavg_rmse(samples[:sweep_min_runs, j], reference[j])
                             for j in range(n_obs)])
        if (not np.isfinite(samples).all()
                or float(np.max(np.abs(samples)))
                > 100.0 * (1.0 + float(np.max(np.abs(reference))))):
            print(f"      M={m_eff:4d}  SOFT DIVERGENCE at {substeps} substeps "
                  f"-- sweep ends for this dimension")
            rows.append({"M": m_eff, "diverged": True,
                         "per_run_cost": per_run})
            break
        rows.append({"M": m_eff, "per_run_cost": per_run,
                     "samples": np.round(samples, ROUND)})
        worst = int(np.argmax(rmse_min))
        print(f"      M={m_eff:4d}  worst={labels[worst]} "
              f"{rmse_min[worst]:.3e}  best={labels[int(np.argmin(rmse_min))]} "
              f"{rmse_min.min():.3e}  per-run={per_run:.3g}s")
        if float(rmse_min.max()) <= SWEEP_STOP_RMSE or m_eff >= n_l:
            break
    return rows


def _mcsolve_runs(H, psi0, c_ops, ntraj, ops=None):
    """(n_obs, ntraj, n_times) per-trajectory expectations."""
    ops = [H] if ops is None else ops
    try:
        res = qutip.mcsolve(H, psi0, TLIST_FINE, c_ops, e_ops=ops, ntraj=ntraj,
                            options=MC_OPTIONS)
    except (TypeError, KeyError):
        res = qutip.mcsolve(H, psi0, TLIST_FINE, c_ops, e_ops=ops, ntraj=ntraj,
                            options={"progress_bar": False,
                                     "keep_runs_results": True})
    return np.real(np.array([[res.runs_expect[j][k] for k in range(ntraj)]
                             for j in range(len(ops))]))


def mc_fit(H, psi0, c_ops, reference, ops=None):
    """Raw material for the S^2 fit: per sampled ntraj, each repeat's
    time-averaged RMSE and the per-trajectory wall-clock. mcsolve is unbiased,
    so rmse^2 * ntraj estimates S^2 at any ntraj; the plot script averages
    these and solves ntraj* = (S/target)^2 for its target.

    A per-dimension wall-clock budget (MC_TIME_BUDGET_S) stops the ntraj ladder
    before it can run for hours at large dim. Skipped values are RECORDED, and
    the S^2 fit simply uses the points that did run -- S^2 is a property of the
    system, so fewer sampled ntraj costs precision, not validity."""
    rows, skipped, per_traj_est = [], [], None
    spent = 0.0
    for nt in MC_FIT_GRID:
        if per_traj_est is not None:
            projected = per_traj_est * nt * MC_REPEATS
            if spent + projected > MC_TIME_BUDGET_S:
                print(f"      mcsolve ntraj={nt:4d}  SKIPPED (projected "
                      f"{projected/60:.0f} min would exceed the "
                      f"{MC_TIME_BUDGET_S/60:.0f} min budget)")
                skipped.append({"ntraj": nt, "projected_s": projected})
                continue
        ref2d = np.atleast_2d(np.asarray(reference, dtype=float))
        rs = []                                   # (repeat, observable)
        t0 = time.perf_counter()
        for _ in range(MC_REPEATS):
            runs = _mcsolve_runs(H, psi0, c_ops, nt, ops)   # (n_obs, ntraj, nt)
            rs.append([tavg_rmse(runs[j], ref2d[j])
                       for j in range(ref2d.shape[0])])
        dt = time.perf_counter() - t0
        spent += dt
        per_traj_est = dt / (MC_REPEATS * nt)
        rows.append({"ntraj": nt, "rmse_repeats": rs,
                     "per_traj_time": per_traj_est})
        worst = np.mean(rs, axis=0).max()
        print(f"      mcsolve ntraj={nt:4d}  worst RMSE~{worst:.3e}  "
              f"per-traj={per_traj_est*1e3:.3g}ms")
    if skipped:
        rows.append({"_skipped": skipped})
    return [r for r in rows if "_skipped" not in r], skipped


def run(name, build, size_points, n_runs_list):
    n_runs_max = max(n_runs_list)
    sweep_min_runs = min(n_runs_list)
    points = []
    print(f"[{name}]  run-count levels = {n_runs_list}; generating "
          f"{n_runs_max} samples per M")
    print(f"[{name}]  sweep floor RMSE = {SWEEP_STOP_RMSE} at "
          f"n={sweep_min_runs} runs")
    full_feasible = True
    for s, substeps in size_points:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        c_ops = build_davies_operators(H, X)
        n_l = len(c_ops)

        # Reference: qutip mesolve while feasible, else the certified native
        # full-dissipator RK4. Previously the sweep simply STOPPED at the wall,
        # which is why this curve ended at dim 32; the fallback lets it
        # continue, with a substep-halving self-check so an uncertifiable
        # reference is skipped rather than trusted.
        ref_substeps = NATIVE_REF_SUBSTEPS_FACTOR * substeps
        reference, t_full, ref_method, ref_selfcheck = None, np.nan, None, None
        if full_feasible and dim <= MAX_FULL_DIM:
            try:
                t0 = time.perf_counter()
                res = qutip.mesolve(H, rho0, TLIST_FINE, c_ops=c_ops,
                                    e_ops=[], options={"store_states": True})
                t_full = time.perf_counter() - t0
                states = res.states
                ref_method = "mesolve"
                if t_full > FULL_TIME_BUDGET:
                    full_feasible = False
            except MemoryError:
                reference, full_feasible = None, False
        if reference is None:
            t0 = time.perf_counter()
            hi = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[H],
                             substeps=ref_substeps, store_states=True)
            t_native = time.perf_counter() - t0
            lo = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[H],
                             substeps=ref_substeps // 2)
            states = hi.states
            # The self-check stays on the energy: it certifies the INTEGRATION,
            # and every observable is read off the same certified states.
            dev = float(np.max(np.abs(np.real(hi.expect[0])
                                      - np.real(lo.expect[0]))))
            ok = bool(np.isfinite(dev) and dev <= 1e-4)
            ref_selfcheck = {"substeps": ref_substeps, "max_abs_dev": dev,
                             "passed": ok}
            ref_method = f"native_rk4_substeps{ref_substeps}"
            print(f"  dim={dim:4d}  reference via {ref_method} in {t_native:.0f}s; "
                  f"self-check dev {dev:.2e} [{'OK' if ok else 'FAILED'}]")
            if not ok:
                print(f"  dim={dim:4d}  reference uncertifiable -- skipping size.")
                continue

        # Observables are fixed here, from the certified reference, exactly as
        # Result 3 fixes them -- so the two sections score the same quantities
        # and their numbers can be compared.
        labels, ops, coherence = observable_set(name, H, states)
        reference = np.array([np.real(qutip.expect(op, states)) for op in ops])
        print(f"  dim={dim:4d}  N_L={n_l:4d}  reference={ref_method}  "
              f"{substeps} substeps  observables={labels}")
        mc_rows, mc_skipped = mc_fit(H, psi0, c_ops, reference, ops)
        points.append({
            "size": s, "dim": dim, "n_l": n_l, "t_full": t_full,
            "substeps": substeps, "reference_method": ref_method,
            "reference_selfcheck": ref_selfcheck,
            "observables": labels, "coherence": coherence,
            "reference": np.round(reference, ROUND),
            "slb_sweep": slb_sweep(
                H, rho0, c_ops, reference, n_l, n_runs_max,
                sweep_min_runs, substeps, ops=ops, labels=labels,
            ),
            "mc_fit": mc_rows, "mc_skipped": mc_skipped,
        })

    meta = run_metadata(
        tlist=TLIST_FINE, max_full_dim=MAX_FULL_DIM,
        system=name, sizes=[s for s, _ in size_points],
        substeps_by_size={str(s): ss for s, ss in size_points},
        N_RUNS_LEVELS=n_runs_list, N_RUNS_MAX=n_runs_max,
        SWEEP_MIN_RUNS=sweep_min_runs, SWEEP_STOP_RMSE=SWEEP_STOP_RMSE,
        M_GRID=M_GRID, MC_FIT_GRID=MC_FIT_GRID, MC_REPEATS=MC_REPEATS,
        rng_sweep=RNG_SWEEP,
    )
    save_data(f"isocost_vs_dim_{name}.json", meta, compact=True, points=points)


def main():
    global MAX_FULL_DIM
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_safety_arguments(ap, SYSTEMS)
    add_max_full_dim_argument(ap, MAX_FULL_DIM)
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="only these configured model sizes (these are not "
                         "Hilbert dimensions)")
    args = ap.parse_args()
    if args.max_full_dim != MAX_FULL_DIM:
        MAX_FULL_DIM = args.max_full_dim
        print(f"[config] exact-mesolve dimension cap raised to {MAX_FULL_DIM}")
    names = selected_systems(args, SYSTEMS)
    work = []
    plans = []
    for name in names:
        build, size_points = SYSTEMS[name]
        if args.sizes:
            available_sizes = {size for size, _ in size_points}
            missing = sorted(set(args.sizes) - available_sizes)
            if missing:
                ap.error(
                    f"{name} has no configured Result 4 model sizes {missing}; "
                    f"available sizes are {sorted(available_sizes)}"
                )
            keep = set(args.sizes)
            size_points = [(s, ss) for s, ss in size_points if s in keep]
        work.append((name, build, size_points, run_counts(name)))
        plans.append((
            f"Result 4: {name}, model sizes={[s for s, _ in size_points]}",
            DATA_DIR / f"isocost_vs_dim_{name}.json",
        ))

    if not preflight_run(
        plans, overwrite=args.overwrite, dry_run=args.dry_run
    ):
        return
    for item in work:
        run(*item)


if __name__ == "__main__":
    main()
