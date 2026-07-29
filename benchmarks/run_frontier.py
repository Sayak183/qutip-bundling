"""
run_frontier.py
===============

DATA-GENERATION HALF of the accuracy-vs-cost frontier benchmark (Result 3:
SLB vs qutip.mcsolve at a fixed size). All the compute, none of the plotting;
the figure is drawn from the saved data by plot_frontier.py.

For each system (at one fixed, reference-feasible size) it records:

  * the exact reference <H(t)>;
  * the substeps guard: the SLB bias at SUBSTEPS and at 2x SUBSTEPS, verifying
    the error floor is the genuine O(1/M) bundling bias and not an unconverged
    timestep. The verdict is saved with the data, so every frontier figure
    carries its own integrator-convergence evidence.
  * SLB: for each bundle size M, N_RUNS_MAX independent runs drawn once, raw
    per-run <H(t)> samples saved with the per-run wall-clock. The averaging
    levels shown in the figure (N_RUNS_SWEEP) subsample these at analysis
    time.
  * mcsolve: for each ntraj, the wall-clock and the derived time-averaged
    BIAS / SEM / RMSE of that estimate (raw per-trajectory curves at
    ntraj=1000 would be megabytes for no analysis-time gain, so the derived
    stats are stored).

Scope
-----
The command requires an explicit --system or --all scope. Use --dims to select
a subset. The dimension-64 configurations are heavy workstation runs; the
baselines (full reference and mcsolve) are the expensive parts. If the
reference raises MemoryError, the RMSE cannot be measured at that size; use the
cost-scaling benchmark for the large-N cost argument beyond the reference wall.

Uses the fine 80-point time grid (TLIST_FINE), like the other accuracy-style
comparisons.

Writes, per system and dimension:  data/frontier_<system>_dim<D>.json
Run:                 python run_frontier.py (--system ... | --all) [--dims ...]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    build_davies_operators,
    build_spin_chain, build_oscillator_bath, TLIST_FINE, SUBSTEPS,
    DATA_DIR, MC_OPTIONS, tavg_bias_sem_rmse, run_metadata, save_data,
)
from benchmark_cli import add_safety_arguments, preflight_run, selected_systems
from qutip_bundling import mesolve_ensemble
from qutip_bundling.native_solver import rk4_mesolve

SUBSTEPS_TOL = 0.05      # warn if doubling substeps moves the bias > 5%
SUBSTEPS_PROBE_M = 16    # M used for the substeps convergence guard
RNG_SWEEP = 1000         # seed for the SLB estimates (matches prior runs)
ROUND = 8                # decimals kept for saved curves

#           name -> (builder, size, M grid, ntraj grid, runs drawn per M)
# Each system sweeps a LIST of size-points:
#   (size, m_grid, ntraj_grid, n_runs_max, substeps)
# run_frontier computes every point ONCE, saving one file per dim
# (frontier_<system>_dim<D>.json); plot_frontier picks a dim via PLOT_DIM.
NATIVE_REF_SUBSTEPS_FACTOR = 2
MAX_FULL_DIM_FALLBACK = 64

# mcsolve cap: at large dim a single ntraj value can take hours. After the
# first ntraj we measure the per-trajectory cost and SKIP any larger ntraj whose
# projected wall-clock exceeds this budget, recording it as skipped rather than
# grinding for days. Capped points are honest omissions, not silent failures.
MC_TIME_BUDGET_S = 3600.0

SYSTEMS = {
    "spin_chain": (build_spin_chain, [
        (4, [1, 2, 4, 8, 16, 32], [10, 50, 200, 1000], 32, 4),   # dim 16
        (5, [1, 2, 4, 8, 16, 32], [10, 50, 200, 1000], 32, 4),   # dim 32
        (6, [2, 4, 8, 16, 32, 64], [10, 50, 200],      16, 4),   # dim 64
    ]),
    "oscillator_bath": (build_oscillator_bath, [
        (8,  [1, 2, 4, 8, 16, 32], [10, 50, 200, 1000], 32, 4),  # dim 16
        (16, [1, 2, 4, 8, 16, 32], [10, 50, 200, 1000], 32, 4),  # dim 32
        (32, [2, 4, 8, 16, 32, 64], [10, 50, 200],      16, 16), # dim 64 (stiff)
    ]),
}


def capped_unique_m_values(m_grid, n_lindblad):
    values = []
    for m in m_grid:
        m_eff = min(int(m), n_lindblad)
        if m_eff > 0 and m_eff not in values:
            values.append(m_eff)
    return values


def slb_samples(H, rho0, c_ops, m_eff, n_runs, substeps):
    """Return (samples (n_runs, n_times), per_run_seconds) for one M."""
    t0 = time.perf_counter()
    ens = mesolve_ensemble(H, rho0, TLIST_FINE, c_ops, M=m_eff, e_ops=[H],
                           n_realizations=n_runs, rng=RNG_SWEEP,
                           backend="native", substeps=substeps)
    per_run = (time.perf_counter() - t0) / n_runs
    return np.real(ens.samples[:, 0, :]), per_run


def substeps_guard(H, rho0, c_ops, reference, m_eff, n_runs):
    """SLB bias at SUBSTEPS vs 2x SUBSTEPS; flat => the floor is bundling bias,
    not the timestep. Returns the record saved with the data."""
    s_lo, s_hi = SUBSTEPS, 2 * SUBSTEPS
    samp_lo, _ = slb_samples(H, rho0, c_ops, m_eff, n_runs, s_lo)
    samp_hi, _ = slb_samples(H, rho0, c_ops, m_eff, n_runs, s_hi)
    bias_lo = tavg_bias_sem_rmse(samp_lo, reference)[0]
    bias_hi = tavg_bias_sem_rmse(samp_hi, reference)[0]
    rel = abs(bias_hi - bias_lo) / max(bias_lo, 1e-12)
    ok = rel <= SUBSTEPS_TOL
    flag = ("OK (floor is bundling bias, not timestep)" if ok
            else f"WARNING: bias moved {rel:.0%} -- RAISE SUBSTEPS")
    print(f"  substeps guard (M={m_eff}): bias {bias_lo:.3e} -> {bias_hi:.3e} "
          f"on {s_lo}->{s_hi} substeps  [{flag}]")
    return {"M": m_eff, "substeps": [s_lo, s_hi],
            "bias": [bias_lo, bias_hi], "rel_change": rel, "tol": SUBSTEPS_TOL,
            "ok": ok}


def _mcsolve_samples(H, psi0, c_ops, ntraj):
    """One mcsolve call; return per-trajectory <H> array (ntraj, n_times)."""
    try:
        res = qutip.mcsolve(H, psi0, TLIST_FINE, c_ops, e_ops=[H], ntraj=ntraj,
                            options=MC_OPTIONS)
    except (TypeError, KeyError):
        res = qutip.mcsolve(H, psi0, TLIST_FINE, c_ops, e_ops=[H], ntraj=ntraj,
                            options={"progress_bar": False,
                                     "keep_runs_results": True})
    return np.real(np.array([res.runs_expect[0][k] for k in range(ntraj)]))


def run(name, build, size, m_grid, ntraj_grid, n_runs_max, substeps):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    c_ops = build_davies_operators(H, X)
    n_l = len(c_ops)
    dim = H.shape[0]

    print(f"\n[{name}] dim={dim}, original Lindblad operators N_L={n_l}")
    # reference: mesolve while feasible, else certified native full-dissipator
    ref_substeps = NATIVE_REF_SUBSTEPS_FACTOR * substeps
    ref_method, ref_selfcheck, reference = None, None, None
    if dim <= MAX_FULL_DIM_FALLBACK:
        try:
            reference = np.real(qutip.mesolve(H, rho0, TLIST_FINE, c_ops=c_ops,
                                              e_ops=[H]).expect[0])
            ref_method = "mesolve"
        except MemoryError:
            reference = None
    if reference is None:
        hi = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[H],
                         substeps=ref_substeps)
        lo = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[H],
                         substeps=ref_substeps // 2)
        reference = np.real(hi.expect[0])
        dev = float(np.max(np.abs(reference - np.real(lo.expect[0]))))
        ref_selfcheck = {"substeps": ref_substeps, "max_abs_dev": dev,
                         "passed": bool(np.isfinite(dev) and dev <= 1e-4)}
        ref_method = f"native_rk4_substeps{ref_substeps}"
        print(f"  reference via {ref_method}; self-check dev {dev:.2e} "
              f"[{'OK' if ref_selfcheck['passed'] else 'FAILED'}]")
        if not ref_selfcheck["passed"]:
            print(f"  dim={dim}: reference uncertifiable -- skipping this size.")
            return

    m_values = capped_unique_m_values(m_grid, n_l)
    guard = substeps_guard(H, rho0, c_ops, reference,
                           min(SUBSTEPS_PROBE_M, n_l), n_runs_max)

    # ---- SLB: draw n_runs_max runs per M; raw samples saved ----
    slb_sweep = []
    print("  bundling (sweep M):")
    for m_eff in m_values:
        samples, per_run = slb_samples(H, rho0, c_ops, m_eff, n_runs_max,
                                       substeps)
        _, _, rmse = tavg_bias_sem_rmse(samples, reference)
        slb_sweep.append({"M": m_eff, "per_run_cost": per_run,
                          "samples": np.round(samples, ROUND)})
        print(f"    M={m_eff:3d}  one-run={per_run*1000:6.1f}ms  "
              f"rmse(n={n_runs_max})={rmse:.2e}")

    # ---- mcsolve: one call per ntraj; derived stats saved ----
    mc = []
    mc_skipped = []
    per_traj_est = None
    print("  mcsolve (sweep ntraj):")
    for nt in ntraj_grid:
        # cap: once we know the per-trajectory cost, refuse ntraj values whose
        # projected time blows the budget (recorded, not silently dropped)
        if per_traj_est is not None:
            projected = per_traj_est * nt
            if projected > MC_TIME_BUDGET_S:
                print(f"    ntraj={nt:5d}  SKIPPED (projected {projected/60:.0f} "
                      f"min > {MC_TIME_BUDGET_S/60:.0f} min budget)")
                mc_skipped.append({"ntraj": nt, "projected_s": projected})
                continue
        t0 = time.perf_counter()
        runs = _mcsolve_samples(H, psi0, c_ops, nt)
        dt = time.perf_counter() - t0
        per_traj_est = dt / nt
        bias, sem, rmse = tavg_bias_sem_rmse(runs, reference)
        mc.append({"ntraj": nt, "cost": dt,
                   "bias": bias, "sem": sem, "rmse": rmse})
        print(f"    ntraj={nt:5d}  time={dt:7.3f}s  rmse={rmse:.3e} "
              f"(bias {bias:.2e}, sem {sem:.2e})")

    meta = run_metadata(
        tlist=TLIST_FINE,
        system=name, size=size, M_GRID=m_grid, NTRAJ_GRID=ntraj_grid,
        N_RUNS_MAX=n_runs_max, SUBSTEPS_TOL=SUBSTEPS_TOL,
        SUBSTEPS_PROBE_M=SUBSTEPS_PROBE_M, rng_sweep=RNG_SWEEP,
    )
    save_data(f"frontier_{name}_dim{dim}.json", meta, compact=True,
              dim=dim, n_l=n_l, substeps=substeps,
              reference_method=ref_method, reference_selfcheck=ref_selfcheck,
              reference=np.round(reference, ROUND),
              guard=guard, slb_sweep=slb_sweep, mc=mc,
              mc_skipped=mc_skipped)
    print(f"  -> wrote frontier_{name}_dim{dim}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_safety_arguments(ap, SYSTEMS)
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="only these Hilbert dims (default: every configured "
                         "size; each saved to its own file)")
    args = ap.parse_args()
    names = selected_systems(args, SYSTEMS)
    work = []
    plans = []
    for name in names:
        build, points = SYSTEMS[name]
        available_dims = set()
        for size, m_grid, ntraj_grid, n_runs_max, substeps in points:
            probe_dim = build(size)[0].shape[0]
            available_dims.add(probe_dim)
            if args.dims and probe_dim not in args.dims:
                continue
            work.append((
                name, build, size, m_grid, ntraj_grid, n_runs_max, substeps,
            ))
            plans.append((
                f"Result 3: {name}, dim {probe_dim}, M={m_grid}, "
                f"ntraj={ntraj_grid}",
                DATA_DIR / f"frontier_{name}_dim{probe_dim}.json",
            ))
        if args.dims:
            missing = sorted(set(args.dims) - available_dims)
            if missing:
                ap.error(
                    f"{name} has no configured Result 3 dimensions {missing}; "
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
