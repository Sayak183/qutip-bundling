"""
run_cost_scaling.py
===================

DATA-GENERATION HALF of the cost-scaling benchmark (Result 2). All the compute,
none of the plotting; the figure is drawn from the saved data by
plot_cost_scaling.py.

At each Hilbert dimension it measures:

  * t_full        -- wall-clock of one full mesolve with all N_L collapse
                     operators (the exact reference; NaN past the feasibility
                     wall).
  * t_slb_fixed   -- wall-clock of ONE SLB realization at the representative
                     fixed bundle size M_REP (the raw per-solve cost curve).
  * m_sweep       -- for each M on an ascending grid: the time-averaged RMSE of
                     an N_ACC-run SLB estimate against the exact reference, the
                     jackknife uncertainty of that RMSE, and the wall-clock of
                     the whole N_ACC-run estimate.

The sweep is the point of the split: the accuracy target that defines the
iso-accuracy curve (M*, its cost, its achieved RMSE) is NOT chosen here.
plot_cost_scaling.py derives all of that from the sweep at analysis time, so
the target can be changed without re-running this (slow) script. The sweep
stops early only once the RMSE falls below SWEEP_STOP_RMSE, a floor recorded in
the metadata and set well below any plausible target so the plot side always
has the data it needs.

Writes, per system:  data/cost_scaling_<system>.json
Run:                 python run_cost_scaling.py [--system spin_chain|oscillator_bath|all]
                     (this is the slow benchmark)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST, SUBSTEPS,
    FULL_TIME_BUDGET, MAX_FULL_DIM, tavg_rmse_jackknife, run_metadata, save_data,
)
from qutip_bundling import davies_operators, mesolve_ensemble

M_REP = 8               # representative fixed bundle size (matches other figures)
N_ACC = 16              # SLB runs averaged to measure accuracy at each size
M_SWEEP_GRID = [1, 2, 4, 8, 16, 32, 64, 128]  # ascending M grid for the sweep
SWEEP_STOP_RMSE = 0.01  # stop the sweep once RMSE falls below this floor
                        # (recorded in meta; must stay <= any plot-side target)
RNG_SWEEP = 100         # seed for the N_ACC-run estimates (matches prior runs)
RNG_TIMING = 0          # seed for the single-realization timing solve

# Sizes extend well past the exact-solver wall: the SLB and construction cost
# curves need no reference, so they keep going where the accuracy sweep stops,
# giving the slope fits a decent lever arm. (The dim-128 oscillator builds
# thousands of Lindblad operators -- expect the construction itself to need
# 1-2 GB and noticeable time; that cost is exactly what the new curve shows.)
SYSTEMS = {
    "spin_chain":      (build_spin_chain,      [2, 3, 4, 5, 6, 7]),  # dims 4..128
    "oscillator_bath": (build_oscillator_bath, [4, 8, 16, 32, 64]),  # dims 8..128
}


def slb_estimate(H, rho0, c_ops, m, n_runs):
    """(per-run <H(t)> samples matrix, wall-clock for the n_runs estimate)."""
    t0 = time.perf_counter()
    ens = mesolve_ensemble(H, rho0, TLIST, c_ops, M=m, e_ops=[H],
                           n_realizations=n_runs, rng=RNG_SWEEP,
                           backend="native", substeps=SUBSTEPS)
    return np.real(ens.samples[:, 0, :]), time.perf_counter() - t0


def sweep_m(H, rho0, c_ops, reference, n_l):
    """Ascending M sweep: [{M, rmse, rmse_std, cost}, ...].

    Stops once RMSE <= SWEEP_STOP_RMSE or M reaches n_l (bundles cannot exceed
    the number of physical operators; grid values are capped and deduplicated).
    """
    rows, seen = [], set()
    for m in M_SWEEP_GRID:
        m_eff = min(m, n_l)
        if m_eff in seen:
            continue
        seen.add(m_eff)
        samples, dt = slb_estimate(H, rho0, c_ops, m_eff, N_ACC)
        rmse, rmse_std = tavg_rmse_jackknife(samples, reference)
        rows.append({"M": m_eff, "rmse": rmse, "rmse_std": rmse_std, "cost": dt})
        print(f"      M={m_eff:4d}  RMSE={rmse:.3e} (+/-{rmse_std:.1e})  "
              f"cost={dt:.3g}s")
        if rmse <= SWEEP_STOP_RMSE or m_eff >= n_l:
            break
    return rows


def run(name, build, sizes):
    points, full_feasible, wall_dim = [], True, None
    print(f"[{name}]  sweep floor RMSE = {SWEEP_STOP_RMSE}")
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        # construction, timed on its own: building the N_L Davies/Lindblad
        # operators is a different cost from the dynamics, with its own
        # scaling; the plot shows it as its own curve.
        t0 = time.perf_counter()
        c_ops = davies_operators(H, X, gamma)
        t_davies = time.perf_counter() - t0
        n_l = len(c_ops)

        # exact reference + its cost (until the feasibility wall)
        reference, t_full = None, np.nan
        if full_feasible and dim <= MAX_FULL_DIM:
            try:
                t0 = time.perf_counter()
                res = qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H])
                t_full = time.perf_counter() - t0
                reference = np.real(res.expect[0])
                if t_full > FULL_TIME_BUDGET:
                    full_feasible, wall_dim = False, dim
            except MemoryError:
                t_full = np.nan
                full_feasible, wall_dim = False, dim
        elif wall_dim is None:
            wall_dim = dim

        # fixed-M per-solve cost: one realization
        m_rep = min(M_REP, n_l)
        t0 = time.perf_counter()
        mesolve_ensemble(H, rho0, TLIST, c_ops, M=m_rep, e_ops=[H],
                         n_realizations=1, rng=RNG_TIMING, backend="native",
                         substeps=SUBSTEPS)
        t_slb_fixed = time.perf_counter() - t0

        ff = "%.3g s" % t_full if np.isfinite(t_full) else "(wall)"
        print(f"  dim={dim:4d}  N_L={n_l:4d}  davies={t_davies:.3g}s  full={ff}  "
              f"SLB one solve (M={m_rep})={t_slb_fixed:.3g}s")

        # accuracy sweep, where a reference exists
        m_sweep = (sweep_m(H, rho0, c_ops, reference, n_l)
                   if reference is not None else None)

        points.append({
            "size": s, "dim": dim, "n_l": n_l, "t_davies": t_davies,
            "t_full": t_full, "t_slb_fixed": t_slb_fixed,
            "reference": reference,
            "m_sweep": m_sweep,
        })

    meta = run_metadata(
        system=name, sizes=sizes, M_REP=M_REP, N_ACC=N_ACC,
        M_SWEEP_GRID=M_SWEEP_GRID, SWEEP_STOP_RMSE=SWEEP_STOP_RMSE,
        rng_sweep=RNG_SWEEP, rng_timing=RNG_TIMING,
    )
    save_data(f"cost_scaling_{name}.json", meta,
              wall_dim=wall_dim, points=points)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--system", default="all",
                    choices=[*SYSTEMS, "all"],
                    help="which system to run (default: all)")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="override the size list (smoke tests); the override "
                         "is recorded in the data file's metadata")
    args = ap.parse_args()
    names = list(SYSTEMS) if args.system == "all" else [args.system]
    for name in names:
        build, sizes = SYSTEMS[name]
        run(name, build, args.sizes if args.sizes else sizes)


if __name__ == "__main__":
    main()
