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
try:
    from qutip_bundling import SolverInstabilityError
except ImportError:
    from qutip_bundling.native_solver import SolverInstabilityError
from qutip_bundling.native_solver import rk4_mesolve

NATIVE_REF_MAX_DIM = 64   # past the mesolve wall, obtain the exact reference
                          # via the native full-dissipator RK4 instead (all N_L
                          # operators, no superoperators: memory ~ the operator
                          # list instead of qutip's kron blow-up). Used ONLY as
                          # the accuracy reference -- the red cost curve stays
                          # honest qutip-mesolve. Cross-validated against
                          # mesolve at the last size where both exist.
NATIVE_REF_SUBSTEPS = 2 * SUBSTEPS   # reference-grade integration margin


def native_full_reference(H, rho0, c_ops):
    """<H(t)> from the native RK4 propagating the FULL dissipator."""
    res = rk4_mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H],
                      substeps=NATIVE_REF_SUBSTEPS)
    return np.real(res.expect[0])

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
    "spin_chain":      (build_spin_chain,      [2, 3, 4, 5, 6, 7, 8]),  # dims 4..256
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
        # MSE budget of this estimate (same decomposition as Result 4's error
        # budget): observed MSE of the plotted mean, and the statistical part
        # SEM^2; implied bias^2 = MSE - SEM^2 is derived at plot time.
        mean = samples.mean(axis=0)
        mse = float(np.mean((mean - np.asarray(reference)) ** 2))
        sem_sq = float(np.mean(samples.var(axis=0, ddof=1) / samples.shape[0]))
        rows.append({"M": m_eff, "rmse": rmse, "rmse_std": rmse_std,
                     "cost": dt, "mse": mse, "sem_sq": sem_sq})
        print(f"      M={m_eff:4d}  RMSE={rmse:.3e} (+/-{rmse_std:.1e})  "
              f"cost={dt:.3g}s")
        if rmse <= SWEEP_STOP_RMSE or m_eff >= n_l:
            break
    return rows


def run(name, build, sizes, full_budget=FULL_TIME_BUDGET,
        native_ref_max=NATIVE_REF_MAX_DIM):
    points, full_feasible, wall_dim = [], True, None
    stiff_dim = None
    last_mesolve = None        # (dim, H, rho0, c_ops, reference) for validation
    native_validation = None   # recorded once, when the fallback first fires
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
        reference, t_full, ref_method, t_native = None, np.nan, None, None
        if full_feasible and dim <= MAX_FULL_DIM:
            try:
                t0 = time.perf_counter()
                res = qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H])
                t_full = time.perf_counter() - t0
                reference = np.real(res.expect[0])
                ref_method = "mesolve"
                last_mesolve = (dim, H, rho0, c_ops, reference)
                if t_full > full_budget:
                    full_feasible, wall_dim = False, dim
            except MemoryError as err:
                print(f"  dim={dim:4d}  exact mesolve raised MemoryError "
                      f"(superoperator construction from {n_l} operators): {err}")
                t_full = np.nan
                full_feasible, wall_dim = False, dim
        elif wall_dim is None:
            wall_dim = dim

        # past the mesolve wall: full-dissipator reference via the native RK4
        # (reference only -- never plotted as the exact method's cost)
        if reference is None and dim <= native_ref_max:
            if native_validation is None and last_mesolve is not None:
                vd, vH, vrho, vc, vref = last_mesolve
                dev = float(np.max(np.abs(
                    native_full_reference(vH, vrho, vc) - vref)))
                native_validation = {"dim": vd, "max_abs_dev": dev,
                                     "substeps": NATIVE_REF_SUBSTEPS}
                print(f"      native full-dissipator reference validated vs "
                      f"mesolve at dim {vd}: max dev {dev:.2e}")
            t0 = time.perf_counter()
            reference = native_full_reference(H, rho0, c_ops)
            t_native = time.perf_counter() - t0
            ref_method = f"native_rk4_substeps{NATIVE_REF_SUBSTEPS}"
            print(f"  dim={dim:4d}  reference via native full dissipator "
                  f"({n_l} operators, {NATIVE_REF_SUBSTEPS} substeps): "
                  f"{t_native:.3g}s")

        # fixed-M per-solve cost: one realization. A fixed-step RK4 curve is
        # only meaningful at UNIFORM substeps: if the generator becomes too
        # stiff for this step size (the oscillator's anharmonic ladder grows
        # like n^2, so its fastest frequencies eventually exceed the RK4
        # stability limit), the honest move is to END the curve there and say
        # so -- not to silently raise substeps at the last point, which would
        # inflate exactly the costs the slope fit leans on.
        m_rep = min(M_REP, n_l)
        try:
            t0 = time.perf_counter()
            mesolve_ensemble(H, rho0, TLIST, c_ops, M=m_rep, e_ops=[H],
                             n_realizations=1, rng=RNG_TIMING, backend="native",
                             substeps=SUBSTEPS)
            t_slb_fixed = time.perf_counter() - t0
        except SolverInstabilityError as err:
            stiff_dim = dim
            print(f"  dim={dim:4d}  RK4 unstable at {SUBSTEPS} substeps -- "
                  f"curve ends here (uniform-substeps benchmark; the stiff\n"
                  f"            generator would need more substeps, which would no longer "
                  f"be cost-comparable). {err}")
            break

        ff = "%.3g s" % t_full if np.isfinite(t_full) else "(wall)"
        print(f"  dim={dim:4d}  N_L={n_l:4d}  davies={t_davies:.3g}s  full={ff}  "
              f"SLB one solve (M={m_rep})={t_slb_fixed:.3g}s")

        # accuracy sweep, where a reference exists
        m_sweep = (sweep_m(H, rho0, c_ops, reference, n_l)
                   if reference is not None else None)

        points.append({
            "size": s, "dim": dim, "n_l": n_l, "t_davies": t_davies,
            "t_full": t_full, "t_slb_fixed": t_slb_fixed,
            "reference": reference, "reference_method": ref_method,
            "t_native_ref": t_native,
            "m_sweep": m_sweep,
        })

    meta = run_metadata(
        system=name, sizes=sizes, M_REP=M_REP, N_ACC=N_ACC,
        full_budget_used=full_budget,
        M_SWEEP_GRID=M_SWEEP_GRID, SWEEP_STOP_RMSE=SWEEP_STOP_RMSE,
        rng_sweep=RNG_SWEEP, rng_timing=RNG_TIMING,
    )
    save_data(f"cost_scaling_{name}.json", meta,
              wall_dim=wall_dim, stiff_dim=stiff_dim,
              native_ref_validation=native_validation, points=points)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--system", default="all",
                    choices=[*SYSTEMS, "all"],
                    help="which system to run (default: all)")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="override the size list (smoke tests); the override "
                         "is recorded in the data file's metadata")
    ap.add_argument("--native-ref-max", type=int, default=NATIVE_REF_MAX_DIM,
                    help="largest dimension for the native full-dissipator "
                         "reference past the mesolve wall (reference only; "
                         "128 costs ~an hour on the spin chain)")
    ap.add_argument("--full-budget", type=float, default=FULL_TIME_BUDGET,
                    help="raise the exact-solver time budget (seconds) to push "
                         "the reference wall out one size, e.g. 2400 buys the "
                         "spin dim-64 point (~35 min solve) and with it one "
                         "more iso-accuracy measurement")
    args = ap.parse_args()
    names = list(SYSTEMS) if args.system == "all" else [args.system]
    for name in names:
        build, sizes = SYSTEMS[name]
        run(name, build, args.sizes if args.sizes else sizes,
            full_budget=args.full_budget, native_ref_max=args.native_ref_max)


if __name__ == "__main__":
    main()
