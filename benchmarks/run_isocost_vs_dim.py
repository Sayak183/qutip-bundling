"""
run_isocost_vs_dim.py
=====================

DATA-GENERATION HALF of the iso-accuracy cost-vs-dimension benchmark (Result 4:
SLB vs mcsolve as the system grows). All the compute, none of the plotting; the
figure is drawn from the saved data by plot_isocost_vs_dim.py.

At each Hilbert dimension (up to the exact-reference wall) it records:

  * the exact reference <H(t)> and its wall-clock cost;
  * SLB: an ascending bundle-size sweep. For each M, N_RUNS_MAX independent
    runs are drawn ONCE and their raw per-run <H(t)> samples are saved, along
    with the per-run wall-clock. Saving the raw samples (not derived RMSEs) is
    what makes the split powerful here: the plot script subsamples the runs, so
    BOTH the accuracy target AND the averaging levels N are analysis-time
    choices -- neither requires re-running this (slow) script.
  * mcsolve: the raw S^2 fit. mcsolve's trajectory average is unbiased, so its
    RMSE is exactly S/sqrt(ntraj); we sample a few small ntraj (with repeats to
    smooth run-to-run noise) and save each repeat's RMSE and the per-trajectory
    time. The plot script solves ntraj* = (S/target)^2 for any target.

The sweep stops early only once the RMSE evaluated at the FEWEST anticipated
runs (SWEEP_MIN_RUNS, the harshest level, largest statistical floor) falls
below SWEEP_STOP_RMSE; both are recorded in the metadata, and the plot script
warns if asked for a level or target the sweep cannot support.

Uses the fine 80-point time grid (TLIST_FINE), matching the published Result 4
and the other accuracy-style comparisons (Results 1 and 3).

Writes, per system:  data/isocost_vs_dim_<system>.json
Run:                 python run_isocost_vs_dim.py [--system ...] [--sizes ...]
                     (this is the slow benchmark)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST_FINE, SUBSTEPS,
    FULL_TIME_BUDGET, MAX_FULL_DIM, MC_OPTIONS, tavg_rmse, run_metadata,
    save_data,
)
from qutip_bundling import davies_operators, mesolve_ensemble

N_RUNS_MAX = 16         # independent SLB runs drawn per M (levels subsample these)
SWEEP_MIN_RUNS = 4      # harshest averaging level the sweep must support
SWEEP_STOP_RMSE = 0.01  # stop the sweep once the SWEEP_MIN_RUNS-level RMSE is
                        # below this floor (recorded in meta; must stay <= any
                        # plot-side target)
M_GRID = [1, 2, 4, 8, 16, 32, 64, 128]          # SLB knob searched
MC_FIT_GRID = [100, 200, 400]                   # mcsolve ntraj sampled to fit S
MC_REPEATS = 4                                  # repeats per point -> smooth noise
RNG_SWEEP = 100         # seed for the SLB estimates (matches prior runs)
ROUND = 8               # decimals kept for saved curves (keeps the JSON compact)

SYSTEMS = {
    "spin_chain":      (build_spin_chain,      [2, 3, 4, 5]),   # dims 4..32
    "oscillator_bath": (build_oscillator_bath, [4, 8, 16]),     # dims 8..32
}


def slb_sweep(H, rho0, c_ops, reference, n_l):
    """Ascending M sweep at N_RUNS_MAX runs each; raw samples saved per M.

    [{M, per_run_cost, samples: (N_RUNS_MAX, n_times)}, ...]. Stops once the
    SWEEP_MIN_RUNS-level RMSE reaches SWEEP_STOP_RMSE or M reaches n_l (grid
    values are capped at n_l and deduplicated).
    """
    rows, seen = [], set()
    for m in M_GRID:
        m_eff = min(m, n_l)
        if m_eff in seen:
            continue
        seen.add(m_eff)
        t0 = time.perf_counter()
        ens = mesolve_ensemble(H, rho0, TLIST_FINE, c_ops, M=m_eff, e_ops=[H],
                               n_realizations=N_RUNS_MAX, rng=RNG_SWEEP,
                               backend="native", substeps=SUBSTEPS)
        per_run = (time.perf_counter() - t0) / N_RUNS_MAX
        samples = np.real(ens.samples[:, 0, :])
        rmse_min = tavg_rmse(samples[:SWEEP_MIN_RUNS], reference)
        rmse_max = tavg_rmse(samples, reference)
        rows.append({"M": m_eff, "per_run_cost": per_run,
                     "samples": np.round(samples, ROUND)})
        print(f"      M={m_eff:4d}  RMSE(n={SWEEP_MIN_RUNS})={rmse_min:.3e}  "
              f"RMSE(n={N_RUNS_MAX})={rmse_max:.3e}  per-run={per_run:.3g}s")
        if rmse_min <= SWEEP_STOP_RMSE or m_eff >= n_l:
            break
    return rows


def _mcsolve_runs(H, psi0, c_ops, ntraj):
    try:
        res = qutip.mcsolve(H, psi0, TLIST_FINE, c_ops, e_ops=[H], ntraj=ntraj,
                            options=MC_OPTIONS)
    except (TypeError, KeyError):
        res = qutip.mcsolve(H, psi0, TLIST_FINE, c_ops, e_ops=[H], ntraj=ntraj,
                            options={"progress_bar": False,
                                     "keep_runs_results": True})
    return np.real(np.array([res.runs_expect[0][k] for k in range(ntraj)]))


def mc_fit(H, psi0, c_ops, reference):
    """Raw material for the S^2 fit: per sampled ntraj, each repeat's
    time-averaged RMSE and the per-trajectory wall-clock. mcsolve is unbiased,
    so rmse^2 * ntraj estimates S^2 at any ntraj; the plot script averages
    these and solves ntraj* = (S/target)^2 for its target."""
    rows = []
    for nt in MC_FIT_GRID:
        rs = []
        t0 = time.perf_counter()
        for _ in range(MC_REPEATS):
            rs.append(tavg_rmse(_mcsolve_runs(H, psi0, c_ops, nt), reference))
        dt = time.perf_counter() - t0
        rows.append({"ntraj": nt, "rmse_repeats": rs,
                     "per_traj_time": dt / (MC_REPEATS * nt)})
        print(f"      mcsolve ntraj={nt:4d}  RMSE~{np.mean(rs):.3e}  "
              f"per-traj={rows[-1]['per_traj_time']*1e3:.3g}ms")
    return rows


def run(name, build, sizes):
    points = []
    print(f"[{name}]  sweep floor RMSE = {SWEEP_STOP_RMSE} at n={SWEEP_MIN_RUNS} runs")
    full_feasible = True
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        c_ops = davies_operators(H, X, gamma)
        n_l = len(c_ops)

        if not (full_feasible and dim <= MAX_FULL_DIM):
            print(f"  dim={dim:4d}  reference beyond wall — stopping "
                  f"(iso-accuracy needs the exact solve)")
            break
        try:
            t0 = time.perf_counter()
            reference = np.real(qutip.mesolve(H, rho0, TLIST_FINE, c_ops=c_ops,
                                              e_ops=[H]).expect[0])
            t_full = time.perf_counter() - t0
        except MemoryError:
            print(f"  dim={dim:4d}  reference OOM — stopping")
            break
        if t_full > FULL_TIME_BUDGET:
            full_feasible = False

        print(f"  dim={dim:4d}  N_L={n_l:4d}  full={t_full:7.2f}s")
        points.append({
            "size": s, "dim": dim, "n_l": n_l, "t_full": t_full,
            "reference": np.round(reference, ROUND),
            "slb_sweep": slb_sweep(H, rho0, c_ops, reference, n_l),
            "mc_fit": mc_fit(H, psi0, c_ops, reference),
        })

    meta = run_metadata(
        tlist=TLIST_FINE,
        system=name, sizes=sizes, N_RUNS_MAX=N_RUNS_MAX,
        SWEEP_MIN_RUNS=SWEEP_MIN_RUNS, SWEEP_STOP_RMSE=SWEEP_STOP_RMSE,
        M_GRID=M_GRID, MC_FIT_GRID=MC_FIT_GRID, MC_REPEATS=MC_REPEATS,
        rng_sweep=RNG_SWEEP,
    )
    save_data(f"isocost_vs_dim_{name}.json", meta, compact=True, points=points)


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
