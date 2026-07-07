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

Presets
-------
  default : the published Result 3 -- both systems at dim 16.
  big     : the heavy workstation variant (spin chain 6 sites -> dim 64 with
            N_L ~ 866; oscillator n_fock=16 -> dim 32), trimmed sweeps.
            Replaces the old benchmark_vs_mcsolve_big.py wrapper. The heavy
            parts are the baselines (full-mesolve reference, mcsolve): >= 16 GB
            RAM recommended. Distinct data/figure names keep the dim-16
            results intact. If the reference raises MemoryError, that failure
            IS the motivation for bundling -- but without a reference the RMSE
            cannot be measured at that size; use the cost-scaling benchmark
            (which needs no reference beyond the wall) for the large-N cost
            argument.

Uses the fine 80-point time grid (TLIST_FINE), like the other accuracy-style
comparisons.

Writes, per system:  data/frontier_<system>.json
Run:                 python run_frontier.py [--preset default|big] [--system ...]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST_FINE, SUBSTEPS,
    MC_OPTIONS, tavg_bias_sem_rmse, run_metadata, save_data,
)
from qutip_bundling import davies_operators, mesolve_ensemble

SUBSTEPS_TOL = 0.05      # warn if doubling substeps moves the bias > 5%
SUBSTEPS_PROBE_M = 16    # M used for the substeps convergence guard
RNG_SWEEP = 1000         # seed for the SLB estimates (matches prior runs)
ROUND = 8                # decimals kept for saved curves

#           name -> (builder, size, M grid, ntraj grid, runs drawn per M)
PRESETS = {
    "default": {
        "spin_chain":      (build_spin_chain, 4,
                            [1, 2, 4, 8, 16, 32], [10, 50, 200, 1000], 32),
        "oscillator_bath": (build_oscillator_bath, 8,
                            [1, 2, 4, 8, 16, 32], [10, 50, 200, 1000], 32),
    },
    "big": {
        "spin_chain_6spin":      (build_spin_chain, 6,
                                  [2, 4, 8, 16, 32, 64], [10, 50, 200], 16),
        "oscillator_bath_dim32": (build_oscillator_bath, 16,
                                  [2, 4, 8, 16, 32, 64], [10, 50, 200], 16),
    },
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


def run(name, build, size, m_grid, ntraj_grid, n_runs_max):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    c_ops = davies_operators(H, X, gamma)
    n_l = len(c_ops)
    dim = H.shape[0]

    print(f"\n[{name}] dim={dim}, original Lindblad operators N_L={n_l}")
    reference = np.real(qutip.mesolve(H, rho0, TLIST_FINE, c_ops=c_ops,
                                      e_ops=[H]).expect[0])

    m_values = capped_unique_m_values(m_grid, n_l)
    guard = substeps_guard(H, rho0, c_ops, reference,
                           min(SUBSTEPS_PROBE_M, n_l), n_runs_max)

    # ---- SLB: draw n_runs_max runs per M; raw samples saved ----
    slb_sweep = []
    print("  bundling (sweep M):")
    for m_eff in m_values:
        samples, per_run = slb_samples(H, rho0, c_ops, m_eff, n_runs_max,
                                       SUBSTEPS)
        _, _, rmse = tavg_bias_sem_rmse(samples, reference)
        slb_sweep.append({"M": m_eff, "per_run_cost": per_run,
                          "samples": np.round(samples, ROUND)})
        print(f"    M={m_eff:3d}  one-run={per_run*1000:6.1f}ms  "
              f"rmse(n={n_runs_max})={rmse:.2e}")

    # ---- mcsolve: one call per ntraj; derived stats saved ----
    mc = []
    print("  mcsolve (sweep ntraj):")
    for nt in ntraj_grid:
        t0 = time.perf_counter()
        runs = _mcsolve_samples(H, psi0, c_ops, nt)
        dt = time.perf_counter() - t0
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
    save_data(f"frontier_{name}.json", meta, compact=True,
              dim=dim, n_l=n_l,
              reference=np.round(reference, ROUND),
              guard=guard, slb_sweep=slb_sweep, mc=mc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--preset", default="default", choices=list(PRESETS),
                    help="default: published dim-16 frontier; big: heavy "
                         "workstation variant (>= 16 GB RAM recommended)")
    ap.add_argument("--system", default="all",
                    help="one system name from the preset, or 'all' (default)")
    args = ap.parse_args()
    preset = PRESETS[args.preset]
    if args.preset == "big":
        print("=" * 72)
        print("BIG-SYSTEM FRONTIER  (workstation job -- not the sandbox / CI)")
        print("  the full-mesolve reference and mcsolve baseline are the heavy")
        print("  parts (RAM + minutes to tens of minutes each); SLB stays cheap")
        print("=" * 72)
    names = list(preset) if args.system == "all" else [args.system]
    for name in names:
        if name not in preset:
            raise SystemExit(f"unknown system {name!r} for preset "
                             f"{args.preset!r}; choose from {list(preset)}")
        run(name, *preset[name])


if __name__ == "__main__":
    main()
