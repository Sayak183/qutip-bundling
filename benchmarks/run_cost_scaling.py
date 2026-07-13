"""
run_cost_scaling.py
===================

UPDATED: Cost scaling benchmark with Davies construction time included
and an easily accessible TARGET_RMSE switch. Now uses online ensemble
accumulation to prevent RAM exhaustion at large Hilbert dimensions.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST, SUBSTEPS,
    FULL_TIME_BUDGET, MAX_FULL_DIM, run_metadata, save_data,
)
from qutip_bundling import davies_operators, mesolve_ensemble

M_REP = 8               # representative fixed bundle size
N_ACC = 16              # SLB runs averaged to measure accuracy
M_SWEEP_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
SWEEP_STOP_RMSE = 0.01  
RNG_TIMING = 0          # seed for single-realization timing

SYSTEMS = {
    "spin_chain":      (build_spin_chain,      [2, 3, 4, 5, 6, 7]),
    "oscillator_bath": (build_oscillator_bath, [4, 8, 16, 32, 64]),
}

def slb_estimate_online(H, rho0, c_ops, m, n_runs, batch_size=4):
    """Accumulates mean and variance online to minimize RAM footprint."""
    t0 = time.perf_counter()
    mean_acc = None
    m2_acc = None
    count = 0
    
    # Process in batches to keep memory usage at O(n_times)
    for _ in range(n_runs // batch_size):
        ens = mesolve_ensemble(H, rho0, TLIST, c_ops, M=m, e_ops=[H],
                               n_realizations=batch_size, rng=None,
                               backend="native", substeps=SUBSTEPS)
        batch_data = np.real(ens.samples[:, 0, :])
        
        for i in range(batch_size):
            count += 1
            sample = batch_data[i]
            if mean_acc is None:
                mean_acc = np.copy(sample)
                m2_acc = np.zeros_like(sample)
            else:
                delta = sample - mean_acc
                mean_acc += delta / count
                delta2 = sample - mean_acc
                m2_acc += delta * delta2
                
    total_time = time.perf_counter() - t0
    variance = m2_acc / (count - 1) if count > 1 else np.zeros_like(mean_acc)
    return mean_acc, variance, total_time

def sweep_m(H, rho0, c_ops, reference, n_l):
    """Ascending M sweep using memory-efficient online accumulation."""
    rows, seen = [], set()
    for m in M_SWEEP_GRID:
        m_eff = min(m, n_l)
        if m_eff in seen: continue
        seen.add(m_eff)
        
        mean, var, dt = slb_estimate_online(H, rho0, c_ops, m_eff, N_ACC)
        
        # Calculate RMSE components directly from variance
        sem = np.sqrt(var / N_ACC)
        bias = np.abs(mean - np.asarray(reference, dtype=float))
        rmse_vec = np.sqrt(bias**2 + sem**2)
        rmse = float(np.mean(rmse_vec))
        rmse_std = float(np.mean(np.sqrt(var))) # Fluctuation estimate
        
        rows.append({"M": m_eff, "rmse": rmse, "rmse_std": rmse_std, "cost": dt})
        print(f"      M={m_eff:4d}  RMSE={rmse:.3e}  cost={dt:.3g}s")
        if rmse <= SWEEP_STOP_RMSE or m_eff >= n_l: break
    return rows

def run(name, build, sizes):
    points, full_feasible, wall_dim = [], True, None
    print(f"[{name}]")
    for s in sizes:
        H, X, psi0 = build(s)
        rho0 = qutip.ket2dm(psi0)
        dim = H.shape[0]
        t0 = time.perf_counter()
        c_ops = davies_operators(H, X, gamma)
        t_davies = time.perf_counter() - t0
        n_l = len(c_ops)

        reference, t_full = None, np.nan
        if full_feasible and dim <= MAX_FULL_DIM:
            try:
                t0 = time.perf_counter()
                res = qutip.mesolve(H, rho0, TLIST, c_ops=c_ops, e_ops=[H])
                t_full = time.perf_counter() - t0
                reference = np.real(res.expect[0])
                if t_full > FULL_TIME_BUDGET: full_feasible, wall_dim = False, dim
            except MemoryError:
                t_full, full_feasible, wall_dim = np.nan, False, dim
        elif wall_dim is None: wall_dim = dim

        m_rep = min(M_REP, n_l)
        t0 = time.perf_counter()
        mesolve_ensemble(H, rho0, TLIST, c_ops, M=m_rep, e_ops=[H],
                         n_realizations=1, rng=RNG_TIMING, backend="native",
                         substeps=SUBSTEPS)
        t_slb_fixed = time.perf_counter() - t0

        m_sweep = sweep_m(H, rho0, c_ops, reference, n_l) if reference is not None else None

        points.append({
            "size": s, "dim": dim, "n_l": n_l, "t_davies": t_davies,
            "t_full": t_full, "t_slb_fixed": t_slb_fixed,
            "reference": reference, "m_sweep": m_sweep,
        })

    meta = run_metadata(system=name, sizes=sizes, M_REP=M_REP, N_ACC=N_ACC)
    save_data(f"cost_scaling_{name}.json", meta, wall_dim=wall_dim, points=points)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all", choices=[*SYSTEMS, "all"])
    args = ap.parse_args()
    names = list(SYSTEMS) if args.system == "all" else [args.system]
    for name in names:
        build, sizes = SYSTEMS[name]
        run(name, build, sizes)

if __name__ == "__main__":
    main()