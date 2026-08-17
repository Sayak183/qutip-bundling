r"""
run_extreme_dimension.py
========================

Run SLB where nothing else can, and check it against physics rather than
against a reference.

Every other benchmark here stops at the dimension where an exact solve is still
possible, because that is what "measuring the error" requires. System B stops at
dimension 128: its operator list is 32,637 dense matrices at 256, about 32 GB,
which does not fit -- and that is the memory of the operators alone, before any
propagation. So the regime the method exists for has never been shown.

The streaming construction removes that wall. `mesolve_ensemble_davies` builds
each Davies operator, folds it into the bundles, and discards it, so peak memory
is a bounded chunk buffer plus the ensemble's bundles -- a few hundred MB,
independent of N_L.

WHAT THIS CAN AND CANNOT CLAIM
------------------------------
There is no exact solve at this size, so there is no error to report. Saying
"SLB ran at dimension 256" proves only that it terminated. The run is therefore
scored on three things that can be checked WITHOUT a reference, each of which
the method could fail:

  1. Convergence in M. The estimate must stop moving as M grows. Necessary, not
     sufficient -- it could converge to the wrong answer -- which is why it is
     not the only check.
  2. Trace preservation. <I> must stay at 1. The bundled generator is Lindblad
     by construction, so this is a check on the integrator at a size where its
     stability has never been tested.
  3. The thermal state. Detailed balance makes the Gibbs state stationary, so a
     long run must relax to Tr(H rho_Gibbs). This is the strong one, and it is
     FREE: it needs only the eigenvalues of H, which the construction already
     computes. It is also independent of everything bundling does.

System B is used deliberately. System A has a Z2 symmetry that gives it extra
conserved quantities, so it relaxes to a symmetry-restricted stationary state
rather than the global Gibbs state -- measured at dimension 16 it settles at
-3.5054 against a Gibbs value of -3.4738. That gap is real physics, and it would
invalidate check 3. System B has no such symmetry.

Writes:  data/extreme_dimension_mixed_chain_dim<D>.json
Run:     python run_extreme_dimension.py [--size 8] [--m-values 8 16 32]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    DATA_DIR,
    DAVIES_DEGENERACY_TOL,
    KT,
    SUBSTEPS,
    TLIST,
    build_mixed_field_chain,
    gamma,
    run_metadata,
    save_data,
)
from qutip_bundling import davies_operator_count, mesolve_ensemble_davies

# Long enough to relax. The benchmark window (TLIST, t <= 5) is deliberately the
# hard transient and reaches only ~87% of the steady state, which is the right
# choice for measuring solver error and the wrong one for testing a thermal
# limit.
# Same STEP SIZE as TLIST, not the same number of points. The first attempt
# used 60 points over t=0..60, which is an 8x coarser step than the benchmark
# grid, and the fixed-step RK4 diverged on this stiff generator at dimension
# 256 -- a failure caused entirely by the grid choice, not by the physics or by
# bundling. Reaching further in time must cost more steps, not bigger ones.
_STEP = float(TLIST[1] - TLIST[0])
TLIST_THERMAL = np.arange(0.0, 60.0 + _STEP, _STEP)
N_REALIZATIONS = 16
RNG = 0
ROUND = 8


def gibbs_energy(H: qutip.Qobj) -> float:
    """Tr(H rho_Gibbs) from the eigenvalues alone.

    Costs one eigendecomposition, which the Davies construction performs anyway,
    so the reference this run is scored against is free at any dimension SLB can
    reach. No superoperator, no propagation.
    """
    energies = np.real(H.eigenenergies())
    weights = np.exp(-(energies - energies.min()) / KT)
    weights /= weights.sum()
    return float(np.dot(weights, energies))


def run(size: int, m_values: list[int]) -> None:
    H, X, psi0 = build_mixed_field_chain(size)
    rho0 = qutip.ket2dm(psi0)
    dimension = H.shape[0]
    identity = qutip.qeye(H.dims[0])

    t0 = time.perf_counter()
    n_l = davies_operator_count(H, X, gamma,
                                degeneracy_tol=DAVIES_DEGENERACY_TOL)
    t_count = time.perf_counter() - t0

    list_bytes = n_l * dimension * dimension * 16
    print(f"[extreme] System B, dim {dimension}, N_L = {n_l:,}")
    print(f"          counting N_L took {t_count:.1f} s and built no operators")
    print(f"          the operator LIST would need "
          f"{list_bytes / 1024**3:.1f} GB -- this run never forms it")

    e_ops = [H, identity]
    thermal = gibbs_energy(H)
    print(f"          Gibbs <H> at kT={KT} is {thermal:+.4f} "
          f"(from eigenvalues; free)")

    # --- check 1: does the estimate stop moving as M grows? --------------
    sweep = []
    for m in m_values:
        start = time.perf_counter()
        result = mesolve_ensemble_davies(
            H, rho0, TLIST, X, gamma, M=m, e_ops=e_ops,
            n_realizations=N_REALIZATIONS, rng=RNG, backend="native",
            substeps=SUBSTEPS, degeneracy_tol=DAVIES_DEGENERACY_TOL,
        )
        wall = time.perf_counter() - start
        energy = np.asarray(result.expect[0], dtype=float)
        trace = np.asarray(result.expect[1], dtype=float)
        sweep.append({
            "M": m,
            "wall_s": wall,
            "energy": np.round(energy, ROUND),
            "sem": np.round(np.asarray(result.sem[0], dtype=float), ROUND),
            "trace": np.round(trace, ROUND),
            "max_trace_deviation": float(np.max(np.abs(trace - 1.0))),
        })
        print(f"    M={m:3d}  {wall:7.1f} s   <H>(t_end) = {energy[-1]:+.5f}   "
              f"max |Tr-1| = {np.max(np.abs(trace - 1.0)):.2e}", flush=True)

    # --- check 3: the long-time limit, at the largest M ------------------
    m_thermal = max(m_values)
    start = time.perf_counter()
    long_run = mesolve_ensemble_davies(
        H, rho0, TLIST_THERMAL, X, gamma, M=m_thermal, e_ops=e_ops,
        n_realizations=max(4, N_REALIZATIONS // 4), rng=RNG + 1,
        backend="native", substeps=SUBSTEPS,
        degeneracy_tol=DAVIES_DEGENERACY_TOL,
    )
    t_thermal = time.perf_counter() - start
    relaxed = np.asarray(long_run.expect[0], dtype=float)
    start_energy = float(relaxed[0])
    final_energy = float(relaxed[-1])
    travelled = abs(final_energy - start_energy)
    remaining = abs(final_energy - thermal)
    fraction = travelled / max(abs(thermal - start_energy), 1e-30)

    print(f"\n  thermal check at M={m_thermal}, t -> {TLIST_THERMAL[-1]:.0f} "
          f"({t_thermal:.1f} s)")
    print(f"    <H> ran {start_energy:+.4f} -> {final_energy:+.4f}")
    print(f"    Gibbs                      {thermal:+.4f}")
    print(f"    covered {100 * fraction:.1f}% of the distance; "
          f"{remaining:.4f} remains")

    meta = run_metadata(
        tlist=TLIST, substeps=SUBSTEPS, system="mixed_chain", size=size,
        M_VALUES=m_values, N_REALIZATIONS=N_REALIZATIONS, rng=RNG,
        route="streaming (mesolve_ensemble_davies)",
        tlist_thermal=[float(TLIST_THERMAL[0]), float(TLIST_THERMAL[-1]),
                       int(TLIST_THERMAL.size)],
    )
    save_data(
        f"extreme_dimension_mixed_chain_dim{dimension}.json", meta, compact=True,
        dim=dimension, n_l=n_l, t_count=t_count,
        operator_list_bytes=list_bytes,
        gibbs_energy=thermal, sweep=sweep,
        thermal={
            "M": m_thermal, "wall_s": t_thermal,
            "times": np.round(TLIST_THERMAL, 6),
            "energy": np.round(relaxed, ROUND),
            "start": start_energy, "final": final_energy,
            "fraction_covered": fraction, "remaining": remaining,
        },
    )
    print(f"\n  -> wrote extreme_dimension_mixed_chain_dim{dimension}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--size", type=int, default=8,
                        help="spins; 8 gives dimension 256 (default)")
    parser.add_argument("--m-values", type=int, nargs="+", default=[8, 16, 32],
                        help="bundle sizes, ascending (default: 8 16 32)")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run(args.size, sorted(args.m_values))


if __name__ == "__main__":
    main()
