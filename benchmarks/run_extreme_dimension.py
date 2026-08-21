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
  3. The thermal state. A long run must relax to the generator's stationary
     state, and that target is FREE for every observable: in a Boltzmann-
     weighted state Tr(A rho) = sum_e p_e <e|A|e>, needing only the
     eigendecomposition the construction already performs. It is also
     independent of everything bundling does.

CHECK 3 IS NOT SIMPLY "COMPARE TO GIBBS"
----------------------------------------
Detailed balance holds -- applying the generator to the Gibbs state gives zero to
machine precision on all three systems -- but STATIONARY IS NOT UNIQUE. A Davies
operator is Pi_e X Pi_e', so levels e and e' are dynamically connected exactly
when <e|X|e'> is non-zero; where that graph is disconnected the space splits into
sectors, each sector's population is separately conserved, and the limit is Gibbs
WITHIN each sector weighted by where rho0 started. Measured kernel dimensions:

    oscillator   1  at dims 8, 16, 32     unique, so the limit IS Gibbs
    mixed chain  2  at dims 4, 8, 16, 32  two sectors at every size
    spin chain   5  at dim 16

The sector-resolved target is computed here alongside the global one. It is
exact where the global one is not: at dimension 16 it reproduces the unbundled
dynamics to six decimals on the mixed chain and to 2e-5 on the spin chain, where
the global Gibbs value is off by 1.4e-2 and 3.2e-2 respectively.

This matters for what the section can claim. A bundle mixes operators from
DIFFERENT sectors, so at small M the bundled dynamics is more ergodic than the
generator it approximates and drifts toward the global Gibbs state. The artefact
is O(1/M) -- sweeping M to N_L on the mixed chain at dim 16 halves the gap at
every doubling -- but it means agreement with GLOBAL Gibbs is not evidence of
success when the generator has more than one sector.

System A is therefore no longer excluded. Its Z2 symmetry was never the
obstruction; the obstruction was scoring against a target that assumes
ergodicity.

Writes:  data/extreme_dimension_<system>_dim<D>.json
Run:     python run_extreme_dimension.py [--system mixed_chain] [--size 8]
                                         [--m-values 8 16 32] [--substeps 4]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from common import (
    DATA_DIR,
    build_oscillator_bath,
    build_spin_chain,
    observable_set,
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


SYSTEMS = {
    "mixed_chain": build_mixed_field_chain,
    "oscillator_bath": build_oscillator_bath,
    "spin_chain": build_spin_chain,
}


# |<e|X|e'>| below this counts as no coupling. Stable from 1e-10 to 1e-4 on
# every system measured; below 1e-12 eigenvector round-off merges the sectors.
SECTOR_COUPLING_TOL = 1e-10


def thermal_reference(system: str, H: qutip.Qobj, X: qutip.Qobj,
                      rho0: qutip.Qobj):
    """(labels, ops, gibbs values, coherence info) from ONE eigendecomposition.

    The Gibbs state is diagonal in the energy basis, so for any observable
    Tr(A rho_Gibbs) = sum_e p_e <e|A|e>: the same Boltzmann weights against the
    eigenVECTORS instead of only the eigenvalues. No superoperator, no
    propagation. That is what makes the reference free at any dimension the
    method can reach -- and free for EVERY observable, not just the energy.

    The coherence operator is chosen differently here than in Result 3, and the
    difference matters. Result 3 takes the most populated off-diagonal pair FROM
    THE REFERENCE; there is no reference at this dimension, which is the point of
    the section, so this uses the two lowest energy eigenstates. It is a
    well-defined, populated coherence -- but it is NOT the quantity Result 3
    reports, and the two must not be compared.
    """
    energies, states = H.eigenstates()
    energies = np.real(energies)
    weights = np.exp(-(energies - energies.min()) / KT)
    weights /= weights.sum()

    labels, ops, _ = observable_set(system, H, None)
    labels = list(labels) + ["coherence_01"]
    ops = list(ops) + [states[0] * states[1].dag()]

    # <e|A|e'> for every observable, once. Everything below is a weighted sum
    # of these, so no propagation and no superoperator is needed.
    diag = np.array([np.real([qutip.expect(op, s) for s in states])
                     for op in ops])

    gibbs = diag @ weights

    # The dynamically reachable sectors. A Davies operator is Pi_e X Pi_e', so
    # levels e and e' are connected exactly when <e|X|e'> is non-zero; when that
    # graph is disconnected, each component's population is separately
    # conserved and the limit is Gibbs WITHIN a component. The threshold is
    # stable over six decades (1e-10 to 1e-4 give the same split at dim 256);
    # below 1e-12 eigenvector round-off invents couplings and merges them.
    V = np.column_stack([s.full().ravel() for s in states])
    X_eig = np.abs(V.conj().T @ X.full() @ V)
    adjacency = (X_eig > SECTOR_COUPLING_TOL).astype(np.int8)
    np.fill_diagonal(adjacency, 1)
    n_sectors, sector_of = connected_components(csr_matrix(adjacency),
                                                directed=False)

    pops0 = np.real(np.diag(V.conj().T @ rho0.full() @ V))
    sector_target = np.zeros(len(ops))
    sector_pops = []
    for s in range(n_sectors):
        mask = sector_of == s
        p_s = float(pops0[mask].sum())
        sector_pops.append(p_s)
        if p_s <= 0.0:
            continue
        w_s = np.exp(-(energies[mask] - energies[mask].min()) / KT)
        w_s /= w_s.sum()
        sector_target += p_s * (diag[:, mask] @ w_s)

    sectors = {
        "count": int(n_sectors),
        "sizes": np.bincount(sector_of).tolist(),
        "initial_population": [float(x) for x in sector_pops],
        "coupling_tol": SECTOR_COUPLING_TOL,
    }
    return (labels, ops, gibbs, sector_target, sectors,
            {"levels": [0, 1], "basis": "energy eigenstates"})


def run(system: str, size: int, m_values: list[int],
        thermal_realizations: int, substeps: int) -> None:
    H, X, psi0 = SYSTEMS[system](size)
    rho0 = qutip.ket2dm(psi0)
    dimension = H.shape[0]
    identity = qutip.qeye(H.dims[0])

    t0 = time.perf_counter()
    n_l = davies_operator_count(H, X, gamma,
                                degeneracy_tol=DAVIES_DEGENERACY_TOL)
    t_count = time.perf_counter() - t0

    list_bytes = n_l * dimension * dimension * 16
    print(f"[extreme] {system}, dim {dimension}, N_L = {n_l:,}, "
          f"{substeps} substeps")
    print(f"          counting N_L took {t_count:.1f} s and built no operators")
    print(f"          the operator LIST would need "
          f"{list_bytes / 1024**3:.1f} GB -- this run never forms it")

    labels, ops, gibbs, sector_target, sectors, coherence = thermal_reference(
        system, H, X, rho0)
    e_ops = list(ops) + [identity]
    i_trace = len(ops)
    print(f"          observables: {labels}")
    print(f"          dynamically reachable sectors: {sectors['count']} "
          f"(sizes {sectors['sizes'][:6]}, initial populations "
          f"{[round(x, 4) for x in sectors['initial_population'][:6]]})")
    if sectors["count"] > 1:
        print(f"          NOTE: the stationary state is NOT unique. Gibbs is "
              f"stationary but so is any sector-weighted mixture, so the")
        print(f"                limit depends on rho0 and the correct target "
              f"is the sector-resolved one.")
    for lbl, g, sg in zip(labels, gibbs, sector_target):
        print(f"            <{lbl}>: global Gibbs {g:+.5f}   "
              f"sector-resolved {sg:+.5f}   (both free)")

    def curves_of(res):
        return np.array([np.asarray(res.expect[j], dtype=float)
                         for j in range(len(ops))])

    def sems_of(res):
        return np.array([np.asarray(res.sem[j], dtype=float)
                         for j in range(len(ops))])

    # --- check 1: does the estimate stop moving as M grows? --------------
    sweep = []
    for m in m_values:
        start = time.perf_counter()
        result = mesolve_ensemble_davies(
            H, rho0, TLIST, X, gamma, M=m, e_ops=e_ops,
            n_realizations=N_REALIZATIONS, rng=RNG, backend="native",
            substeps=substeps, degeneracy_tol=DAVIES_DEGENERACY_TOL,
        )
        wall = time.perf_counter() - start
        curves = curves_of(result)
        trace = np.asarray(result.expect[i_trace], dtype=float)
        sweep.append({
            "M": m,
            "wall_s": wall,
            "curves": np.round(curves, ROUND),
            "sem": np.round(sems_of(result), ROUND),
            "trace": np.round(trace, ROUND),
            "max_trace_deviation": float(np.max(np.abs(trace - 1.0))),
        })
        ends = "  ".join(f"{lbl}={curves[j, -1]:+.4f}"
                         for j, lbl in enumerate(labels))
        print(f"    M={m:3d}  {wall:7.1f} s   {ends}   "
              f"max |Tr-1| = {np.max(np.abs(trace - 1.0)):.2e}", flush=True)

    # --- check 3: the long-time limit, at the largest M ------------------
    m_thermal = max(m_values)
    start = time.perf_counter()
    long_run = mesolve_ensemble_davies(
        H, rho0, TLIST_THERMAL, X, gamma, M=m_thermal, e_ops=e_ops,
        n_realizations=thermal_realizations, rng=RNG + 1,
        backend="native", substeps=substeps,
        degeneracy_tol=DAVIES_DEGENERACY_TOL,
    )
    t_thermal = time.perf_counter() - start
    relaxed = curves_of(long_run)
    sem_thermal = sems_of(long_run)

    print(f"\n  thermal check at M={m_thermal}, {thermal_realizations} "
          f"realizations, t -> {TLIST_THERMAL[-1]:.0f} ({t_thermal:.1f} s)")
    print(f"    {'observable':<14} {'end':>11} {'globalGibbs':>12} "
          f"{'to global':>10} {'sectorGibbs':>12} {'to sector':>10} "
          f"{'s.e.m.':>9}")
    per_obs = []
    for j, lbl in enumerate(labels):
        a, b = float(relaxed[j, 0]), float(relaxed[j, -1])
        g, sg = float(gibbs[j]), float(sector_target[j])
        sem = float(sem_thermal[j, -1])
        rem_g, rem_s = abs(b - g), abs(b - sg)
        per_obs.append({
            "observable": lbl, "start": a, "final": b,
            "gibbs": g, "sector_gibbs": sg,
            "remaining": rem_g,           # legacy name: vs GLOBAL Gibbs
            "remaining_sector": rem_s,
            "fraction_covered": abs(b - a) / max(abs(g - a), 1e-30),
            "fraction_covered_sector": abs(b - a) / max(abs(sg - a), 1e-30),
            "sem_end": sem,
            "remaining_in_sem": rem_g / max(sem, 1e-30),
            "remaining_sector_in_sem": rem_s / max(sem, 1e-30),
        })
        print(f"    {lbl:<14} {b:11.5f} {g:12.5f} "
              f"{rem_g / max(sem, 1e-30):9.1f}s {sg:12.5f} "
              f"{rem_s / max(sem, 1e-30):9.1f}s {sem:9.5f}")
    if sectors["count"] > 1:
        print("    ('to sector' is the honest column: with a non-unique "
              "stationary state, global Gibbs is not where the exact")
        print("     generator goes. Agreement with it means the bundled "
              "dynamics is more ergodic than the generator it approximates,")
        print("     which is an O(1/M) artefact -- it shrinks as M rises.)")

    meta = run_metadata(
        tlist=TLIST, substeps=substeps, system=system, size=size,
        M_VALUES=m_values, N_REALIZATIONS=N_REALIZATIONS, rng=RNG,
        thermal_realizations=thermal_realizations,
        route="streaming (mesolve_ensemble_davies)",
        tlist_thermal=[float(TLIST_THERMAL[0]), float(TLIST_THERMAL[-1]),
                       int(TLIST_THERMAL.size)],
    )
    save_data(
        f"extreme_dimension_{system}_dim{dimension}.json", meta, compact=True,
        dim=dimension, n_l=n_l, t_count=t_count,
        operator_list_bytes=list_bytes,
        observables=labels, coherence=coherence,
        gibbs=np.round(gibbs, ROUND),
        sector_gibbs=np.round(sector_target, ROUND),
        sectors=sectors,
        gibbs_energy=float(gibbs[0]),      # kept so older readers still load
        sweep=sweep,
        thermal={
            "M": m_thermal, "wall_s": t_thermal,
            # Recorded rather than assumed. How close the endpoint sits to Gibbs
            # is only meaningful against this run's own statistical error, and a
            # plot that hardcoded the realization count would mis-size the band
            # that decides the whole check.
            "n_realizations": thermal_realizations,
            "times": np.round(TLIST_THERMAL, 6),
            "curves": np.round(relaxed, ROUND),
            "sem": np.round(sem_thermal, ROUND),
            "per_observable": per_obs,
            # Legacy energy-shaped fields, so the committed plot script and the
            # older data files keep working while both schemas are in the tree.
            "energy": np.round(relaxed[0], ROUND),
            "start": per_obs[0]["start"], "final": per_obs[0]["final"],
            "fraction_covered": per_obs[0]["fraction_covered"],
            "remaining": per_obs[0]["remaining"],
        },
    )
    print(f"\n  -> wrote extreme_dimension_{system}_dim{dimension}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--system", choices=sorted(SYSTEMS),
                        default="mixed_chain",
                        help="default mixed_chain. spin_chain is NOT a valid "
                             "target for the thermal check: its Z2 symmetry "
                             "gives extra conserved quantities, so it relaxes "
                             "to a symmetry-restricted state rather than the "
                             "global Gibbs state, and check 3 would fail for "
                             "reasons unrelated to bundling.")
    parser.add_argument("--size", type=int, default=8,
                        help="spins (or Fock cutoff); 8 gives dimension 256 "
                             "on the chains (default)")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS,
                        help=f"RK4 substeps (default {SUBSTEPS}). The "
                             f"oscillator is stiff and needs far more: 32 at "
                             f"dim 64, 128 at dim 256.")
    parser.add_argument("--m-values", type=int, nargs="+", default=[8, 16, 32],
                        help="bundle sizes, ascending (default: 8 16 32)")
    parser.add_argument("--thermal-realizations", type=int, default=16,
                        help="realizations for the thermal check (default: 16). "
                             "This sets the error bar the residual gap to Gibbs "
                             "is judged against, and the thermal run is the "
                             "most expensive part of this script, so it is the "
                             "one knob worth turning.")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run(args.system, args.size, sorted(args.m_values),
        args.thermal_realizations, args.substeps)


if __name__ == "__main__":
    main()
