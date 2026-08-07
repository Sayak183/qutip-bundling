"""
run_accuracy_vs_M.py
====================

DATA-GENERATION HALF of the accuracy-versus-bundle-size benchmark (Result 1).
All the compute, none of the plotting; the figures are drawn from the saved
data by plot_accuracy_vs_M.py.

For each system, at one fixed reference-feasible size, it records:

  * construction vs dynamics, timed separately: how long building the N_L
    Davies/Lindblad operators takes, versus how long the reference solve and
    each SLB ensemble propagation take. The two are different costs with
    different scalings, and the saved numbers keep them distinct instead of
    blurring them into one.
  * the exact reference dynamics for TWO observables: the energy <H(t)>
    (diagonal-dominated) and the dominant coherence <C(t)> -- the energy-
    eigenstate pair (a,b) whose off-diagonal is most populated by the actual
    dynamics, chosen from the reference states. Tracking both shows SLB
    reproduces populations AND coherences.
  * SLB, for each bundle size M on the ladder: the raw per-realization curves
    of both observables (N_REALIZATIONS x n_times each) and the wall-clock of
    that ensemble solve. Saving raw realizations is the point of the split:
    the mean curve, the +/-1 std band, and the peak-error bias/fluctuation
    decomposition are all derived at analysis time, so new views of this data
    (different bands, different error anatomies) need no re-run.

Uses the fine 80-point time grid (TLIST_FINE), like the published Result 1.

Writes, per system and dimension:  data/accuracy_vs_M_<system>_dim<D>.json
Run:                 python run_accuracy_vs_M.py (--system ... | --all)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    build_davies_operators,
    build_spin_chain, build_oscillator_bath, build_mixed_field_chain,
    TLIST_FINE, SUBSTEPS,
    DATA_DIR, MAX_FULL_DIM, populated_coherence_op, run_metadata, save_data,
)
from benchmark_cli import (
    add_max_full_dim_argument, add_safety_arguments, preflight_run,
    selected_systems,
)
from qutip_bundling import mesolve_ensemble
from qutip_bundling.native_solver import rk4_mesolve, SolverInstabilityError

N_REALIZATIONS = 200    # realizations per M (fixed across the ladder: nothing
                        # about the sampling is tuned, so any trend with M is
                        # purely the effect of M)
RNG = 0                 # seed (matches prior runs)
ROUND = 8               # decimals kept for saved curves

# M ladder per system: system-specific so the plots do not become visually
# cluttered when M=8 already sits essentially on the reference.
# Each system sweeps a LIST of size-points (size, m_ladder, substeps). run_*
# computes every point ONCE, saving one file per dim; plot_* picks a dim to
# draw. substeps>4 flags a disclosed higher-resolution run (the oscillator's
# stiff dim-64 needs 16).
NATIVE_REF_SUBSTEPS_FACTOR = 2

SYSTEMS = {
    "spin_chain": (build_spin_chain, [
        (4, [2, 4, 8, 16, 32, 64], 4),
        (5, [2, 4, 8, 16, 32, 64], 4),
        (6, [2, 4, 8, 16, 32, 64], 4),
    ]),
    # Same sizes as the TFIM chain so the two are directly comparable: they
    # differ only by the longitudinal field.
    "mixed_chain": (build_mixed_field_chain, [
        (4, [2, 4, 8, 16, 32, 64], 4),
        (5, [2, 4, 8, 16, 32, 64], 4),
        (6, [2, 4, 8, 16, 32, 64], 4),
    ]),
    "oscillator_bath": (build_oscillator_bath, [
        (8,  [2, 4, 8, 16, 32, 64], 4),
        (16, [2, 4, 8, 16, 32, 64], 4),
        (32, [2, 4, 8, 16, 32, 64], 16),
    ]),
}


def capped_unique_m_values(requested, n_lindblad):
    values = []
    for m in requested:
        m_eff = min(int(m), n_lindblad)
        if m_eff > 0 and m_eff not in values:
            values.append(m_eff)
    return values


def run(name, build, size, m_ladder, substeps):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    dim = H.shape[0]

    # --- construction, timed on its own (this is NOT dynamics) ---
    t0 = time.perf_counter()
    c_ops = build_davies_operators(H, X)
    t_davies = time.perf_counter() - t0
    n_l = len(c_ops)

    # --- exact reference: mesolve while feasible, else certified native RK4 ---
    ref_substeps = NATIVE_REF_SUBSTEPS_FACTOR * substeps
    ref_method, ref_selfcheck, ref_states = None, None, None
    if dim <= MAX_FULL_DIM:
        try:
            t0 = time.perf_counter()
            ref_states = qutip.mesolve(H, rho0, TLIST_FINE, c_ops=c_ops,
                                       e_ops=[]).states
            t_reference = time.perf_counter() - t0
            ref_method = "mesolve"
        except MemoryError:
            ref_states = None
    if ref_states is None:
        t0 = time.perf_counter()
        res = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[],
                          substeps=ref_substeps, store_states=True)
        t_reference = time.perf_counter() - t0
        ref_states = res.states
        lo = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[H],
                         substeps=ref_substeps // 2)
        hi = rk4_mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[H],
                         substeps=ref_substeps)
        dev = float(np.max(np.abs(np.real(hi.expect[0]) - np.real(lo.expect[0]))))
        ref_selfcheck = {"substeps": ref_substeps, "max_abs_dev": dev,
                         "passed": bool(np.isfinite(dev) and dev <= 1e-4)}
        ref_method = f"native_rk4_substeps{ref_substeps}"
        if not ref_selfcheck["passed"]:
            print(f"[{name}] dim={dim}: native reference self-check FAILED "
                  f"(dev {dev:.1e}) -- skipping (uncertifiable).")
            return
    C, (ia, ib), peak = populated_coherence_op(H, ref_states)
    e_ops = [H, C]
    ref_energy = np.real(qutip.expect(H, ref_states))
    ref_coherence = np.real(qutip.expect(C, ref_states))

    print(f"[{name}] dim={dim}, N_L={n_l}; Davies {t_davies*1e3:.1f} ms, "
          f"reference ({ref_method}) {t_reference:.2f} s, {substeps} substeps")
    print(f"  coherence on eigenstate pair ({ia},{ib}), peak |rho_ab|={peak:.2e}")

    # --- SLB ensemble per M: raw realizations of both observables ---
    m_values = capped_unique_m_values(m_ladder, n_l)
    guard = 100.0 * (1.0 + float(np.max(np.abs(ref_energy))))
    sweep = []
    for m_eff in m_values:
        t0 = time.perf_counter()
        ens = mesolve_ensemble(H, rho0, TLIST_FINE, c_ops, M=m_eff, e_ops=e_ops,
                               n_realizations=N_REALIZATIONS, rng=RNG,
                               backend="native", substeps=substeps)
        dt = time.perf_counter() - t0
        se = np.real(ens.samples[:, 0, :])
        if not np.isfinite(se).all() or float(np.max(np.abs(se))) > guard:
            print(f"    M={m_eff:3d}  SOFT DIVERGENCE at {substeps} substeps -- skipped")
            continue
        sweep.append({
            "M": m_eff, "cost": dt,
            "samples_energy": np.round(se, ROUND),
            "samples_coherence": np.round(np.real(ens.samples[:, 1, :]), ROUND),
        })
        print(f"    M={m_eff:3d}  ensemble ({N_REALIZATIONS} realizations) "
              f"= {dt:.2f} s")

    meta = run_metadata(
        tlist=TLIST_FINE, max_full_dim=MAX_FULL_DIM,
        system=name, size=size, M_LADDER=m_ladder, substeps=substeps,
        N_REALIZATIONS=N_REALIZATIONS, rng=RNG,
    )
    save_data(f"accuracy_vs_M_{name}_dim{dim}.json", meta, compact=True,
              dim=dim, n_l=n_l, substeps=substeps,
              reference_method=ref_method, reference_selfcheck=ref_selfcheck,
              t_davies=t_davies, t_reference=t_reference,
              coherence_pair=[ia, ib], coherence_peak=peak,
              reference_energy=np.round(ref_energy, ROUND),
              reference_coherence=np.round(ref_coherence, ROUND),
              slb_sweep=sweep)
    print(f"  -> wrote accuracy_vs_M_{name}_dim{dim}.json")


def main():
    global MAX_FULL_DIM
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_safety_arguments(ap, SYSTEMS)
    add_max_full_dim_argument(ap, MAX_FULL_DIM)
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="only these Hilbert dims (default: all configured "
                         "sizes; each saved to its own file).")
    args = ap.parse_args()
    if args.max_full_dim != MAX_FULL_DIM:
        MAX_FULL_DIM = args.max_full_dim
        print(f"[config] exact-mesolve dimension cap raised to {MAX_FULL_DIM}")
    names = selected_systems(args, SYSTEMS)
    work = []
    plans = []
    for name in names:
        build, points = SYSTEMS[name]
        available_dims = set()
        for size, m_ladder, substeps in points:
            probe_dim = build(size)[0].shape[0]
            available_dims.add(probe_dim)
            if args.dims and probe_dim not in args.dims:
                continue
            work.append((name, build, size, m_ladder, substeps))
            plans.append((
                f"Result 1: {name}, dim {probe_dim}, M={m_ladder}",
                DATA_DIR / f"accuracy_vs_M_{name}_dim{probe_dim}.json",
            ))
        if args.dims:
            missing = sorted(set(args.dims) - available_dims)
            if missing:
                ap.error(
                    f"{name} has no configured Result 1 dimensions {missing}; "
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
