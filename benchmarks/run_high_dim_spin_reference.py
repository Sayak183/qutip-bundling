"""Run an isolated, memory-guarded native RK4 spin-chain reference.

The regular cost-scaling runner writes the complete multi-dimension dataset.
This focused runner lets us extend the deterministic reference wall without
overwriting that established file.

Example
-------
    python benchmarks/run_high_dim_spin_reference.py --dim 128
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    DATA_DIR,
    TLIST,
    build_davies_operators,
    build_spin_chain,
    run_metadata,
    save_data,
)
from probe_high_dim_spin import probe_dimension
from qutip_bundling.native_solver import SolverInstabilityError, rk4_mesolve


DEFAULT_SUBSTEPS = 8
DEFAULT_CHECK_SUBSTEPS = 4
DEFAULT_TOL = 1e-4
DEFAULT_MAX_CORE_GIB = 8.0


def _solve(H, rho0, c_ops, substeps: int) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    result = rk4_mesolve(
        H,
        rho0,
        TLIST,
        c_ops=c_ops,
        e_ops=[H],
        substeps=substeps,
    )
    elapsed = time.perf_counter() - started
    return np.real(result.expect[0]), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
    parser.add_argument(
        "--check-substeps",
        type=int,
        default=DEFAULT_CHECK_SUBSTEPS,
        help="coarser comparison used to certify time-step convergence",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument(
        "--max-core-gib",
        type=float,
        default=DEFAULT_MAX_CORE_GIB,
        help="refuse the run when estimated native core storage exceeds this",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.substeps <= 0 or args.check_substeps <= 0:
        parser.error("substep counts must be positive")
    if args.tol <= 0 or args.max_core_gib <= 0:
        parser.error("--tol and --max-core-gib must be positive")

    output_name = f"high_dim_reference_spin_chain_dim{args.dim}.json"
    output_path = DATA_DIR / output_name
    if output_path.exists() and not args.overwrite:
        parser.error(f"{output_path} already exists; pass --overwrite to replace it")

    print(f"[preflight] counting Davies sectors for dim={args.dim}", flush=True)
    resources = probe_dimension(args.dim)
    print(
        f"[preflight] N_L={resources['n_l']}; estimated native core "
        f"{resources['native_core_gib']:.2f} GiB",
        flush=True,
    )
    if resources["native_core_gib"] > args.max_core_gib:
        parser.error(
            "estimated native core storage "
            f"{resources['native_core_gib']:.2f} GiB exceeds the "
            f"{args.max_core_gib:.2f} GiB safety limit"
        )

    n_sites = args.dim.bit_length() - 1
    H, X, psi0 = build_spin_chain(n_sites)
    rho0 = qutip.ket2dm(psi0)

    started = time.perf_counter()
    c_ops = build_davies_operators(H, X)
    t_davies = time.perf_counter() - started
    if len(c_ops) != resources["n_l"]:
        raise RuntimeError(
            f"preflight counted {resources['n_l']} operators but construction "
            f"returned {len(c_ops)}"
        )
    print(
        f"[run] primary native RK4: dim={args.dim}, N_L={len(c_ops)}, "
        f"substeps={args.substeps}",
        flush=True,
    )
    reference, t_reference = _solve(H, rho0, c_ops, args.substeps)
    print(f"[run] primary finished in {t_reference:.2f} s", flush=True)

    direction = "down"
    partner_substeps = args.check_substeps
    try:
        print(
            f"[check] comparing against substeps={partner_substeps}",
            flush=True,
        )
        partner, t_partner = _solve(H, rho0, c_ops, partner_substeps)
        max_abs_dev = float(np.max(np.abs(partner - reference)))
    except SolverInstabilityError:
        direction = "up"
        partner_substeps = 2 * args.substeps
        print(
            f"[check] coarse solve was unstable; retrying upward at "
            f"substeps={partner_substeps}",
            flush=True,
        )
        partner, t_partner = _solve(H, rho0, c_ops, partner_substeps)
        max_abs_dev = float(np.max(np.abs(partner - reference)))

    passed = bool(np.isfinite(max_abs_dev) and max_abs_dev <= args.tol)
    print(
        f"[check] max |delta <H>|={max_abs_dev:.3e}; "
        f"{'PASS' if passed else 'FAIL'} (tol={args.tol:g})",
        flush=True,
    )

    meta = run_metadata(
        tlist=TLIST,
        substeps=args.substeps,
        system="spin_chain",
        purpose="isolated_high_dimension_native_reference",
        dimension=args.dim,
        n_sites=n_sites,
    )
    save_data(
        output_name,
        meta,
        resource_preflight=resources,
        point={
            "dim": args.dim,
            "n_sites": n_sites,
            "n_l": len(c_ops),
            "t_davies": t_davies,
            "reference_method": f"native_rk4_substeps{args.substeps}",
            "reference_energy": reference,
            "t_reference": t_reference,
            "selfcheck": {
                "substeps_pair": sorted([args.substeps, partner_substeps]),
                "direction": direction,
                "max_abs_dev": max_abs_dev,
                "tol": args.tol,
                "passed": passed,
                "t_partner": t_partner,
            },
        },
    )

    if not passed:
        raise SystemExit(
            "Reference was saved for diagnosis but failed certification."
        )
    print(f"[done] certified reference saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
