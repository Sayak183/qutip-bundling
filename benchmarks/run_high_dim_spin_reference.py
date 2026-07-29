"""Run an isolated, memory-guarded native RK4 spin-chain reference.

The regular cost-scaling runner writes the complete multi-dimension dataset.
This focused runner lets us extend the deterministic reference wall without
overwriting that established file.  It saves the complete primary density-
matrix trajectory as compressed NPZ so new single-time observables can be
calculated later without repeating the expensive propagation.

Example
-------
    python benchmarks/run_high_dim_spin_reference.py --dim 256
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

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


def _solve(
    H,
    rho0,
    c_ops,
    substeps: int,
    *,
    store_states: bool,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    started = time.perf_counter()
    result = rk4_mesolve(
        H,
        rho0,
        TLIST,
        c_ops=c_ops,
        e_ops=[H],
        substeps=substeps,
        store_states=store_states,
    )
    elapsed = time.perf_counter() - started
    states = None
    if store_states:
        states = np.stack(
            [
                np.asarray(state.full(), dtype=np.complex128)
                for state in result.states
            ],
            axis=0,
        )
    return np.real(result.expect[0]), states, elapsed


def _state_diagnostics(states: np.ndarray) -> dict:
    adjoints = np.swapaxes(states.conj(), 1, 2)
    traces = np.trace(states, axis1=1, axis2=2)
    purities = np.real(
        np.einsum("tij,tji->t", states, states, optimize=True)
    )
    return {
        "max_hermiticity_error": float(np.max(np.abs(states - adjoints))),
        "max_trace_drift": float(np.max(np.abs(traces - traces[0]))),
        "trace_real": np.real(traces),
        "trace_imag": np.imag(traces),
        "purity": purities,
    }


def _state_comparison(
    reference_states: np.ndarray,
    partner_states: np.ndarray,
) -> dict:
    differences = partner_states - reference_states
    flat = differences.reshape(differences.shape[0], -1)
    frobenius = np.linalg.norm(flat, axis=1)
    trace_distance = []
    for difference in differences:
        hermitian_difference = 0.5 * (
            difference + difference.conj().T
        )
        eigenvalues = np.linalg.eigvalsh(hermitian_difference)
        trace_distance.append(0.5 * float(np.sum(np.abs(eigenvalues))))
    trace_distance = np.asarray(trace_distance, dtype=float)
    return {
        "frobenius_by_time": frobenius,
        "trace_distance_by_time": trace_distance,
        "max_frobenius_dev": float(np.max(frobenius)),
        "max_trace_distance": float(np.max(trace_distance)),
    }


def _write_state_archive(
    path: Path,
    *,
    times: np.ndarray,
    states: np.ndarray,
    dim: int,
    n_sites: int,
    substeps: int,
) -> None:
    """Atomically save reusable density matrices in a compact binary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                times=np.asarray(times, dtype=float),
                states=np.asarray(states, dtype=np.complex128),
                dim=np.asarray(dim, dtype=np.int64),
                n_sites=np.asarray(n_sites, dtype=np.int64),
                substeps=np.asarray(substeps, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    stem = f"high_dim_reference_spin_chain_dim{args.dim}"
    output_name = f"{stem}.json"
    output_path = DATA_DIR / output_name
    state_path = DATA_DIR / f"{stem}.npz"
    existing = [path for path in (output_path, state_path) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        parser.error(f"{paths} already exist; pass --overwrite to replace them")

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
    reference, reference_states, t_reference = _solve(
        H,
        rho0,
        c_ops,
        args.substeps,
        store_states=True,
    )
    if reference_states is None:
        raise RuntimeError("primary solve did not return stored states")
    print(f"[run] primary finished in {t_reference:.2f} s", flush=True)

    state_diagnostics = _state_diagnostics(reference_states)
    _write_state_archive(
        state_path,
        times=TLIST,
        states=reference_states,
        dim=args.dim,
        n_sites=n_sites,
        substeps=args.substeps,
    )
    state_sha256 = _sha256(state_path)
    print(
        f"[run] reusable density matrices saved to {state_path}",
        flush=True,
    )

    direction = "down"
    partner_substeps = args.check_substeps
    try:
        print(
            f"[check] comparing against substeps={partner_substeps}",
            flush=True,
        )
        partner, partner_states, t_partner = _solve(
            H,
            rho0,
            c_ops,
            partner_substeps,
            store_states=True,
        )
    except SolverInstabilityError:
        direction = "up"
        partner_substeps = 2 * args.substeps
        print(
            f"[check] coarse solve was unstable; retrying upward at "
            f"substeps={partner_substeps}",
            flush=True,
        )
        partner, partner_states, t_partner = _solve(
            H,
            rho0,
            c_ops,
            partner_substeps,
            store_states=True,
        )

    if partner_states is None:
        raise RuntimeError("comparison solve did not return stored states")
    max_abs_dev = float(np.max(np.abs(partner - reference)))
    state_comparison = _state_comparison(reference_states, partner_states)
    max_trace_distance = state_comparison["max_trace_distance"]
    passed = bool(
        np.isfinite(max_abs_dev)
        and max_abs_dev <= args.tol
        and np.isfinite(max_trace_distance)
        and max_trace_distance <= args.tol
    )
    print(
        f"[check] max |delta <H>|={max_abs_dev:.3e}; "
        f"max state trace distance={max_trace_distance:.3e}; "
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
            "state_archive": {
                "filename": state_path.name,
                "format": "numpy_npz_compressed",
                "array": "states",
                "shape": list(reference_states.shape),
                "dtype": str(reference_states.dtype),
                "sha256": state_sha256,
            },
            "state_diagnostics": state_diagnostics,
            "selfcheck": {
                "substeps_pair": sorted([args.substeps, partner_substeps]),
                "direction": direction,
                "max_abs_dev": max_abs_dev,
                **state_comparison,
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
    print(
        f"[done] certified reference saved to {output_path} and {state_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
