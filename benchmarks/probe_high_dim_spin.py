"""Resource-safe feasibility probe for large spin-chain dimensions.

This script counts the grouped Davies frequency sectors without materializing
one dense collapse operator per sector.  It is intended to answer whether a
dimension is safe to attempt before launching a native RK4, mesolve, mcsolve,
or bundled benchmark.

Examples
--------
    python benchmarks/probe_high_dim_spin.py --dims 128 256 512
    python benchmarks/probe_high_dim_spin.py --dims 1024 --json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from common import (
    DAVIES_DEGENERACY_TOL,
    build_spin_chain,
    gamma,
)


COMPLEX_BYTES = np.dtype(np.complex128).itemsize


def _cluster_sorted(values: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Match the non-bridging clustering used by davies_operators."""
    if len(values) == 0:
        return []
    groups: list[list[int]] = [[0]]
    first = float(values[0])
    for index in range(1, len(values)):
        value = float(values[index])
        if value - first <= tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
            first = value
    return [np.asarray(group, dtype=int) for group in groups]


def count_davies_sectors(H, X, tolerance: float) -> dict:
    """Count retained Davies sectors using the production grouping rules."""
    H_arr = np.asarray(H.full())
    H_arr = 0.5 * (H_arr + H_arr.conj().T)
    evals, evecs = np.linalg.eigh(H_arr)
    X_eig = evecs.conj().T @ np.asarray(X.full()) @ evecs

    energy_groups = _cluster_sorted(evals, tolerance)
    energies = np.asarray(
        [float(np.mean(evals[group])) for group in energy_groups],
        dtype=float,
    )

    transitions: list[tuple[float, float]] = []
    for a_group, a_indices in enumerate(energy_groups):
        for b_group, b_indices in enumerate(energy_groups):
            block_norm = float(
                np.linalg.norm(X_eig[np.ix_(a_indices, b_indices)], ord="fro")
            )
            if block_norm < 1e-14:
                continue
            transitions.append(
                (float(energies[b_group] - energies[a_group]), block_norm)
            )

    transitions.sort(key=lambda item: item[0])
    sectors: list[list[tuple[float, float]]] = []
    for transition in transitions:
        if (
            not sectors
            or transition[0] - sectors[-1][0][0] > tolerance
        ):
            sectors.append([transition])
        else:
            sectors[-1].append(transition)

    retained = 0
    for sector in sectors:
        omega = float(np.mean([transition[0] for transition in sector]))
        if abs(omega) <= tolerance:
            omega = 0.0
        bare_norm = math.sqrt(sum(norm * norm for _, norm in sector))
        if math.sqrt(gamma(omega)) * bare_norm != 0.0:
            retained += 1

    return {
        "n_energy_spaces": len(energy_groups),
        "n_nonzero_blocks": len(transitions),
        "n_l": retained,
    }


def probe_dimension(dim: int) -> dict:
    if dim < 2 or dim & (dim - 1):
        raise ValueError(f"dimension must be a power of two, got {dim}")
    n_sites = dim.bit_length() - 1

    started = time.perf_counter()
    H, X, _ = build_spin_chain(n_sites)
    built_s = time.perf_counter() - started

    started = time.perf_counter()
    counts = count_davies_sectors(H, X, DAVIES_DEGENERACY_TOL)
    counted_s = time.perf_counter() - started

    n_l = counts["n_l"]
    one_operator_gib = dim * dim * COMPLEX_BYTES / 2**30
    collapse_list_gib = n_l * one_operator_gib
    native_core_gib = 3.0 * collapse_list_gib

    return {
        "dim": dim,
        "n_sites": n_sites,
        **counts,
        "system_build_s": built_s,
        "sector_count_s": counted_s,
        "dense_operator_mib": one_operator_gib * 1024.0,
        "collapse_list_gib": collapse_list_gib,
        "native_core_gib": native_core_gib,
        "notes": (
            "native_core_gib counts the retained Qobj list plus native RK4's "
            "dense C and C-dagger arrays; solver temporaries and construction "
            "peak memory are additional"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="power-of-two Hilbert dimensions to inspect",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of a compact table",
    )
    args = parser.parse_args()

    rows = [probe_dimension(dim) for dim in args.dims]
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(
        " dim  sites    N_L  energy spaces  blocks  "
        "count time  c_ops GiB  native core GiB"
    )
    for row in rows:
        print(
            f"{row['dim']:4d}  {row['n_sites']:5d}  {row['n_l']:5d}"
            f"  {row['n_energy_spaces']:13d}  "
            f"{row['n_nonzero_blocks']:6d}  "
            f"{row['sector_count_s']:9.2f}s  "
            f"{row['collapse_list_gib']:9.2f}  "
            f"{row['native_core_gib']:15.2f}"
        )


if __name__ == "__main__":
    main()
