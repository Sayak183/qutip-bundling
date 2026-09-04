from __future__ import annotations

import argparse
import math
import os
import time
import numpy as np
import qutip

try:
    import resource
    def get_mem_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:
    try:
        import psutil
        def get_mem_mb():
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        def get_mem_mb():
            return 0.0

import common
from qutip_bundling import rk4_mesolve
from qutip_bundling.operators import (
    davies_operators, davies_operator_count,
    bundle, bundle_davies_from_phases, random_phases,
)

# Below this dimension we use the list path (davies_operators + bundle).
# Above it we use the streaming path (bundle_davies_from_phases) to avoid OOM.
STREAMING_THRESHOLD = 256

# RK4's stability limit on |lambda * dt| for eigenvalues near the imaginary
# axis. A Lindblad generator sits close enough to that axis for this to be the
# binding constraint in practice.
RK4_STABILITY_LIMIT = 2.0 * math.sqrt(2.0)


def spectral_bound(H):
    """Gershgorin upper bound on the largest |eigenvalue| of Hermitian H.

    The spectral radius never exceeds the largest absolute row sum. Used in
    place of an eigendecomposition because it is one pass over H and errs
    high -- the safe direction for a step-size rule.
    """
    return float(np.abs(H.full()).sum(axis=1).max())


def table_substeps(system_name, dim):
    """The hand-tuned substeps this benchmark has always used."""
    if system_name == 'oscillator_bath':
        if dim <= 16:    return 4
        elif dim <= 64:  return 16
        elif dim <= 128: return 32
        elif dim <= 256: return 64
        elif dim <= 512: return 128
        else:            return 256
    if dim <= 64:  return 4
    elif dim <= 256: return 8
    else:          return 16


def choose_substeps(H, tlist, system_name):
    """Substeps per tlist interval: the table, raised where RK4 needs more.

    WHY THIS IS NO LONGER JUST A TABLE. The oscillator Hamiltonian carries an
    anharmonic `anh * n^2` term, so its top energy grows as the SQUARE of the
    Fock cutoff -- doubling the cutoff quadruples the largest frequency. The
    table doubled per octave, held on borrowed margin for four octaves, and
    then failed: job 19602988 diverged at Fock 512 with substeps=256, matrix
    entries reaching 1.7e184 by t=0.05.

    Measured against the stability limit, at dt = 0.05:

        Fock 128, substeps  64  ->  1.32   ran
        Fock 256, substeps 128  ->  2.64   ran, 7% to spare
        Fock 512, substeps 256  ->  5.20   diverged

    Taking max(table, rule) rather than the rule alone is deliberate. The rule
    on its own would halve substeps at Fock 128 -- defensible, but it would
    make new runs incomparable with the ones already completed. This raises
    substeps only where the table was too small to be stable.

    Spin chains are left on the table: their spectral bound is about 17 at 11
    spins, so the rule would ask for substeps=1 against the table's 16.
    """
    table = table_substeps(system_name, H.shape[0])
    if system_name != 'oscillator_bath':
        return table

    dt = float(np.min(np.diff(np.asarray(tlist, dtype=float))))
    needed = spectral_bound(H) * dt / RK4_STABILITY_LIMIT
    substeps = 4
    while substeps < needed:
        substeps *= 2
    return max(table, substeps)


def run_system_frontier(system_name: str, dims: list[int],
                        m_values: list[int] = [16, 32, 64],
                        out_name: str | None = None):
    print(f"\n{'='*70}")
    print(f"  FRONTIER BENCHMARK: {system_name.upper()}")
    print(f"  Dimensions to test: {dims}")
    print(f"  Bundle sizes (M): {m_values}")
    print(f"  Streaming threshold: N > {STREAMING_THRESHOLD}")
    print(f"{'='*70}\n")

    results = []
    out_name = out_name or f"frontier_spins_{system_name}.json"
    meta = common.run_metadata(
        tlist=np.linspace(0, 5.0, 101), substeps=None,
        system=system_name, dims=list(dims), m_values=list(m_values),
        streaming_threshold=STREAMING_THRESHOLD,
        substeps_rule="max(table, RK4 stability bound); see choose_substeps",
    )

    for dim in dims:
        print(f"\n--- Dimension N = {dim} ---")
        mem_start = get_mem_mb()

        if system_name == 'mixed_chain':
            n_spins = int(round(np.log2(dim)))
            print(f"  System: Mixed-Field Spin Chain with {n_spins} spins (N = {dim})")
            H, X, psi0 = common.build_mixed_field_chain(n_spins)
        elif system_name == 'spin_chain':
            n_spins = int(round(np.log2(dim)))
            print(f"  System: TFIM Spin Chain with {n_spins} spins (N = {dim})")
            H, X, psi0 = common.build_spin_chain(n_spins)
        elif system_name == 'oscillator_bath':
            fock_cutoff = dim // 2
            print(f"  System: Oscillator + Spin with Fock cutoff {fock_cutoff} (N = {dim})")
            H, X, psi0 = common.build_oscillator_bath(fock_cutoff)
        else:
            raise ValueError(f"Unknown system: {system_name}")

        rho0 = psi0 * psi0.dag()
        tlist = np.linspace(0, 5.0, 101)
        use_streaming = dim > STREAMING_THRESHOLD
        davies_kw = dict(degeneracy_tol=common.DAVIES_DEGENERACY_TOL)

        # --- [1] Davies operator count (lightweight, no memory) ---
        t0 = time.perf_counter()
        n_l = davies_operator_count(H, X, common.gamma, **davies_kw)
        t_count = time.perf_counter() - t0

        if not use_streaming:
            # Small dim: build the full operator list
            t0 = time.perf_counter()
            c_ops = davies_operators(H, X, common.gamma, **davies_kw)
            t_davies = time.perf_counter() - t0
        else:
            t_davies = t_count  # streaming skips list construction

        mem_after_davies = get_mem_mb()
        route = "STREAMING (never holds N_L)" if use_streaming else "LIST (all in memory)"

        print(f"  [1] Davies Construction ({route}):")
        print(f"      - Operators (N_L): {n_l:,}")
        print(f"      - Construction Time: {t_davies:.2f} s")
        print(f"      - Memory: {mem_after_davies - mem_start:.1f} MB (Total: {mem_after_davies:.1f} MB)")

        dim_res = {
            'dim': dim,
            'n_l': n_l,
            't_davies': t_davies,
            'streaming': use_streaming,
            'm_runs': {}
        }

        substeps = choose_substeps(H, tlist, system_name)
        dim_res['substeps'] = substeps
        table = table_substeps(system_name, dim)
        note = "" if substeps == table else f"  [raised from {table} for stability]"
        print(f"  [2] SLB Dynamics Propagation (substeps={substeps}){note}")

        rng = np.random.default_rng(42)
        final_states = {}
        for M in m_values:
            if M > n_l:
                print(f"      - M = {M}: Skipped (M > N_L)")
                continue

            t0_bundle = time.perf_counter()
            if use_streaming:
                # Streaming: build M bundles without ever holding N_L operators
                phases = random_phases(M, n_l, distribution="phase", rng=rng)
                b_ops = bundle_davies_from_phases(H, X, common.gamma, phases, **davies_kw)
            else:
                b_ops = bundle(c_ops, M=M, rng=rng)
            t_bundle_prep = time.perf_counter() - t0_bundle

            t0_dyn = time.perf_counter()
            res = rk4_mesolve(H, rho0, tlist, c_ops=b_ops, substeps=substeps, store_states=True)
            t_dyn = time.perf_counter() - t0_dyn

            final_states[M] = np.asarray(res.states[-1].full())
            mem_dyn = get_mem_mb()

            print(f"      - M = {M:2d}: Dynamics Time = {t_dyn:7.2f}s  "
                  f"(Prep: {t_bundle_prep*1000:4.1f} ms) | RAM: {mem_dyn:.1f} MB")

            dim_res['m_runs'][M] = {
                't_dyn': t_dyn,
                't_bundle_prep': t_bundle_prep
            }

        if 32 in final_states and 64 in final_states:
            frobenius_diff = np.linalg.norm(final_states[64] - final_states[32], ord='fro')
            trace_diff = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(final_states[64] - final_states[32])))
            print(f"  [3] Self-Convergence (M=32 -> M=64 at t=5.0):")
            print(f"      - Frobenius distance: {frobenius_diff:.4e}")
            print(f"      - Trace distance:     {trace_diff:.4e}")
            dim_res['self_convergence'] = {
                'from_m': 32, 'to_m': 64,
                'frobenius': float(frobenius_diff),
                'trace': float(trace_diff),
            }

        results.append(dim_res)

        # Write after EVERY dimension, not at the end. Job 19602988 spent 24 h
        # on Fock 32-256, diverged at Fock 512, and lost all of it because this
        # runner used to return its results and never save them.
        common.save_data(out_name, meta, points=results)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*70}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description='Frontier large-dimension SLB benchmark')
    parser.add_argument('--system', default='mixed_chain', choices=['mixed_chain', 'spin_chain', 'oscillator_bath'])
    parser.add_argument('--dims', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--m-values', type=int, nargs='+', default=[16, 32, 64])
    parser.add_argument('--out', default=None,
                        help='output filename under data/ '
                             '(default: frontier_spins_<system>.json)')
    args = parser.parse_args()

    run_system_frontier(args.system, args.dims, args.m_values, args.out)


if __name__ == '__main__':
    main()
