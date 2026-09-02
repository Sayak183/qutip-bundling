from __future__ import annotations

import argparse
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
from qutip_bundling import davies_operators, bundle, rk4_mesolve


def run_system_frontier(system_name: str, dims: list[int], m_values: list[int] = [16, 32, 64]):
    print(f"\n{'='*70}")
    print(f"  FRONTIER BENCHMARK: {system_name.upper()}")
    print(f"  Dimensions to test: {dims}")
    print(f"  Bundle sizes (M): {m_values}")
    print(f"{'='*70}\n")

    results = []

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

        t0 = time.perf_counter()
        c_ops = davies_operators(H, X, common.gamma, degeneracy_tol=common.DAVIES_DEGENERACY_TOL)
        t_davies = time.perf_counter() - t0
        n_l = len(c_ops)
        mem_after_davies = get_mem_mb()

        print(f"  [1] Davies Construction:")
        print(f"      - Operators (N_L): {n_l:,}")
        print(f"      - Construction Time: {t_davies:.2f} s")
        print(f"      - Memory: {mem_after_davies - mem_start:.1f} MB (Total: {mem_after_davies:.1f} MB)")

        dim_res = {
            'dim': dim,
            'n_l': n_l,
            't_davies': t_davies,
            'm_runs': {}
        }

        substeps = 4 if dim <= 64 else (8 if dim <= 256 else 16)
        print(f"  [2] SLB Dynamics Propagation (substeps={substeps}):")

        final_states = {}
        for M in m_values:
            if M > n_l:
                print(f"      - M = {M}: Skipped (M > N_L)")
                continue

            t0_bundle = time.perf_counter()
            b_ops = bundle(c_ops, M=M)
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

        results.append(dim_res)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*70}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description='Frontier large-dimension SLF benchmark')
    parser.add_argument('--system', default='mixed_chain', choices=['mixed_chain', 'spin_chain', 'oscillator_bath'])
    parser.add_argument('--dims', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--m-values', type=int, nargs='+', default=[16, 32, 64])
    args = parser.parse_args()

    run_system_frontier(args.system, args.dims, args.m_values)


if __name__ == '__main__':
    main()
