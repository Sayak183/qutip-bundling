r"""
probe_oq4_accuracy.py
====================
Probe Open Question 4: What predicts bundling accuracy?

Analyzes:
1. Error scaling with M across saved dataset files.
2. Dissipator variance & cross-operator overlap sum via exact trace identities.
3. Transition locality / bandwidth of c_\alpha in the Hamiltonian eigenbasis.
"""

from __future__ import annotations
import math
import json
import sys
from pathlib import Path

# Ensure repo root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import qutip

from benchmarks.common import (
    build_spin_chain,
    build_mixed_field_chain,
    build_oscillator_bath,
    build_davies_operators,
    DATA_DIR,
)

def analyze_dataset_errors():
    print("=== PROBE 1: DATASET ERROR SCALING WITH M & SYSTEM COMPARISON ===", flush=True)
    systems = ["spin_chain", "mixed_chain", "oscillator_bath"]
    dims = [16, 32, 64, 128]
    
    print(f"{'System':<18} | {'Dim':<4} | {'M':<3} | {'N_L':<5} | {'Rel Error (Energy)':<20} | {'Abs Error':<12}", flush=True)
    print("-" * 75, flush=True)
    
    for sys_name in systems:
        for dim in dims:
            filename = f"method_comparison_{sys_name}_dim{dim}.json"
            path = DATA_DIR / filename
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            
            point = data.get("point", {})
            N_L = point.get("n_l", "N/A")
            ref_dict = point.get("reference", {})
            ref_curves = ref_dict.get("curves", {})
            if "energy" not in ref_curves:
                continue
            
            ref_e = np.array(ref_curves["energy"])
            delta_ref = np.ptp(ref_e)
            
            slb_list = point.get("methods", {}).get("slb", [])
            for slb_item in slb_list:
                M = slb_item.get("M")
                samples = np.array(slb_item.get("samples", [])) # (n_runs, n_obs, n_times)
                if samples.ndim == 3:
                    # Index 0 is energy
                    energy_samples = samples[:, 0, :]
                    mean_e = energy_samples.mean(axis=0)
                    abs_err = np.mean(np.abs(mean_e - ref_e))
                    rel_err = abs_err / delta_ref if delta_ref > 1e-12 else np.nan
                    print(f"{sys_name:<18} | {dim:<4d} | {M:<3d} | {str(N_L):<5s} | {rel_err:<20.4e} | {abs_err:<12.4e}", flush=True)


def compute_operator_structure(H, X):
    r"""Compute structural metrics of Davies collapse operators:
    - N_L: number of operators
    - Total norm sum: sum_alpha ||c_alpha||_F^2
    - Cross-operator Frobenius overlap: sum_{alpha != beta} |Tr(c_alpha^\dagger c_beta)|^2
    - Cross-dissipator potential variance: sum_{alpha != beta} ||c_alpha c_beta^\dagger||_F^2
    - Mean bandwidth in energy eigenbasis
    """
    c_ops = build_davies_operators(H, X)
    N_L = len(c_ops)
    if N_L == 0:
        return {}
    
    Ha = 0.5 * (np.asarray(H.full()) + np.asarray(H.full()).conj().T)
    evals, evecs = np.linalg.eigh(Ha)
    
    C = np.array([c.full() for c in c_ops]) # (N_L, N, N)
    N = C.shape[1]
    
    # 1. Total Frobenius norm squared
    norms_sq = np.sum(np.abs(C)**2, axis=(1, 2))
    total_norm_sq = float(np.sum(norms_sq))
    
    # 2. Trace overlap matrix: Tr(c_i^\dagger c_j)
    tr_mat = np.einsum('ijk,ljk->il', C.conj(), C)
    cross_tr_sq = float(np.sum(np.abs(tr_mat)**2) - np.sum(np.abs(np.diagonal(tr_mat))**2))
    
    # 3. Fast vectorized sum_{i != j} ||c_i c_j^\dagger||_F^2
    A_dagA = np.sum(np.einsum('ijk,ijl->ikl', C.conj(), C), axis=0) # (N, N)
    A_Adag = np.sum(np.einsum('ijk,ilk->ijl', C, C.conj()), axis=0) # (N, N)
    
    total_prod_norm_sq = np.real(np.trace(A_dagA @ A_Adag))
    diag_prod_norm_sq = sum(np.linalg.norm(C[i] @ C[i].conj().T, 'fro')**2 for i in range(N_L))
    cross_prod_norm_sq = float(total_prod_norm_sq - diag_prod_norm_sq)
    
    # 4. Bandwidth in Hamiltonian eigenbasis
    C_tilde = np.einsum('ba,iab,bc->iac', evecs.conj(), C, evecs)
    idx_i, idx_j = np.ogrid[:N, :N]
    dist_mat = np.abs(idx_i - idx_j)
    
    weighted_dist = float(np.sum(dist_mat * np.abs(C_tilde)**2))
    mean_bandwidth = weighted_dist / total_norm_sq if total_norm_sq > 0 else 0.0
    
    E_diff = np.abs(evals[:, None] - evals[None, :])
    weighted_energy_dist = float(np.sum(E_diff * np.abs(C_tilde)**2))
    mean_energy_bandwidth = weighted_energy_dist / total_norm_sq if total_norm_sq > 0 else 0.0
    
    # 5. Variance ratio
    var_ratio = cross_prod_norm_sq / (total_norm_sq**2) if total_norm_sq > 0 else 0.0

    return {
        "N_L": N_L,
        "total_norm_sq": total_norm_sq,
        "cross_tr_sq": cross_tr_sq,
        "cross_prod_norm_sq": cross_prod_norm_sq,
        "var_ratio": var_ratio,
        "mean_index_bandwidth": mean_bandwidth,
        "mean_energy_bandwidth": mean_energy_bandwidth,
    }


def analyze_operator_structures():
    print("\n=== PROBE 2: OPERATOR MATRIX TOPOLOGY & CROSS-INTERACTION VARIANCE ===", flush=True)
    dims = [16, 32, 64]
    
    print(f"{'System':<18} | {'Dim':<4} | {'N_L':<5} | {'NormSq':<8} | {'Bandwidth (idx)':<15} | {'Bandwidth (E)':<15} | {'Var Ratio':<12}", flush=True)
    print("-" * 92, flush=True)
    
    configs = [
        ("spin_chain", lambda d: build_spin_chain(int(np.log2(d)))),
        ("mixed_chain", lambda d: build_mixed_field_chain(int(np.log2(d)))),
        ("oscillator_bath", lambda d: build_oscillator_bath(d)),
    ]
    
    for sys_name, builder in configs:
        for dim in dims:
            try:
                H, X, _ = builder(dim)
                res = compute_operator_structure(H, X)
                print(
                    f"{sys_name:<18} | {dim:<4d} | {res['N_L']:<5d} | {res['total_norm_sq']:<8.2f} | "
                    f"{res['mean_index_bandwidth']:<15.3f} | {res['mean_energy_bandwidth']:<15.3f} | {res['var_ratio']:<12.4e}",
                    flush=True
                )
            except Exception as e:
                print(f"{sys_name:<18} | {dim:<4d} | ERROR: {e}", flush=True)

if __name__ == "__main__":
    analyze_dataset_errors()
    analyze_operator_structures()
