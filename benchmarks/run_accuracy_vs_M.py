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

Writes, per system:  data/accuracy_vs_M_<system>.json
Run:                 python run_accuracy_vs_M.py [--system ...]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import qutip

from common import (
    gamma, build_spin_chain, build_oscillator_bath, TLIST_FINE, SUBSTEPS,
    run_metadata, save_data,
)
from qutip_bundling import davies_operators, mesolve_ensemble

N_REALIZATIONS = 200    # realizations per M (fixed across the ladder: nothing
                        # about the sampling is tuned, so any trend with M is
                        # purely the effect of M)
RNG = 0                 # seed (matches prior runs)
ROUND = 8               # decimals kept for saved curves

# M ladder per system: system-specific so the plots do not become visually
# cluttered when M=8 already sits essentially on the reference.
SYSTEMS = {
    "spin_chain":      (build_spin_chain,      4, [2, 4, 8, 16, 32, 64]),  # dim 16
    "oscillator_bath": (build_oscillator_bath, 8, [2, 4, 8, 16, 32, 64]),  # dim 16
}


def capped_unique_m_values(requested, n_lindblad):
    values = []
    for m in requested:
        m_eff = min(int(m), n_lindblad)
        if m_eff > 0 and m_eff not in values:
            values.append(m_eff)
    return values


def populated_coherence_op(H, ref_states):
    """Hermitian coherence operator |a><b| + h.c. for the energy-eigenstate pair
    (a, b) whose coherence is *most populated by the actual dynamics* (largest
    |<a|rho(t)|b>| over the reference trajectory). This guarantees we track a
    coherence the system genuinely develops -- picking by coupling strength can
    land on a pair the dynamics never populates (value ~ machine zero), which is
    uninformative. <H> is essentially diagonal, so this off-diagonal is exactly
    what energy cannot see."""
    Ha = 0.5 * (np.asarray(H.full()) + np.asarray(H.full()).conj().T)
    evals, evecs = np.linalg.eigh(Ha)
    R = evecs.conj().T  # rows are eigenvectors
    best = (0, 1, -1.0)
    for s in ref_states:
        rho_e = R @ np.asarray(s.full()) @ R.conj().T
        ab = np.abs(rho_e)
        np.fill_diagonal(ab, 0.0)
        i, j = np.unravel_index(int(np.argmax(ab)), ab.shape)
        if ab[i, j] > best[2]:
            best = (int(i), int(j), float(ab[i, j]))
    a, b = best[0], best[1]
    P = np.outer(evecs[:, a], evecs[:, b].conj())
    C = qutip.Qobj(P + P.conj().T, dims=H.dims)
    return C, (a, b), best[2]


def run(name, build, size, m_ladder):
    H, X, psi0 = build(size)
    rho0 = qutip.ket2dm(psi0)
    dim = H.shape[0]

    # --- construction, timed on its own (this is NOT dynamics) ---
    t0 = time.perf_counter()
    c_ops = davies_operators(H, X, gamma)
    t_davies = time.perf_counter() - t0
    n_l = len(c_ops)

    # --- exact reference (states kept: they choose the coherence pair) ---
    t0 = time.perf_counter()
    ref_states = qutip.mesolve(H, rho0, TLIST_FINE, c_ops=c_ops, e_ops=[]).states
    t_reference = time.perf_counter() - t0
    C, (ia, ib), peak = populated_coherence_op(H, ref_states)
    e_ops = [H, C]
    ref_energy = np.real(qutip.expect(H, ref_states))
    ref_coherence = np.real(qutip.expect(C, ref_states))

    print(f"[{name}] dim={dim}, N_L={n_l}; Davies construction {t_davies*1e3:.1f} ms, "
          f"reference solve {t_reference:.2f} s")
    print(f"  coherence on eigenstate pair ({ia},{ib}), peak |rho_ab|={peak:.2e}")

    # --- SLB ensemble per M: raw realizations of both observables ---
    m_values = capped_unique_m_values(m_ladder, n_l)
    sweep = []
    for m_eff in m_values:
        t0 = time.perf_counter()
        ens = mesolve_ensemble(H, rho0, TLIST_FINE, c_ops, M=m_eff, e_ops=e_ops,
                               n_realizations=N_REALIZATIONS, rng=RNG,
                               backend="native", substeps=SUBSTEPS)
        dt = time.perf_counter() - t0
        sweep.append({
            "M": m_eff, "cost": dt,
            "samples_energy": np.round(np.real(ens.samples[:, 0, :]), ROUND),
            "samples_coherence": np.round(np.real(ens.samples[:, 1, :]), ROUND),
        })
        print(f"    M={m_eff:3d}  ensemble ({N_REALIZATIONS} realizations) "
              f"= {dt:.2f} s")

    meta = run_metadata(
        tlist=TLIST_FINE,
        system=name, size=size, M_LADDER=m_ladder,
        N_REALIZATIONS=N_REALIZATIONS, rng=RNG,
    )
    save_data(f"accuracy_vs_M_{name}.json", meta, compact=True,
              dim=dim, n_l=n_l,
              t_davies=t_davies, t_reference=t_reference,
              coherence_pair=[ia, ib], coherence_peak=peak,
              reference_energy=np.round(ref_energy, ROUND),
              reference_coherence=np.round(ref_coherence, ROUND),
              slb_sweep=sweep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--system", default="all",
                    choices=[*SYSTEMS, "all"],
                    help="which system to run (default: all)")
    args = ap.parse_args()
    names = list(SYSTEMS) if args.system == "all" else [args.system]
    for name in names:
        run(name, *SYSTEMS[name])


if __name__ == "__main__":
    main()
