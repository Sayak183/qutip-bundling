r"""
probe_oq4_crossterm.py
======================
Probe Open Question 4 with the cross-term ratio.

Bundling's error is EXACTLY the cross terms c_a rho c_b^dag with a != b: they
average to zero over the random signs, so at finite M they are the whole
deviation from the true dissipator. This measures how much of that weight
exists, relative to the true dissipator, and tests it against the two failures
that disqualified the bandwidth measure of probe_oq4_accuracy.py:

  Failure 1: within the chains, bandwidth FALLS with dimension while the error
             RISES. Does the cross-term ratio point the right way?
  Failure 2: the magnitude does not carry across families -- the oscillator's
             fit under-predicts the chains by 40x. Does the ratio close it?

Verdict as of 2026-08-10: fixes the direction, does not fix the magnitude, and
inverts A against B. See BENCHMARKS.md section 2.6.

||c_a rho c_b^dag||_F^2 = Tr[(rho c_a^dag c_a rho) c_b^dag c_b], so the whole
N_L x N_L table is one matrix product of two (N_L, d^2) stacks -- the N_L^2 d^2
tensor is never formed. That is what makes dimension 64 at N_L = 2017 feasible.

Run:  python probe_oq4_crossterm.py       (~22 min on a laptop)
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import qutip
from common import (build_mixed_field_chain, build_spin_chain,
                    build_oscillator_bath, build_davies_operators)


def stacked_dissipator(ops, rho):
    tmp = ops @ rho                                   # (N,d,d)
    gain = np.einsum("aij,akj->ik", tmp, ops.conj(), optimize=True)
    anti = np.einsum("aji,ajk->ik", ops.conj(), ops, optimize=True)   # sum c^dag c
    return gain - 0.5 * (anti @ rho + rho @ anti)


def evolve(H_diag, ops, rho0, t_end=1.0, steps=200):
    """RK4 on the full Lindblad equation, in the energy eigenbasis."""
    h = t_end / steps
    E = H_diag[:, None] - H_diag[None, :]             # -i[H,rho] = -i E_ab rho_ab
    rho = rho0.astype(complex).copy()

    def rhs(r):
        return -1j * E * r + stacked_dissipator(ops, r)

    for _ in range(steps):
        k1 = rhs(rho)
        k2 = rhs(rho + 0.5 * h * k1)
        k3 = rhs(rho + 0.5 * h * k2)
        k4 = rhs(rho + h * k3)
        rho = rho + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return rho


def cross_ratio(ops, rho):
    """(live fraction, cross weight / true weight) without forming N^2 d^2."""
    n, d, _ = ops.shape
    A = np.einsum("aji,ajk->aik", ops.conj(), ops, optimize=True)   # c^dag c
    B = rho @ A @ rho                                              # rho A rho
    # Tr[B_a A_b] = sum_ij B_a[i,j] A_b[j,i]
    norms2 = (B.reshape(n, -1) @ np.transpose(A, (0, 2, 1)).reshape(n, -1).T).real
    norms2 = np.clip(norms2, 0.0, None)
    norms = np.sqrt(norms2)
    true = np.diag(norms).copy()
    cross = norms - np.diag(true)
    live = int((cross > 1e-10 * true.max()).sum())
    return live / max(n * (n - 1), 1), float(cross.sum() / true.sum())


CASES = [
    ("mixed chain", build_mixed_field_chain, [4, 5, 6]),
    ("oscillator ", build_oscillator_bath,   [8, 16, 32]),
    ("TFIM chain ", lambda n: build_spin_chain(n, g=0.0), [4, 5, 6]),
]

print(f"{'system':<13}{'dim':>5}{'N_L':>7}{'live% t=0':>11}{'ratio t=0':>11}"
      f"{'live% t=1':>11}{'ratio t=1':>11}{'secs':>8}")
for label, build, sizes in CASES:
    for size in sizes:
        t0 = time.perf_counter()
        H, X, psi0 = build(size)
        c_ops = build_davies_operators(H, X)
        E, V = np.linalg.eigh(H.full())
        ops = np.ascontiguousarray(
            np.array([V.conj().T @ c.full() @ V for c in c_ops]))
        rho0 = V.conj().T @ qutip.ket2dm(psi0).full() @ V

        live0, ratio0 = cross_ratio(ops, rho0)
        rho1 = evolve(E, ops, rho0)
        trace_err = abs(np.trace(rho1).real - 1.0)
        live1, ratio1 = cross_ratio(ops, rho1)
        dt = time.perf_counter() - t0
        flag = "" if trace_err < 1e-6 else f"  [trace off by {trace_err:.1e}]"
        print(f"{label:<13}{H.shape[0]:>5}{len(c_ops):>7}{100*live0:>10.1f}%"
              f"{ratio0:>11.2f}{100*live1:>10.1f}%{ratio1:>11.2f}{dt:>8.1f}{flag}",
              flush=True)
