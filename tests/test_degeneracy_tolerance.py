"""Pin the consequences of ``DAVIES_DEGENERACY_TOL``.

The default moved from ``1e-10`` to ``1e-5`` on 2026-08-12. That is a
physics-affecting knob: it decides which Bohr frequencies the construction
treats as one bath channel, so it sets ``N_L``, which every cost claim in
BENCHMARKS.md is built on.

The change was measured before being accepted and is small, but it is *not*
free: at these sizes ``1e-5`` sits inside the real spectrum rather than safely
below it, so a few genuinely distinct frequencies are merged and ``N_L`` becomes
weakly tolerance-dependent. These tests record both facts so a future change to
the tolerance -- or a drift in the grouping code -- fails loudly instead of
quietly moving the headline numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from common import (  # noqa: E402
    DAVIES_DEGENERACY_TOL,
    build_mixed_field_chain,
    build_oscillator_bath,
    build_spin_chain,
    gamma,
)
from qutip_bundling import davies_operators  # noqa: E402


def test_shipped_tolerance_is_the_documented_one():
    """BENCHMARKS.md quotes this value; keep them from drifting apart."""
    assert DAVIES_DEGENERACY_TOL == 1e-5


@pytest.mark.parametrize("build, size, expected", [
    (lambda n: build_spin_chain(n, g=0.0), 4, 13),
    (lambda n: build_spin_chain(n, g=0.0), 5, 21),
    (lambda n: build_spin_chain(n, g=0.0), 6, 31),
    (build_mixed_field_chain, 4, 121),
    (build_mixed_field_chain, 5, 513),
    (build_mixed_field_chain, 6, 2015),      # 2017 at the old 1e-10
    # build_oscillator_bath takes Fock levels; the Hilbert dimension is twice
    # that, because each oscillator level carries the spin's two states.
    (build_oscillator_bath, 8, 128),      # dim 16
    (build_oscillator_bath, 16, 408),     # dim 32
    (build_oscillator_bath, 32, 890),     # dim 64
])
def test_operator_counts_at_the_shipped_tolerance(build, size, expected):
    """The N_L values BENCHMARKS.md quotes, at the shipped default."""
    H, X, _ = build(size)
    assert len(davies_operators(H, X, gamma,
                                degeneracy_tol=DAVIES_DEGENERACY_TOL)) == expected


def test_loosening_the_tolerance_barely_moves_the_dissipator():
    """Going 1e-10 -> 1e-5 must stay numerically irrelevant.

    Measured at 4.7e-9 relative on the system that changes most (System B at
    dimension 64, where two of 2,017 operators merge). A dissipative solve here
    costs hours, so the state is evolved unitarily instead -- enough to leave
    the initial product state without needing the full dynamics.
    """
    H, X, psi0 = build_mixed_field_chain(6)
    strict = np.array([L.full() for L in
                       davies_operators(H, X, gamma, degeneracy_tol=1e-10)])
    loose = np.array([L.full() for L in
                      davies_operators(H, X, gamma, degeneracy_tol=1e-5)])
    assert len(strict) == 2017 and len(loose) == 2015

    energies, vectors = np.linalg.eigh(H.full())
    phase = vectors @ np.diag(np.exp(-1j * energies)) @ vectors.conj().T
    rho = phase @ qutip.ket2dm(psi0).full() @ phase.conj().T

    def dissipator(ops):
        gain = np.einsum("aij,akj->ik", ops @ rho, ops.conj(), optimize=True)
        anti = np.einsum("aji,ajk->ik", ops.conj(), ops, optimize=True)
        return gain - 0.5 * (anti @ rho + rho @ anti)

    reference = dissipator(strict)
    relative = (np.linalg.norm(dissipator(loose) - reference)
                / np.linalg.norm(reference))
    assert relative < 1e-7, f"tolerance change is no longer negligible: {relative:.2e}"


def test_tolerance_sits_inside_the_spectrum_at_benchmark_sizes():
    """Record the uncomfortable fact, so it is not rediscovered as a surprise.

    The smallest separation between two genuinely distinct driven frequencies
    falls below 1e-5 for System B by dimension 64. The merging is defensible --
    frequencies that close are unresolvable by the bath over the benchmark
    window -- but it means N_L is tolerance-dependent, and a denser spectrum
    would need a tighter default.
    """
    H, X, _ = build_mixed_field_chain(6)
    energies, vectors = np.linalg.eigh(H.full())
    coupling = vectors.conj().T @ X.full() @ vectors
    driven = np.sort([energies[b] - energies[a]
                      for a in range(len(energies)) for b in range(len(energies))
                      if abs(coupling[a, b]) > 1e-12])
    separations = np.diff(driven)
    smallest = separations[separations > 1e-14].min()

    assert smallest < DAVIES_DEGENERACY_TOL, (
        "the tolerance is now safely below the spectrum -- good news, but "
        "update the caveat in BENCHMARKS.md 2.1 which says otherwise"
    )
    assert smallest == pytest.approx(2.467e-06, rel=0.05)
