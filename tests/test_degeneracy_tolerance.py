"""Pin ``DAVIES_DEGENERACY_TOL`` and the plateau that justifies its value.

The tolerance decides which Bohr frequencies the construction treats as one
bath channel, so it sets ``N_L``, which every cost claim in BENCHMARKS.md rests
on. It is therefore a physics-affecting knob, not a numerical detail.

It was briefly changed to ``1e-5`` on 2026-08-12 to "absorb chaotic
level-repulsion noise". Measurement showed there was no such noise to absorb --
the count is identical from ``1e-14`` to ``1e-6`` -- and that ``1e-5`` is the
first value where genuine frequencies start merging. It was reverted the same
day. These tests record the sweep so the argument does not have to be
rediscovered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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
    """BENCHMARKS.md 2.1 quotes this value; keep them from drifting apart."""
    assert DAVIES_DEGENERACY_TOL == 1e-10


@pytest.mark.parametrize("build, size, expected", [
    (lambda n: build_spin_chain(n, g=0.0), 4, 13),
    (lambda n: build_spin_chain(n, g=0.0), 5, 21),
    (lambda n: build_spin_chain(n, g=0.0), 6, 31),
    (build_mixed_field_chain, 4, 121),
    (build_mixed_field_chain, 5, 513),
    (build_mixed_field_chain, 6, 2017),
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


@pytest.mark.parametrize("build, size", [
    (lambda n: build_spin_chain(n, g=0.0), 6),
    (build_mixed_field_chain, 5),
    (build_mixed_field_chain, 6),
    (build_oscillator_bath, 16),
    (build_oscillator_bath, 32),
])
def test_count_is_flat_across_the_plateau(build, size):
    """N_L must not depend on where in 1e-14..1e-6 the tolerance sits.

    This is the property that makes N_L a statement about the model rather than
    about a constant we chose. If this test starts failing, the spectrum has
    grown dense enough that the tolerance is resolving real structure, and the
    default needs tightening -- not loosening.
    """
    H, X, _ = build(size)
    counts = {tol: len(davies_operators(H, X, gamma, degeneracy_tol=tol))
              for tol in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6)}
    assert len(set(counts.values())) == 1, f"count varies across the plateau: {counts}"


def test_loosening_past_the_plateau_starts_merging_real_frequencies():
    """Record where the cliff is, so nobody wanders over it again.

    System B at dimension 64 is the first system to erode, and it does so at
    1e-5 -- one step off the plateau. By 1e-3 a tenth of the operators are gone.
    """
    H, X, _ = build_mixed_field_chain(6)
    counts = {tol: len(davies_operators(H, X, gamma, degeneracy_tol=tol))
              for tol in (1e-6, 1e-5, 1e-4, 1e-3)}
    assert counts[1e-6] == 2017
    assert counts[1e-5] == 2015
    assert counts[1e-4] == 1991
    assert counts[1e-3] == 1813
    assert counts[1e-3] < 0.92 * counts[1e-6], "the cliff has moved; re-check the default"


def test_tolerance_stays_below_the_real_spectrum():
    """The shipped value must sit under the smallest genuine gap separation.

    At 1e-10 it does, by four orders of magnitude even on the densest system
    benchmarked. This is the guarantee the 1e-5 experiment gave up: there, the
    tolerance sat *above* the 2.5e-6 separation and merged real frequencies.
    """
    H, X, _ = build_mixed_field_chain(6)
    energies, vectors = np.linalg.eigh(H.full())
    coupling = vectors.conj().T @ X.full() @ vectors
    driven = np.sort([energies[b] - energies[a]
                      for a in range(len(energies)) for b in range(len(energies))
                      if abs(coupling[a, b]) > 1e-12])
    separations = np.diff(driven)
    smallest = separations[separations > 1e-14].min()

    assert smallest == pytest.approx(2.467e-06, rel=0.05)
    assert DAVIES_DEGENERACY_TOL < smallest / 100, (
        f"tolerance {DAVIES_DEGENERACY_TOL:.0e} is within two orders of the "
        f"smallest real gap separation {smallest:.2e} -- it will start merging "
        f"physical frequencies"
    )
