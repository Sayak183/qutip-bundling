"""Regression test pinning the vectorized bundle_from_phases against the
explicit reference summation. The vectorized path must reproduce the loop
result to floating-point tolerance for every M and operator count.
"""
import math

import numpy as np
import qutip
import pytest

from qutip_bundling.operators import bundle_from_phases, random_phases


def _reference_bundle(c_ops, phases):
    """The explicit definition R_m = (1/sqrt(M)) sum_a phases[m,a] c_ops[a]."""
    M = phases.shape[0]
    inv = 1.0 / math.sqrt(M)
    out = []
    for m in range(M):
        acc = complex(phases[m, 0]) * c_ops[0]
        for a in range(1, len(c_ops)):
            acc = acc + complex(phases[m, a]) * c_ops[a]
        out.append(inv * acc)
    return out


def _random_ops(n_l, dim, seed):
    rng = np.random.default_rng(seed)
    return [qutip.Qobj(rng.standard_normal((dim, dim))
                       + 1j * rng.standard_normal((dim, dim)))
            for _ in range(n_l)]


@pytest.mark.parametrize("n_l,dim", [(3, 2), (8, 4), (20, 4), (40, 8)])
@pytest.mark.parametrize("M", [1, 2, 4, 8])
def test_vectorized_matches_reference(n_l, dim, M):
    c_ops = _random_ops(n_l, dim, seed=n_l * 100 + M)
    phases = random_phases(M, n_l, rng=7)
    fast = bundle_from_phases(c_ops, phases)
    ref = _reference_bundle(c_ops, phases)
    assert len(fast) == len(ref) == M
    for a, b in zip(fast, ref):
        assert a.dims == b.dims
        assert np.max(np.abs(a.full() - b.full())) < 1e-12


def test_normalization_scales_as_inv_sqrt_m():
    """A single-operator bundle with unit phase must equal c/sqrt(M)."""
    c = qutip.sigmax()
    for M in (1, 4, 9):
        phases = np.ones((M, 1))
        out = bundle_from_phases([c], phases)
        assert np.max(np.abs(out[0].full() - c.full() / math.sqrt(M))) < 1e-12
