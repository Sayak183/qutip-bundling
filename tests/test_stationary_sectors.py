"""The Gibbs state is stationary for the Davies generator, but not uniquely so.

Result 5 originally scored its long-time check against the global Gibbs state on
the grounds that detailed balance makes it stationary. Detailed balance does
hold -- the generator annihilates the Gibbs state to machine precision -- but a
state being stationary does not make it the LIMIT. A Davies operator is built as
Pi_e X Pi_e', so two energy levels are dynamically connected exactly when
<e|X|e'> is non-zero; where that graph is disconnected the space splits into
sectors, each sector's population is separately conserved, and the limit is
Gibbs *within* each sector, weighted by where rho0 started.

That error survived review because the number it produced looked good: at
dimension 256 the run sat 0.29 s.e.m. from global Gibbs, and 10.26 s.e.m. from
the state the generator actually reaches. These tests pin the three facts that
distinguish those cases, so the distinction cannot be lost again silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

common = pytest.importorskip(
    "common", reason="benchmark scripts require benchmarks/ on sys.path")
plot_extreme = pytest.importorskip("plot_extreme_dimension")

from qutip_bundling import davies_operators  # noqa: E402


def _generator(build, size):
    H, X, psi0 = build(size)
    c_ops = davies_operators(
        H, X, common.gamma, degeneracy_tol=common.DAVIES_DEGENERACY_TOL)
    return H, X, psi0, c_ops


def _gibbs(H):
    energies, vecs = H.eigenstates()
    w = np.exp(-(np.real(energies) - np.real(energies).min()) / common.KT)
    w /= w.sum()
    return sum(wi * (v * v.dag()) for wi, v in zip(w, vecs))


# --- 1. the construction is right: Gibbs IS stationary --------------------

@pytest.mark.parametrize("name,size", [("mixed_chain", 3),
                                       ("oscillator_bath", 8),
                                       ("spin_chain", 3)])
def test_gibbs_state_is_stationary(name, size):
    """L[rho_Gibbs] = 0. If this fails, detailed balance is broken and the
    problem is the construction, not the uniqueness of the limit."""
    build = {"mixed_chain": common.build_mixed_field_chain,
             "oscillator_bath": common.build_oscillator_bath,
             "spin_chain": common.build_spin_chain}[name]
    H, _X, _psi0, c_ops = _generator(build, size)
    L = qutip.liouvillian(H, c_ops)
    drho = qutip.vector_to_operator(L * qutip.operator_to_vector(_gibbs(H)))
    assert float(drho.norm("max")) < 1e-10


# --- 2. but it is not always the UNIQUE stationary state ------------------

def test_oscillator_generator_is_ergodic():
    """One sector, so Gibbs is the limit and the two targets coincide."""
    H, X, _psi0, c_ops = _generator(common.build_oscillator_bath, 8)
    L = qutip.liouvillian(H, c_ops).full()
    sv = np.linalg.svd(L, compute_uv=False)
    assert int(np.sum(sv < 1e-9 * sv[0])) == 1


def test_mixed_chain_generator_is_not_ergodic():
    """Two stationary states, at every size tested. This is the fact the
    original check assumed away."""
    for size in (3, 4):
        H, X, _psi0, c_ops = _generator(common.build_mixed_field_chain, size)
        L = qutip.liouvillian(H, c_ops).full()
        sv = np.linalg.svd(L, compute_uv=False)
        kernel = int(np.sum(sv < 1e-9 * sv[0]))
        assert kernel == 2, f"size {size}: kernel {kernel}, expected 2"


# --- 3. the sector target predicts the limit; the global one does not -----

def test_sector_resolved_target_matches_the_actual_limit():
    """Propagate the UNBUNDLED generator to its stationary state and check
    which target it lands on. The mixed chain at dim 16 settles at -4.982237,
    while the global Gibbs energy is -4.968520."""
    H, X, psi0, c_ops = _generator(common.build_mixed_field_chain, 4)
    rho0 = qutip.ket2dm(psi0)
    tl = np.linspace(0.0, 200.0, 400)
    curve = np.real(qutip.mesolve(H, rho0, tl, c_ops=c_ops, e_ops=[H]).expect[0])

    # settled, so the endpoint is the limit rather than a snapshot
    assert abs(curve[-1] - curve[len(curve) // 2]) < 1e-8

    energies = np.real(H.eigenenergies())
    w = np.exp(-(energies - energies.min()) / common.KT)
    w /= w.sum()
    global_gibbs = float(np.dot(w, energies))

    doc = {"meta": {"params": {"system": "mixed_chain", "size": 4}}}
    sector, n_sectors = plot_extreme.sector_resolved_energy(doc)

    assert n_sectors == 2
    assert sector is not None
    # The sector target is right; the global one is off by a thousand times the
    # tolerance the sector target meets.
    assert abs(curve[-1] - sector) < 1e-5
    assert abs(curve[-1] - global_gibbs) > 1e-3


def test_ergodic_system_reports_no_separate_sector_target():
    """Where the generator is ergodic the two targets coincide, and the helper
    says so by returning None rather than a duplicate number -- so a plot can
    draw one line instead of two identical ones."""
    doc = {"meta": {"params": {"system": "oscillator_bath", "size": 8}}}
    sector, n_sectors = plot_extreme.sector_resolved_energy(doc)
    assert n_sectors == 1
    assert sector is None
