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


# --- 3b. bundling CANNOT mix sectors -------------------------------------

def test_a_bundle_is_block_diagonal_in_the_sectors():
    """Result 5 explained its dimension-256 endpoint by saying a bundle "mixes
    operators from different sectors, so the bundled generator connects what the
    exact one cannot". That is algebraically impossible and it stood for weeks.

    Every c_alpha is block-diagonal in the sectors, so any linear combination of
    them is too. Sector populations are conserved at every M, and whatever the
    dim-256 run was doing, leaking between sectors was not it.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    from qutip_bundling import bundle

    H, X, _psi0, c_ops = _generator(common.build_mixed_field_chain, 4)
    _energies, vecs = H.eigenstates()
    U = np.column_stack([v.full().ravel() for v in vecs])

    coupling = np.abs(np.array([[complex(a.dag() * X * b) for b in vecs]
                                for a in vecs]))
    n_sectors, labels = connected_components(
        csr_matrix(coupling > 1e-10), directed=False)
    assert n_sectors == 2
    cross = labels[:, None] != labels[None, :]

    for c in c_ops:                       # the exact operators
        block = np.abs(U.conj().T @ c.full() @ U)
        assert block[cross].max() < 1e-12

    for seed in range(8):                 # and any bundle built from them
        for m in (4, 16, 64):
            for R in bundle(c_ops, M=m, rng=seed):
                block = np.abs(U.conj().T @ R.full() @ U)
                assert block[cross].max() < 1e-12, (
                    f"M={m} seed={seed}: a bundle acquired a cross-sector "
                    f"element of {block[cross].max():.2e}")


def test_the_bundled_stationary_state_tends_to_the_sector_limit():
    """The corrected mechanism: finite M perturbs detailed balance WITHIN each
    sector, so the stationary state of a bundled draw sits near the sector limit
    and moves toward it -- it does not drift to the global Gibbs state."""
    from qutip_bundling import bundle

    H, _X, _psi0, c_ops = _generator(common.build_mixed_field_chain, 4)
    Hf = H.full()

    energies = np.real(H.eigenenergies())
    w = np.exp(-(energies - energies.min()) / common.KT)
    w /= w.sum()
    global_gibbs = float(np.dot(w, energies))
    sector = plot_extreme.sector_resolved_energy(
        {"meta": {"params": {"system": "mixed_chain", "size": 4}}})[0]

    stationary = []
    for seed in range(8):
        L = qutip.liouvillian(H, bundle(c_ops, M=32, rng=seed)).full()
        rho = np.linalg.svd(L)[2][-1].conj().reshape(Hf.shape)
        stationary.append(float(np.real(np.trace(rho / np.trace(rho) @ Hf))))
    mean = float(np.mean(stationary))

    assert abs(mean - sector) < abs(mean - global_gibbs), (
        f"bundled stationary energy {mean:.6f} is closer to global Gibbs "
        f"{global_gibbs:.6f} than to the sector limit {sector:.6f}")


# --- 4. the sectors are reflection parity, not spin-flip parity -----------

def _reflection(n):
    """|b0..b_{n-1}> -> |b_{n-1}..b0> as an explicit permutation matrix.
    Do NOT build this by permuting an identity Qobj -- permuting the identity
    returns the identity, which silently passes every commutator test."""
    d = 2 ** n
    R = np.zeros((d, d))
    for k in range(d):
        bits = [(k >> (n - 1 - j)) & 1 for j in range(n)]
        R[sum(b << (n - 1 - j) for j, b in enumerate(bits[::-1])), k] = 1.0
    return qutip.Qobj(R, dims=[[2] * n, [2] * n])


def test_reflection_operator_is_not_the_identity():
    """Guards the trap above: the reflection must actually move something."""
    assert (_reflection(3) - qutip.qeye([2, 2, 2])).norm("max") > 0.5


@pytest.mark.parametrize("name,n,sizes", [("spin_chain", 4, [6, 4, 4, 1, 1]),
                                          ("mixed_chain", 4, [10, 6]),
                                          ("mixed_chain", 3, [6, 2])])
def test_sectors_are_the_connected_components_of_reflection_parity(name, n, sizes):
    """The document names left-right reflection as the cause and quotes these
    sizes. Both halves are pinned: the component sizes, and the fact that each
    component is uniform in reflection parity."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    build = {"spin_chain": common.build_spin_chain,
             "mixed_chain": common.build_mixed_field_chain}[name]
    H, X, _psi0, _c = _generator(build, n)
    _energies, vecs = H.eigenstates()

    Xm = np.abs(np.array([[complex(a.dag() * X * b) for b in vecs]
                          for a in vecs]))
    ncomp, labels = connected_components(csr_matrix(Xm > 1e-10), directed=False)
    assert sorted(np.bincount(labels))[::-1] == sizes

    R = _reflection(n)
    parity = np.sign(np.round(
        [float(np.real(complex(v.dag() * R * v))) for v in vecs], 6))
    for c in range(ncomp):
        assert len(set(parity[labels == c])) == 1, "component mixes parity"


def test_longitudinal_field_breaks_spin_flip_parity_but_not_reflection():
    """Why System A fragments further than System B. The obvious guess --
    spin-flip parity -- does NOT survive in B, so it cannot be the cause there.
    X commutes with both in both systems; it is H that loses one."""
    n = 4
    P, R = qutip.tensor([qutip.sigmax()] * n), _reflection(n)

    Ha, Xa, _ = common.build_spin_chain(n)
    Hb, Xb, _ = common.build_mixed_field_chain(n)

    assert float((Ha * P - P * Ha).norm("max")) < 1e-12       # kept at g=0
    assert float((Hb * P - P * Hb).norm("max")) == pytest.approx(3.2, abs=1e-9)
    for H, X in ((Ha, Xa), (Hb, Xb)):
        assert float((H * R - R * H).norm("max")) < 1e-12
        assert float((X * R - R * X).norm("max")) < 1e-12
        assert float((X * P - P * X).norm("max")) < 1e-12


def test_ergodic_system_reports_no_separate_sector_target():
    """Where the generator is ergodic the two targets coincide, and the helper
    says so by returning None rather than a duplicate number -- so a plot can
    draw one line instead of two identical ones."""
    doc = {"meta": {"params": {"system": "oscillator_bath", "size": 8}}}
    sector, n_sectors = plot_extreme.sector_resolved_energy(doc)
    assert n_sectors == 1
    assert sector is None
