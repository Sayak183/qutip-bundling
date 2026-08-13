"""Regression tests for strict Davies grouping in degenerate sectors."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip

from qutip_bundling import davies_operators


def thermal_gamma(kT: float):
    """Detailed-balance rate using the package's positive-downward convention."""

    def gamma(omega: float) -> float:
        if abs(omega) < 1e-12:
            return 0.3 * kT
        return 0.3 * omega / (1.0 - math.exp(-omega / kT))

    return gamma


def dissipator(c_ops):
    return sum(qutip.lindblad_dissipator(c_op) for c_op in c_ops)


def benchmark_spin_chain(n_sites):
    J, h = 1.0, 0.6
    sx, sz, identity = qutip.sigmax(), qutip.sigmaz(), qutip.qeye(2)

    def op(single_site_op, site):
        return qutip.tensor(
            [
                single_site_op if index == site else identity
                for index in range(n_sites)
            ]
        )

    H = sum(
        -J * op(sz, site) * op(sz, site + 1)
        for site in range(n_sites - 1)
    )
    H += sum(-h * op(sx, site) for site in range(n_sites))
    X = sum(op(sx, site) for site in range(n_sites))
    return H, X


def test_harmonic_ladder_has_one_operator_per_bohr_frequency():
    """Equal ladder gaps must combine into a and a-dagger, not pairwise jumps."""
    dim = 5
    a = qutip.destroy(dim)
    H = qutip.num(dim)
    X = a + a.dag()

    c_ops, (bare, omegas) = davies_operators(
        H, X, lambda omega: 1.0, return_bare=True
    )

    assert len(c_ops) == len(bare) == 2
    np.testing.assert_allclose(omegas, [-1.0, 1.0], atol=1e-12)
    assert (bare[0] - a.dag()).norm() < 1e-12
    assert (bare[1] - a).norm() < 1e-12


def test_grouped_dissipator_keeps_cross_terms_pairwise_form_drops():
    """This directly detects the historical pairwise-construction bug."""
    dim = 4
    a = qutip.destroy(dim)
    H = qutip.num(dim)
    X = a + a.dag()
    grouped = davies_operators(H, X, lambda omega: 1.0)

    pairwise = []
    for n in range(1, dim):
        down = math.sqrt(n) * qutip.basis(dim, n - 1) * qutip.basis(dim, n).dag()
        pairwise.extend([down, down.dag()])

    assert (dissipator(grouped) - dissipator(pairwise)).norm() > 1.0


def test_degenerate_projectors_are_covariant_under_subspace_rotations():
    """Rotating bases inside degenerate eigenspaces cannot change A(omega)."""
    H = qutip.Qobj(np.diag([0.0, 0.0, 1.0, 1.0]))
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    X = qutip.Qobj(raw + raw.conj().T)

    theta, phi = 0.37, -0.52
    r0 = np.array(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]],
        dtype=complex,
    )
    r1 = np.array(
        [[math.cos(phi), -math.sin(phi)],
         [math.sin(phi), math.cos(phi)]],
        dtype=complex,
    )
    U = qutip.Qobj(
        np.block(
            [
                [r0, np.zeros((2, 2))],
                [np.zeros((2, 2)), r1],
            ]
        )
    )
    H_rot = U * H * U.dag()
    X_rot = U * X * U.dag()

    _, (bare, omegas) = davies_operators(
        H, X, lambda omega: 1.0, return_bare=True
    )
    _, (bare_rot, omegas_rot) = davies_operators(
        H_rot, X_rot, lambda omega: 1.0, return_bare=True
    )

    np.testing.assert_allclose(omegas_rot, omegas, atol=1e-12)
    for original, rotated in zip(bare, bare_rot):
        assert (rotated - U * original * U.dag()).norm() < 1e-10


def test_spin_chain_sectors_are_unique_and_gibbs_state_is_stationary():
    """Regression for the symmetry-degenerate benchmark spin chain."""
    n_sites = 4
    kT = 0.5
    H, X = benchmark_spin_chain(n_sites)

    c_ops, (bare, omegas) = davies_operators(
        H, X, thermal_gamma(kT), return_bare=True
    )

    assert len(omegas) > 0
    assert np.all(np.diff(omegas) > 1e-10)
    for operator, omega in zip(bare, omegas):
        assert (H * operator - operator * H + omega * operator).norm() < 1e-8

    gibbs = (-H / kT).expm()
    gibbs /= gibbs.tr()
    liouvillian = qutip.liouvillian(H, c_ops)
    stationary_residual = qutip.vector_to_operator(
        liouvillian * qutip.operator_to_vector(gibbs)
    )
    assert stationary_residual.norm() < 1e-9

    identity_full = qutip.qeye(H.dims[0])
    trace_residual = qutip.vector_to_operator(
        liouvillian.dag() * qutip.operator_to_vector(identity_full)
    )
    assert trace_residual.norm() < 1e-10


def test_degeneracy_tolerance_controls_only_numerical_grouping():
    """A user can resolve or merge a tiny gap by choosing the tolerance.

    Asserted against explicit tolerances on both sides rather than against
    whatever the default happens to be. The default moved from 1e-10 to 1e-5 on
    2026-08-12 and silently inverted this test's outcome, which is exactly the
    coupling worth removing: the knob's behaviour is the invariant here, not the
    shipped value of the knob.
    """
    H = qutip.Qobj(np.diag([0.0, 1.0, 2.0 + 1e-8]))
    X = qutip.destroy(3) + qutip.create(3)

    resolved = davies_operators(H, X, lambda omega: 1.0, degeneracy_tol=1e-10)
    merged = davies_operators(H, X, lambda omega: 1.0, degeneracy_tol=1e-7)

    assert len(resolved) == 4, "a tolerance below the splitting must keep it split"
    assert len(merged) == 2, "a tolerance above the splitting must merge it"


def test_shipped_default_resolves_a_1e_8_splitting():
    """The shipped default is tight enough to keep a 1e-8 gap distinct.

    At 1e-10 the default sits below such a splitting, so it resolves it into
    four operators rather than merging to two. A brief experiment with 1e-5 on
    2026-08-12 inverted this; see tests/test_degeneracy_tolerance.py for the
    sweep that argued it back.
    """
    H = qutip.Qobj(np.diag([0.0, 1.0, 2.0 + 1e-8]))
    X = qutip.destroy(3) + qutip.create(3)

    assert len(davies_operators(H, X, lambda omega: 1.0)) == 4


def test_negative_degeneracy_tolerance_is_rejected():
    with pytest.raises(ValueError, match="degeneracy_tol"):
        davies_operators(
            qutip.num(2),
            qutip.sigmax(),
            lambda omega: 1.0,
            degeneracy_tol=-1.0,
        )


@pytest.mark.parametrize(
    ("n_sites", "expected_count"),
    [(4, 13), (5, 21), (6, 31), (7, 43)],
)
def test_spin_chain_roundoff_floor_has_stable_sector_counts(
    n_sites, expected_count
):
    H, X = benchmark_spin_chain(n_sites)
    c_ops = davies_operators(H, X, lambda omega: 1.0)

    assert len(c_ops) == expected_count


def test_roundoff_floor_scales_with_physical_coupling():
    H, X = benchmark_spin_chain(4)
    scale = 1e-9

    _, (bare, omegas) = davies_operators(
        H, X, lambda omega: 1.0, return_bare=True
    )
    _, (scaled_bare, scaled_omegas) = davies_operators(
        H, scale * X, lambda omega: 1.0, return_bare=True
    )

    np.testing.assert_allclose(scaled_omegas, omegas, atol=1e-12)
    assert len(scaled_bare) == len(bare)
    for original, scaled in zip(bare, scaled_bare):
        np.testing.assert_allclose(
            scaled.full(),
            scale * original.full(),
            rtol=1e-11,
            atol=np.finfo(float).eps * scale,
        )


def test_negative_coupling_threshold_is_rejected():
    with pytest.raises(ValueError, match="coupling_threshold"):
        davies_operators(
            qutip.num(2),
            qutip.sigmax(),
            lambda omega: 1.0,
            coupling_threshold=-1.0,
        )


def test_zero_coupling_operator_produces_no_sectors():
    H = qutip.num(3)
    X = 0.0 * qutip.qeye(3)

    assert davies_operators(H, X, lambda omega: 1.0) == []
