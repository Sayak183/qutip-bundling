"""Tests for qutip_bundling.

Run with:  pytest tests/
"""

import math
import numpy as np
import qutip
import pytest

from qutip_bundling import (
    BundledOps,
    build_collapse_ops,
    bundle,
    bundle_from_phases,
    davies_operators,
    lamb_shift_hamiltonian,
    mesolve_ensemble,
    mesolve_jackknife,
    prepare_bundled_dynamics,
    random_phases,
    rk4_mesolve,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_bare_operators(n_ops, dim, seed=0):
    """Bare Lindblad operators L_alpha + aligned Bohr frequencies."""
    rng = np.random.default_rng(seed)
    L_ops, omegas = [], []
    for _ in range(n_ops):
        m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        L_ops.append(qutip.Qobj(0.3 * m))
        omegas.append(rng.uniform(-2.0, 2.0))
    return L_ops, np.asarray(omegas)


def random_collapse_ops(n_ops, dim, seed=0):
    """Already-scaled collapse operators (the input to ``bundle``)."""
    L_ops, _ = make_bare_operators(n_ops, dim, seed=seed)
    return L_ops


def full_dissipator(c_ops):
    return sum(qutip.lindblad_dissipator(c) for c in c_ops)


# --------------------------------------------------------------------------
# random_phases
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dist", ["phase", "pm1", "uniform"])
def test_random_phases_moments(dist):
    r = random_phases(2000, 5, distribution=dist, rng=42)
    assert r.shape == (2000, 5)
    assert abs(r.mean()) < 0.1
    assert abs(np.mean(np.abs(r) ** 2) - 1.0) < 0.1


def test_random_phases_reproducible():
    a = random_phases(10, 4, rng=7)
    b = random_phases(10, 4, rng=7)
    np.testing.assert_array_equal(a, b)


def test_random_phases_rejects_bad_distribution():
    with pytest.raises(ValueError):
        random_phases(4, 4, distribution="gaussian")


# --------------------------------------------------------------------------
# build_collapse_ops
# --------------------------------------------------------------------------
def test_build_collapse_ops_with_callable():
    L_ops, omegas = make_bare_operators(8, 4)
    c = build_collapse_ops(L_ops, omegas, gamma=lambda w: abs(w) + 0.1)
    assert len(c) == 8
    # Operator norms must scale by sqrt(gamma) relative to the bare ops
    sg = np.sqrt(np.abs(omegas) + 0.1)
    for L, scaled, s in zip(L_ops, c, sg):
        assert abs((scaled - s * L).norm()) < 1e-10


def test_build_collapse_ops_with_array():
    L_ops, omegas = make_bare_operators(6, 4)
    gvals = np.full(omegas.shape, 0.4)
    c = build_collapse_ops(L_ops, omegas, gamma=gvals)
    assert len(c) == 6
    expected_factor = np.sqrt(0.4)
    for L, scaled in zip(L_ops, c):
        assert abs((scaled - expected_factor * L).norm()) < 1e-10


def test_build_collapse_ops_threshold_drops_small():
    L_ops, omegas = make_bare_operators(6, 4)
    # gamma=0 for the first three, large for the rest
    gvals = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    c = build_collapse_ops(L_ops, omegas, gamma=gvals)
    assert len(c) == 3


def test_build_collapse_ops_threshold_uses_product_not_rate():
    """`threshold` compares against sqrt(gamma)*||L||, not sqrt(gamma) alone.
    An operator with a small rate but large coupling can survive a threshold
    that the rate alone would fall below -- matching davies_operators."""
    dim = 4
    big = qutip.Qobj(5.0 * qutip.create(dim).full())     # large-norm operator
    small = qutip.Qobj(0.01 * qutip.create(dim).full())  # small-norm operator
    L_ops = [big, small]
    omegas = np.array([1.0, 1.0])
    gvals = np.array([0.04, 0.04])                        # sqrt(gamma) = 0.2

    # Rate alone (0.2) is below 0.5, so the OLD behavior would drop both.
    # Product weight is 0.2*||L||: large for `big`, tiny for `small`.
    c = build_collapse_ops(L_ops, omegas, gamma=gvals, threshold=0.5)
    assert len(c) == 1                                    # only `big` survives
    assert c[0].norm() == pytest.approx(0.2 * big.norm())


def test_build_collapse_ops_threshold_matches_davies():
    """The same threshold selects the same operators whether you go through
    davies_operators or rebuild via build_collapse_ops."""
    dim = 5
    H = qutip.num(dim) + 0.1 * qutip.num(dim) ** 2
    X = qutip.destroy(dim) + qutip.create(dim)
    gamma = _ohmic(0.4)
    thr = 0.05

    c_davies, (L_ops, omegas) = davies_operators(
        H, X, gamma, threshold=thr, return_bare=True)
    # Rebuild bare->collapse with the same threshold; bare ops here are the
    # post-threshold survivors, so re-thresholding must keep all of them and
    # reproduce the davies collapse operators.
    c_rebuilt = build_collapse_ops(L_ops, omegas, gamma, threshold=thr)
    assert len(c_rebuilt) == len(c_davies)
    D_davies = sum(qutip.lindblad_dissipator(c) for c in c_davies)
    D_rebuilt = sum(qutip.lindblad_dissipator(c) for c in c_rebuilt)
    assert (D_davies - D_rebuilt).norm() < 1e-10


def test_build_collapse_ops_rejects_negative_gamma():
    L_ops, omegas = make_bare_operators(4, 3)
    with pytest.raises(ValueError):
        build_collapse_ops(L_ops, omegas, gamma=lambda w: -1.0)


def test_build_collapse_ops_length_mismatch():
    L_ops, _ = make_bare_operators(4, 3)
    with pytest.raises(ValueError):
        build_collapse_ops(L_ops, [1.0, 2.0], gamma=lambda w: 1.0)


# --------------------------------------------------------------------------
# bundle / bundle_from_phases
# --------------------------------------------------------------------------
def test_bundle_count_and_dims():
    c_ops = random_collapse_ops(30, 4)
    R = bundle(c_ops, M=6, rng=0)
    assert len(R) == 6
    for r in R:
        assert r.dims == c_ops[0].dims
        assert r.shape == c_ops[0].shape


def test_bundle_rejects_bad_M():
    c_ops = random_collapse_ops(5, 3)
    with pytest.raises(ValueError):
        bundle(c_ops, M=0)


def test_bundle_from_phases_shape_check():
    c_ops = random_collapse_ops(5, 3)
    with pytest.raises(ValueError):
        bundle_from_phases(c_ops, np.ones((4, 99)))


@pytest.mark.parametrize("dist", ["phase", "pm1"])
def test_single_operator_dissipator_is_exact(dist):
    """|r|=1 ensembles reproduce the dissipator exactly for one operator."""
    c = random_collapse_ops(1, 4, seed=3)
    D_full = full_dissipator(c)
    for seed in range(5):
        R = bundle(c, M=7, distribution=dist, rng=seed)
        assert (full_dissipator(R) - D_full).norm() < 1e-10


def test_bundled_dissipator_is_unbiased():
    c_ops = random_collapse_ops(40, 4, seed=5)
    D_full = full_dissipator(c_ops)
    rng = np.random.default_rng(0)
    n_samples = 300
    acc = None
    for _ in range(n_samples):
        R = bundle(c_ops, M=8, rng=rng)
        d = full_dissipator(R)
        acc = d if acc is None else acc + d
    rel_err = (acc / n_samples - D_full).norm() / D_full.norm()
    assert rel_err < 0.05


# --------------------------------------------------------------------------
# lamb_shift_hamiltonian
# --------------------------------------------------------------------------
def test_lamb_shift_zero_when_imag_gamma_zero():
    L_ops, omegas = make_bare_operators(8, 4)
    H_LS = lamb_shift_hamiltonian(L_ops, omegas, imag_gamma=0.0)
    assert H_LS.norm() < 1e-12


def test_lamb_shift_zero_callable():
    L_ops, omegas = make_bare_operators(8, 4)
    H_LS = lamb_shift_hamiltonian(L_ops, omegas, imag_gamma=lambda w: 0.0)
    assert H_LS.norm() < 1e-12


def test_lamb_shift_is_hermitian():
    L_ops, omegas = make_bare_operators(12, 5, seed=11)
    H_LS = lamb_shift_hamiltonian(L_ops, omegas, imag_gamma=lambda w: 0.3 * w)
    assert (H_LS - H_LS.dag()).norm() < 1e-10


def test_lamb_shift_independent_of_bundling():
    """The Lamb shift must be built from the bare L_alpha, never bundled."""
    L_ops, omegas = make_bare_operators(15, 4, seed=2)
    H_LS = lamb_shift_hamiltonian(L_ops, omegas, imag_gamma=lambda w: 0.2 * w)

    # Two unrelated bundle calls must not change H_LS at all
    _ = bundle(L_ops, M=4, rng=1)
    _ = bundle(L_ops, M=8, rng=2)
    H_LS_again = lamb_shift_hamiltonian(L_ops, omegas,
                                          imag_gamma=lambda w: 0.2 * w)
    assert (H_LS - H_LS_again).norm() < 1e-12


def test_lamb_shift_matches_explicit_sum():
    """Compare against the literal definition from eq. 4 of the paper."""
    L_ops, omegas = make_bare_operators(10, 4, seed=12)
    imag = lambda w: 0.25 * w
    explicit = None
    for L, w in zip(L_ops, omegas):
        term = imag(w) * (L.dag() * L)
        explicit = term if explicit is None else explicit + term
    explicit = 0.5 * (explicit + explicit.dag())
    H_LS = lamb_shift_hamiltonian(L_ops, omegas, imag_gamma=imag)
    assert (H_LS - explicit).norm() < 1e-10


# --------------------------------------------------------------------------
# prepare_bundled_dynamics
# --------------------------------------------------------------------------
def test_prepare_bundled_dynamics_full_pipeline():
    L_ops, omegas = make_bare_operators(20, 4, seed=8)
    out = prepare_bundled_dynamics(
        L_ops, omegas, gamma=lambda w: 0.1, M=6,
        imag_gamma=lambda w: 0.05 * w, rng=0,
    )
    assert isinstance(out, BundledOps)
    assert len(out.c_ops) == 6
    assert out.M == 6
    assert out.H_lamb_shift.norm() > 0
    assert out.extras["n_full_ops"] == 20


def test_prepare_bundled_dynamics_no_lamb_shift():
    L_ops, omegas = make_bare_operators(15, 4, seed=9)
    out = prepare_bundled_dynamics(
        L_ops, omegas, gamma=lambda w: 0.1, M=4, rng=0,
    )
    assert out.H_lamb_shift.norm() < 1e-12


# --------------------------------------------------------------------------
# Solver wrappers
# --------------------------------------------------------------------------
def test_mesolve_ensemble_tracks_full_dynamics():
    """Ensemble bundle must reproduce full mesolve on a genuinely
    relaxing system to within a small fraction of the dynamic range."""
    dim = 6
    H = qutip.num(dim)
    a = qutip.destroy(dim)
    c_ops = [0.45 * a]                           # strong relaxation channel
    extra, _ = make_bare_operators(20, dim, seed=8)
    c_ops += [0.5 * c for c in extra]
    rho0 = qutip.ket2dm(qutip.basis(dim, dim - 1))
    tlist = np.linspace(0, 8, 40)
    e_ops = [qutip.num(dim)]

    full = qutip.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=e_ops)
    ens = mesolve_ensemble(H, rho0, tlist, c_ops, M=12, e_ops=e_ops,
                            n_realizations=60, rng=0)
    rms = np.sqrt(np.mean((ens.expect[0] - np.real(full.expect[0])) ** 2))
    span = np.ptp(np.real(full.expect[0]))
    assert span > 1.0
    assert rms < 0.1 * span


def test_mesolve_jackknife_requires_even_M():
    L_ops, _ = make_bare_operators(10, 4)
    dim = 4
    H = qutip.num(dim)
    rho0 = qutip.ket2dm(qutip.basis(dim, 1))
    tlist = np.linspace(0, 3, 10)
    e_ops = [qutip.num(dim)]
    with pytest.raises(ValueError):
        mesolve_jackknife(H, rho0, tlist, L_ops, M=7, e_ops=e_ops)


def test_mesolve_jackknife_provides_direct_comparison():
    dim = 4
    H = qutip.num(dim)
    c_ops = random_collapse_ops(15, dim, seed=4)
    rho0 = qutip.ket2dm(qutip.basis(dim, 1))
    tlist = np.linspace(0, 3, 15)
    e_ops = [qutip.num(dim)]
    res = mesolve_jackknife(H, rho0, tlist, c_ops, M=8, e_ops=e_ops,
                              n_realizations=6, rng=0)
    assert res.expect[0].shape == tlist.shape
    assert "direct" in res.extra


# --------------------------------------------------------------------------
# Trace preservation -- a basic CPTP check
# --------------------------------------------------------------------------
def test_bundled_mesolve_preserves_trace():
    dim = 4
    H = qutip.num(dim)
    c_ops = random_collapse_ops(20, dim, seed=2)
    rho0 = qutip.ket2dm(qutip.basis(dim, 1))
    R = bundle(c_ops, M=6, rng=1)
    res = qutip.mesolve(H, rho0, np.linspace(0, 5, 20), c_ops=R)
    for state in res.states:
        assert abs(state.tr() - 1.0) < 1e-6


# --------------------------------------------------------------------------
# davies_operators -- correct construction & sign convention
# --------------------------------------------------------------------------
def _thermal_gamma(kT):
    """Detailed-balance ohmic gamma; large at positive omega."""
    def gamma(w):
        if abs(w) < 1e-10:
            return 0.3 * kT
        return 0.3 * w / (1.0 - math.exp(-w / kT))
    return gamma


def test_davies_operators_drives_to_ground_state():
    """At low temperature the system must relax DOWNWARD in energy.
    A wrong sign convention would make it heat up instead."""
    import math
    dim = 5
    H = qutip.num(dim) + 0.1 * qutip.num(dim) ** 2          # anharmonic ladder
    X = qutip.destroy(dim) + qutip.create(dim)
    kT = 0.1
    c_ops = davies_operators(H, X, _thermal_gamma(kT))
    # start in the highest level
    rho0 = qutip.ket2dm(qutip.basis(dim, dim - 1))
    tlist = np.linspace(0, 60, 40)
    res = qutip.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[H])
    E = np.real(res.expect[0])
    assert E[-1] < E[0] - 0.5            # energy went DOWN
    # ground-state energy is the smallest eigenvalue
    Egs = float(np.min(H.eigenenergies()))
    assert abs(E[-1] - Egs) < 0.2        # settled near the ground state


def test_davies_operators_matches_manual_build():
    """davies_operators should agree with a hand build using omega=E_b-E_a."""
    import math
    dim = 4
    H = qutip.num(dim) + 0.05 * qutip.num(dim) ** 2
    X = qutip.destroy(dim) + qutip.create(dim)
    gamma = _thermal_gamma(0.5)

    c_auto = davies_operators(H, X, gamma)

    evals, evecs = np.linalg.eigh(H.full())
    Xe = evecs.conj().T @ X.full() @ evecs
    c_manual = []
    for a in range(dim):
        for b in range(dim):
            amp = Xe[a, b]
            if abs(amp) < 1e-14:
                continue
            omega = evals[b] - evals[a]
            sg = math.sqrt(gamma(float(omega)))
            if sg == 0:
                continue
            P = np.outer(evecs[:, a], evecs[:, b].conj())
            c_manual.append(qutip.Qobj(sg * amp * P, dims=H.dims))

    # Same total dissipator (operator order may differ)
    D_auto = sum(qutip.lindblad_dissipator(c) for c in c_auto)
    D_manual = sum(qutip.lindblad_dissipator(c) for c in c_manual)
    assert (D_auto - D_manual).norm() < 1e-10


def test_davies_operators_with_lamb_shift_and_bare():
    dim = 4
    H = qutip.num(dim)
    X = qutip.destroy(dim) + qutip.create(dim)
    c_ops, H_LS, (L_ops, omegas) = davies_operators(
        H, X, _thermal_gamma(0.5),
        imag_gamma=lambda w: 0.02 * w, return_bare=True,
    )
    assert len(c_ops) == len(L_ops) == omegas.size
    assert (H_LS - H_LS.dag()).norm() < 1e-10       # Hermitian
    assert H_LS.norm() > 0


def test_davies_operators_rejects_array_gamma():
    dim = 3
    H = qutip.num(dim)
    X = qutip.destroy(dim) + qutip.create(dim)
    with pytest.raises(TypeError):
        davies_operators(H, X, np.ones(9))


# --------------------------------------------------------------------------
# Ensemble result: std, sem, and raw samples
# --------------------------------------------------------------------------
def test_ensemble_exposes_std_sem_and_samples():
    dim = 5
    H = qutip.num(dim)
    c_ops = random_collapse_ops(20, dim, seed=3)
    rho0 = qutip.ket2dm(qutip.basis(dim, dim - 1))
    tlist = np.linspace(0, 4, 12)
    e_ops = [qutip.num(dim), H]
    n = 20

    res = mesolve_ensemble(H, rho0, tlist, c_ops, M=6, e_ops=e_ops,
                            n_realizations=n, rng=0)

    # shapes
    assert res.samples.shape == (n, len(e_ops), tlist.size)
    assert len(res.std) == len(e_ops)
    assert res.std[0].shape == tlist.shape

    # relationships: mean, std, sem must be consistent with raw samples
    for k in range(len(e_ops)):
        np.testing.assert_allclose(res.expect[k], res.samples[:, k, :].mean(axis=0),
                                    atol=1e-12)
        np.testing.assert_allclose(res.std[k],
                                    res.samples[:, k, :].std(axis=0, ddof=1),
                                    atol=1e-12)
        np.testing.assert_allclose(res.sem[k], res.std[k] / math.sqrt(n),
                                    atol=1e-12)


def test_std_does_not_shrink_but_sem_does():
    """std characterizes single-trajectory noise (roughly constant in n);
    sem is the uncertainty of the mean (shrinks as 1/sqrt(n))."""
    dim = 5
    H = qutip.num(dim)
    c_ops = random_collapse_ops(20, dim, seed=7)
    rho0 = qutip.ket2dm(qutip.basis(dim, dim - 1))
    tlist = np.linspace(0, 4, 10)
    e_ops = [qutip.num(dim)]

    small = mesolve_ensemble(H, rho0, tlist, c_ops, M=6, e_ops=e_ops,
                              n_realizations=12, rng=1)
    large = mesolve_ensemble(H, rho0, tlist, c_ops, M=6, e_ops=e_ops,
                              n_realizations=96, rng=2)

    std_small = np.max(small.std[0])
    std_large = np.max(large.std[0])
    sem_small = np.max(small.sem[0])
    sem_large = np.max(large.sem[0])

    # std stays the same order of magnitude (within 2x); sem drops noticeably
    assert 0.5 < std_large / std_small < 2.0
    assert sem_large < 0.6 * sem_small


def test_jackknife_exposes_samples():
    dim = 4
    H = qutip.num(dim)
    c_ops = random_collapse_ops(15, dim, seed=4)
    rho0 = qutip.ket2dm(qutip.basis(dim, 1))
    tlist = np.linspace(0, 3, 10)
    e_ops = [qutip.num(dim)]
    res = mesolve_jackknife(H, rho0, tlist, c_ops, M=8, e_ops=e_ops,
                              n_realizations=6, rng=0)
    assert res.samples.shape == (6, 1, tlist.size)
    assert "direct_samples" in res.extra
    assert res.std[0].shape == tlist.shape


# --------------------------------------------------------------------------
# Generality: bundling is not tied to Davies operators or ohmic baths
# --------------------------------------------------------------------------
def test_bundle_works_on_arbitrary_operators():
    """bundle() must reproduce the full dissipator for ANY collapse-operator
    list, regardless of how the operators were built (no Davies assumption)."""
    dim = 5
    H = qutip.num(dim) + 0.1 * qutip.num(dim) ** 2
    rng = np.random.default_rng(0)
    # operators with no relation to any H eigenbasis or spectral function
    my_ops = [qutip.Qobj(0.3 * (rng.standard_normal((dim, dim))
                                + 1j * rng.standard_normal((dim, dim))))
              for _ in range(12)]
    rho0 = qutip.ket2dm(qutip.basis(dim, dim - 1))
    tlist = np.linspace(0, 4, 25)
    e_ops = [qutip.num(dim)]

    det = np.real(np.asarray(
        qutip.mesolve(H, rho0, tlist, c_ops=my_ops, e_ops=e_ops).expect[0]))
    ens = mesolve_ensemble(H, rho0, tlist, my_ops, M=4, e_ops=e_ops,
                            n_realizations=40, rng=0)
    rms = np.sqrt(np.mean((ens.expect[0] - det) ** 2))
    assert rms < 0.1 * max(np.ptp(det), 1e-9)


def test_davies_works_with_non_ohmic_spectral_function():
    """davies_operators must accept an arbitrary gamma (here Drude-Lorentz),
    not only the ohmic form used elsewhere in the tests."""
    def gamma_drude(omega, lam=0.5, gamma_c=2.0, kT=1.0):
        if abs(omega) < 1e-12:
            return 2.0 * lam * kT / gamma_c
        J = 2.0 * lam * gamma_c * omega / (omega ** 2 + gamma_c ** 2)
        return J / (1.0 - math.exp(-omega / kT))

    dim = 5
    H = qutip.num(dim) + 0.1 * qutip.num(dim) ** 2
    X = qutip.destroy(dim) + qutip.create(dim)
    c_ops = davies_operators(H, X, gamma_drude)
    assert len(c_ops) > 0
    # still relaxes downward in energy (detailed balance respected)
    rho0 = qutip.ket2dm(qutip.basis(dim, dim - 1))
    tlist = np.linspace(0, 40, 30)
    E = np.real(np.asarray(
        qutip.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[H]).expect[0]))
    assert E[-1] < E[0] - 0.5

# --------------------------------------------------------------------------
# davies_operators -- sparsity / threshold options (opt-in, default-preserving)
# --------------------------------------------------------------------------
def _ohmic(kT):
    def gamma(w):
        if abs(w) < 1e-10:
            return 0.3 * kT
        return 0.3 * w / (1.0 - math.exp(-w / kT))
    return gamma


def test_coupling_threshold_default_is_identical():
    """coupling_threshold=0.0 (default) must reproduce the full operator set
    exactly -- same count and same operators."""
    dim = 6
    H = qutip.num(dim) + 0.1 * qutip.num(dim) ** 2
    X = qutip.destroy(dim) + qutip.create(dim)
    g = _ohmic(0.3)
    c_default = davies_operators(H, X, g)
    c_explicit = davies_operators(H, X, g, coupling_threshold=0.0)
    assert len(c_default) == len(c_explicit)
    for ca, cb in zip(c_default, c_explicit):
        assert (ca - cb).norm() < 1e-15


def test_coupling_threshold_prunes_negligible_without_error():
    """A modest coupling_threshold drops only negligible-coupling operators,
    leaving the dissipator unchanged when X is sparse in the eigenbasis."""
    N = 4
    si, sx, sz = qutip.qeye(2), qutip.sigmax(), qutip.sigmaz()

    def op(o, i):
        lst = [si] * N
        lst[i] = o
        return qutip.tensor(lst)

    H = sum(op(sz, i) for i in range(N)) + 0.5 * sum(
        op(sx, i) * op(sx, i + 1) for i in range(N - 1))
    X = op(sx, 0)
    g = _ohmic(0.5)

    c_full = davies_operators(H, X, g)
    c_pruned = davies_operators(H, X, g, coupling_threshold=1e-2)
    assert len(c_pruned) <= len(c_full)

    D_full = sum(qutip.lindblad_dissipator(c) for c in c_full)
    D_pruned = sum(qutip.lindblad_dissipator(c) for c in c_pruned)
    assert (D_full - D_pruned).norm() / D_full.norm() < 1e-9


def test_coupling_threshold_reduces_operator_count_when_aggressive():
    """A large coupling_threshold actually removes operators."""
    dim = 6
    H = qutip.num(dim) + 0.1 * qutip.num(dim) ** 2
    X = qutip.destroy(dim) + qutip.create(dim)
    g = _ohmic(0.5)
    n_full = len(davies_operators(H, X, g))
    n_pruned = len(davies_operators(H, X, g, coupling_threshold=10.0))
    assert n_pruned < n_full


def test_lamb_shift_threshold_defaults_to_operator_threshold():
    """With lamb_shift_threshold unset, the Lamb shift uses the operator
    threshold -- the original behavior."""
    dim = 5
    H = qutip.num(dim) + 0.1 * qutip.num(dim) ** 2
    X = qutip.destroy(dim) + qutip.create(dim)
    ig = lambda w: 0.02 * w
    _, H_LS = davies_operators(H, X, _ohmic(0.5), imag_gamma=ig, threshold=0.0)
    _, H_LS_inherit = davies_operators(
        H, X, _ohmic(0.5), imag_gamma=ig, threshold=0.0,
        lamb_shift_threshold=None)
    assert (H_LS - H_LS_inherit).norm() < 1e-12


def test_lamb_shift_threshold_decouples_from_operator_threshold():
    """An aggressive operator threshold should NOT silently kill the Lamb
    shift when lamb_shift_threshold is set independently."""
    dim = 5
    H = qutip.num(dim) + 0.1 * qutip.num(dim) ** 2
    X = qutip.destroy(dim) + qutip.create(dim)
    ig = lambda w: 0.02 * w
    _, H_LS_baseline = davies_operators(H, X, _ohmic(0.5), imag_gamma=ig)
    # inheriting an aggressive operator threshold wipes the Lamb shift...
    _, H_LS_inherit = davies_operators(
        H, X, _ohmic(0.5), imag_gamma=ig, threshold=0.05)
    assert H_LS_inherit.norm() < 1e-12
    # ...but decoupling restores it to the unfiltered value.
    _, H_LS_decoupled = davies_operators(
        H, X, _ohmic(0.5), imag_gamma=ig, threshold=0.05,
        lamb_shift_threshold=0.0)
    assert (H_LS_decoupled - H_LS_baseline).norm() < 1e-12
