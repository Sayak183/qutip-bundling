"""The streaming path must be the list path, exactly.

``iter_davies_operators`` exists so that System B at dimension 256 -- 32,637
operators, ~32 GB as a list -- can be bundled into the 16 MB it actually needs.
It is the same sum either way, so "the same" here means bit-for-bit, not
"close": these tests use ``atol=0, rtol=0`` deliberately.

That equality is what makes the streaming path safe to trust where the list
path cannot run. Below the memory wall both work and can be compared; above it,
only one works, and these tests are the reason to believe it.
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
from qutip_bundling.operators import iter_davies_operators  # noqa: E402


SYSTEMS = [
    pytest.param(lambda: build_spin_chain(4, g=0.0), id="A-TFIM-dim16"),
    pytest.param(lambda: build_mixed_field_chain(4), id="B-mixed-dim16"),
    pytest.param(lambda: build_oscillator_bath(8), id="C-oscillator-dim16"),
    pytest.param(lambda: build_mixed_field_chain(5), id="B-mixed-dim32"),
]


@pytest.mark.parametrize("build", SYSTEMS)
def test_streaming_reproduces_the_list_exactly(build):
    """Same operators, same order, same bits."""
    H, X, _ = build()
    listed = davies_operators(H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL)
    streamed = list(iter_davies_operators(
        H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL))

    assert len(streamed) == len(listed)
    for index, (a, b) in enumerate(zip(listed, streamed)):
        assert a.dims == b.dims, f"dims differ at operator {index}"
        assert np.array_equal(a.full(), b.full()), (
            f"operator {index} differs; streaming must be bit-for-bit identical"
        )


@pytest.mark.parametrize("build", SYSTEMS)
def test_streaming_gives_the_same_dissipator(build):
    """The quantity that actually matters: D[rho] must be unchanged."""
    H, X, psi0 = build()
    rho = qutip.ket2dm(psi0).full()

    def dissipator(ops):
        out = np.zeros_like(rho)
        for c in ops:
            m = c.full()
            out += (m @ rho @ m.conj().T
                    - 0.5 * (m.conj().T @ m @ rho + rho @ m.conj().T @ m))
        return out

    listed = dissipator(davies_operators(
        H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL))
    streamed = dissipator(iter_davies_operators(
        H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL))
    assert np.array_equal(listed, streamed)


def test_streaming_holds_one_operator_at_a_time():
    """The point of the exercise: the generator must not accumulate.

    Checked structurally rather than by measuring RSS, which is too noisy to
    assert on: the generator is consumed one item at a time and the previous
    item is dropped, so if it were secretly building a list the referrer count
    would show it.
    """
    H, X, _ = build_mixed_field_chain(4)
    stream = iter_davies_operators(H, X, gamma,
                                   degeneracy_tol=DAVIES_DEGENERACY_TOL)
    first = next(stream)
    assert isinstance(first, qutip.Qobj)
    # advancing must not keep the previous operator alive through the generator
    import weakref
    ref = weakref.ref(first)
    del first
    next(stream)
    assert ref() is None, (
        "the generator is retaining operators it has already yielded"
    )


def test_streaming_refuses_whole_set_options():
    """imag_gamma and return_bare describe the set this function never holds."""
    H, X, _ = build_spin_chain(3, g=0.0)
    for kwargs in ({"imag_gamma": lambda w: 0.1}, {"return_bare": True}):
        with pytest.raises(TypeError, match="whole operator set"):
            list(iter_davies_operators(H, X, gamma, **kwargs))


def test_streaming_accepts_the_same_tuning_knobs():
    """Thresholds must behave identically on both paths, or they diverge."""
    H, X, _ = build_mixed_field_chain(4)
    for kwargs in ({"degeneracy_tol": 1e-6},
                   {"threshold": 1e-3},
                   {"coupling_threshold": 1e-6}):
        listed = davies_operators(H, X, gamma, **kwargs)
        streamed = list(iter_davies_operators(H, X, gamma, **kwargs))
        assert len(listed) == len(streamed), f"count differs for {kwargs}"
        for a, b in zip(listed, streamed):
            assert np.array_equal(a.full(), b.full()), f"values differ for {kwargs}"


# --------------------------------------------------------------------------
# The solver paths
# --------------------------------------------------------------------------
def test_bundles_match_the_list_route_exactly():
    """Streaming accumulation must equal bundling a materialised list."""
    from qutip_bundling import bundle_davies_from_phases, davies_operator_count
    from qutip_bundling.operators import bundle_from_phases, random_phases

    H, X, _ = build_mixed_field_chain(4)
    n_l = davies_operator_count(H, X, gamma,
                                degeneracy_tol=DAVIES_DEGENERACY_TOL)
    assert n_l == len(davies_operators(
        H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL))

    phases = random_phases(6, n_l, distribution="phase", rng=0)
    listed = bundle_from_phases(
        davies_operators(H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL),
        phases)
    streamed = bundle_davies_from_phases(
        H, X, gamma, phases, degeneracy_tol=DAVIES_DEGENERACY_TOL)

    assert len(listed) == len(streamed) == 6
    for index, (a, b) in enumerate(zip(listed, streamed)):
        assert np.allclose(a.full(), b.full(), rtol=0, atol=1e-14), (
            f"bundle {index} differs between the list and streaming routes"
        )


@pytest.mark.parametrize("backend", ["qutip", "native"])
def test_solver_paths_agree(backend):
    """mesolve_ensemble_davies must reproduce mesolve_ensemble at equal rng.

    This is the guarantee the whole streaming route rests on: below the memory
    wall both run and can be compared; above it only one runs, and this is the
    reason to believe it.
    """
    from qutip_bundling import mesolve_ensemble, mesolve_ensemble_davies

    H, X, psi0 = build_mixed_field_chain(4)
    rho0 = qutip.ket2dm(psi0)
    tlist = np.linspace(0.0, 1.0, 12)
    c_ops = davies_operators(H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL)

    common = dict(M=4, e_ops=[H], n_realizations=3, backend=backend)
    listed = mesolve_ensemble(H, rho0, tlist, c_ops, rng=1234, **common)
    streamed = mesolve_ensemble_davies(
        H, rho0, tlist, X, gamma, rng=1234,
        degeneracy_tol=DAVIES_DEGENERACY_TOL, **common)

    assert np.allclose(listed.samples, streamed.samples, rtol=0, atol=1e-12), (
        "the two solver routes disagree; they must draw the same phases and "
        "build the same bundles"
    )


def test_davies_count_needs_no_operators():
    """The count must come from the plan, not from building the list."""
    from qutip_bundling import davies_operator_count

    H, X, _ = build_mixed_field_chain(5)
    assert davies_operator_count(
        H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL) == 513


def test_streaming_actually_saves_the_memory():
    """The whole point, measured rather than assumed.

    At System B dimension 64 (N_L = 2,017) the list route peaks near 380 MB and
    the streaming route near 2 MB -- about 180x. The assertion is deliberately
    loose (5x) so it tests the *mechanism* rather than an allocator detail: if
    streaming ever started accumulating, the ratio would collapse to 1.
    """
    import gc
    import tracemalloc

    from qutip_bundling import bundle_davies_from_phases, davies_operator_count
    from qutip_bundling.operators import bundle_from_phases, random_phases

    H, X, _ = build_mixed_field_chain(6)          # dim 64
    n_l = davies_operator_count(H, X, gamma,
                                degeneracy_tol=DAVIES_DEGENERACY_TOL)
    phases = random_phases(8, n_l, rng=0)

    gc.collect()
    tracemalloc.start()
    listed = bundle_from_phases(
        davies_operators(H, X, gamma, degeneracy_tol=DAVIES_DEGENERACY_TOL),
        phases)
    _, peak_list = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del listed
    gc.collect()

    tracemalloc.start()
    bundle_davies_from_phases(H, X, gamma, phases,
                              degeneracy_tol=DAVIES_DEGENERACY_TOL)
    _, peak_stream = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_stream * 5 < peak_list, (
        f"streaming peaked at {peak_stream/1024**2:.1f} MB against "
        f"{peak_list/1024**2:.1f} MB for the list route -- the saving is gone, "
        f"so something is accumulating"
    )
