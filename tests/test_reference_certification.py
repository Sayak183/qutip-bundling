"""The reference certification must not throw away a good reference.

Certification compares the primary solve against a different resolution. The
cheap comparison is downward, at half the substeps -- but a halved run is not
guaranteed stable just because the primary is. On the stiff oscillator at
dimension 256 the primary ran fine at 128 substeps while the 64-substep partner
diverged, and the naive implementation propagated that exception, discarding
2.4 days of completed work.

A diverging downward partner means the *check* was under-resolved, not that the
reference is bad. These tests pin the escalation, and pin that escalating never
turns into rubber-stamping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from qutip_bundling.native_solver import SolverInstabilityError  # noqa: E402

run_method_comparison = pytest.importorskip(
    "run_method_comparison",
    reason="benchmark scripts require the benchmarks/ directory on sys.path",
)


class _Result:
    """Minimal stand-in for NativeResult."""

    def __init__(self, value, n_times):
        self.expect = [np.full(n_times, float(value))]
        self.states = [None] * n_times


def _solver(min_stable_substeps, value_at):
    """Fake rk4_mesolve: raises below ``min_stable_substeps``."""

    def solve(H, rho0, tlist, c_ops, e_ops=None, substeps=1, store_states=False):
        if substeps < min_stable_substeps:
            raise SolverInstabilityError(f"diverged at substeps={substeps}")
        return _Result(value_at(substeps), len(tlist))

    return solve


@pytest.fixture
def patched(monkeypatch):
    def apply(min_stable, value_at=lambda s: 1.0):
        monkeypatch.setattr(run_method_comparison, "rk4_mesolve",
                            _solver(min_stable, value_at))
    return apply


def test_downward_check_used_when_it_is_stable(patched):
    patched(min_stable=1)
    _, _, selfcheck = run_method_comparison.certified_reference(
        None, None, [], 32)
    assert selfcheck["direction"] == "down"
    assert selfcheck["substeps_pair"] == [16, 32]
    assert selfcheck["passed"]


def test_check_escalates_upward_instead_of_discarding_the_reference(patched):
    """The dimension-256 failure: primary fine, halved partner unstable."""
    patched(min_stable=128)
    _, _, selfcheck = run_method_comparison.certified_reference(
        None, None, [], 128)
    assert selfcheck["direction"] == "up"
    assert selfcheck["substeps_pair"] == [128, 256]
    assert selfcheck["primary_substeps"] == 128
    assert selfcheck["passed"]


def test_escalation_still_refuses_a_reference_that_disagrees(patched):
    """Escalating must not become rubber-stamping: disagreement still fails."""
    patched(min_stable=128, value_at=lambda s: 1.0 if s == 128 else 5.0)
    _, _, selfcheck = run_method_comparison.certified_reference(
        None, None, [], 128)
    assert selfcheck["direction"] == "up"
    assert selfcheck["max_abs_dev"] == pytest.approx(4.0)
    assert not selfcheck["passed"]
