"""The rules deciding which points get drawn, and against what target.

These are pure functions, but they carry judgement that took a wrong turn once
already and would take it again silently:

  * `_curve_window` originally cut a curve at the FIRST point where sampling
    noise overtook bias. That does not survive contact with data, because
    error/s.e.m. is itself estimated from a finite number of realizations and
    scatters across the threshold: on x_sx the oscillator at dimension 8 runs
    hollow, hollow, hollow, FILLED, hollow, and cutting at the first hollow
    point left a ONE-POINT curve. The rule is now the first point from which
    every larger M is also hollow.

  * `observable_targets` exists because one absolute tolerance is a different
    standard for each observable. At oscillator dimension 64, an RMSE of 0.02
    is 0.008% of n^2's span and 374% of the coherence's -- a factor of 47,000
    on the same system -- and the resulting M* produced a cost curve that FELL
    with dimension, which is impossible.

  * `derive_slb` must take the worst observable relative to ITS OWN target, not
    the largest raw error, or the biggest-scale observable always wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

pmc = pytest.importorskip(
    "plot_method_comparison",
    reason="benchmark scripts require benchmarks/ on sys.path")
piso = pytest.importorskip("plot_isocost_vs_dim")


def _row(m, error, sem):
    """A method_errors row: (family, wall, error, note, n_samples, sem)."""
    return ("slb", 1.0, error, f"M={m}", 16, sem)


# --- the bias/noise verdict ----------------------------------------------

def test_bias_limited_threshold_is_where_the_two_contributions_are_equal():
    """error^2 ~= bias^2 + sem^2, so bias exceeds noise exactly at sqrt(2)."""
    # bool() because the comparison returns a numpy bool, not a Python one.
    assert bool(pmc._bias_limited(1.0, 1.0 / np.sqrt(2) - 1e-9))
    assert not bool(pmc._bias_limited(1.0, 1.0 / np.sqrt(2) + 1e-9))


def test_a_point_with_no_ensemble_counts_as_bias_limited():
    """A deterministic solver has no measurable noise; it must not be reported
    as noise-limited just because its s.e.m. is missing."""
    assert bool(pmc._bias_limited(1e-3, None))
    assert bool(pmc._bias_limited(1e-3, 0.0))


# --- the curve window ----------------------------------------------------

def test_lone_hollow_point_is_scatter_and_does_not_truncate():
    """hollow, hollow, hollow, FILLED, hollow -- the shape measured on x_sx.
    Cutting at the first hollow point would leave one point."""
    rows = [_row(2, 1.0, 1.0),     # hollow
            _row(4, 1.0, 1.0),     # hollow
            _row(8, 1.0, 1.0),     # hollow
            _row(16, 1.0, 0.1),    # FILLED
            _row(32, 1.0, 1.0)]    # hollow
    kept = [pmc._m_of(r) for r in pmc._curve_window(rows)]
    assert len(kept) > 1, "a lone hollow point must not collapse the curve"
    assert kept[-1] == 32


def test_sustained_crossover_truncates_and_keeps_the_crossing_point():
    """Once every larger M is noise-limited, raising M is not what helps, so
    the curve stops there -- keeping that point as the evidence."""
    rows = [_row(2, 1.0, 0.01), _row(4, 1.0, 0.01), _row(8, 1.0, 0.01),
            _row(16, 1.0, 1.0), _row(32, 1.0, 1.0)]     # both hollow
    kept = [pmc._m_of(r) for r in pmc._curve_window(rows)]
    assert kept[-1] == 16, "should stop at the first of the sustained run"
    assert 32 not in kept


def test_window_keeps_at_most_the_configured_number_of_points():
    rows = [_row(m, 1.0, 0.01) for m in (2, 4, 8, 16, 32, 64, 128, 256)]
    kept = pmc._curve_window(rows)
    assert len(kept) == pmc.MAX_CURVE_POINTS
    assert pmc._m_of(kept[-1]) == 256, "the largest M must survive trimming"


# --- the per-observable target -------------------------------------------

def test_target_scales_with_each_observables_own_span():
    """Two observables differing by 1000x in scale get targets differing by
    1000x, which is the whole point."""
    point = {"reference": [[0.0, 1.0], [0.0, 1000.0]],
             "observables": ["small", "large"]}
    targets = piso.observable_targets(point)
    assert targets[0] == pytest.approx(piso.TARGET_REL * 1.0)
    assert targets[1] == pytest.approx(piso.TARGET_REL * 1000.0)


def test_single_observable_files_still_load():
    """Older Result 4 files stored one observable as a flat list."""
    point = {"reference": [0.0, 1.0]}
    targets = piso.observable_targets(point)
    assert targets.shape == (1,)


def test_mstar_is_set_by_the_hardest_observable_not_the_largest_error():
    """A large-scale observable with a proportionally large target must not
    dominate a small-scale one that is missing its own target."""
    n_runs = 4
    # observable 0: tiny span, error well outside its 3% target
    # observable 1: huge span, big raw error but comfortably inside its target
    ref = np.array([[0.0, 1.0], [0.0, 1000.0]])
    point = {
        "reference": ref.tolist(),
        "observables": ["tight", "loose"],
        "slb_sweep": [{
            "M": 8, "per_run_cost": 1.0,
            # obs "tight": mean 0.5 against reference [0, 1] -> error ~0.5,
            #   against a target of 0.03. Misses by ~17x.
            # obs "loose": sits ON its reference -> error 0, against a target
            #   of 30. Its RAW scale is 1000x bigger, which is the trap.
            "samples": np.stack([np.stack([np.full(2, 0.5),
                                           np.array([0.0, 1000.0])])]
                                * n_runs).tolist(),
        }],
    }
    targets = piso.observable_targets(point)
    m, cost, reached, bias_sq, noise_sq, binding = piso.derive_slb(
        point, n_runs, targets, "ensemble")
    assert binding == "tight", "the binding observable must be scale-relative"
    assert reached is False
