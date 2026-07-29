from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS_DIR))

from run_high_dim_spin_reference import (  # noqa: E402
    _sha256,
    _state_comparison,
    _state_diagnostics,
    _write_state_archive,
)


def test_state_diagnostics_for_pure_density_matrices():
    states = np.zeros((2, 2, 2), dtype=np.complex128)
    states[:, 0, 0] = 1.0

    diagnostics = _state_diagnostics(states)

    assert diagnostics["max_hermiticity_error"] == 0.0
    assert diagnostics["max_trace_drift"] == 0.0
    np.testing.assert_allclose(diagnostics["trace_real"], [1.0, 1.0])
    np.testing.assert_allclose(diagnostics["trace_imag"], [0.0, 0.0])
    np.testing.assert_allclose(diagnostics["purity"], [1.0, 1.0])


def test_state_comparison_reports_trace_and_frobenius_distances():
    reference = np.zeros((2, 2, 2), dtype=np.complex128)
    reference[:, 0, 0] = 1.0
    partner = reference.copy()
    epsilon = 0.125
    partner[1, 0, 0] -= epsilon
    partner[1, 1, 1] += epsilon

    comparison = _state_comparison(reference, partner)

    np.testing.assert_allclose(
        comparison["trace_distance_by_time"], [0.0, epsilon]
    )
    np.testing.assert_allclose(
        comparison["frobenius_by_time"],
        [0.0, np.sqrt(2.0) * epsilon],
    )
    assert comparison["max_trace_distance"] == epsilon


def test_state_archive_round_trip_and_checksum(tmp_path):
    path = tmp_path / "reference.npz"
    times = np.asarray([0.0, 1.0])
    states = np.zeros((2, 2, 2), dtype=np.complex128)
    states[:, 0, 0] = 1.0

    _write_state_archive(
        path,
        times=times,
        states=states,
        dim=2,
        n_sites=1,
        substeps=8,
    )

    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    assert len(_sha256(path)) == 64
    with np.load(path, allow_pickle=False) as archive:
        assert archive.files == ["times", "states", "dim", "n_sites", "substeps"]
        np.testing.assert_array_equal(archive["times"], times)
        np.testing.assert_array_equal(archive["states"], states)
        assert int(archive["dim"]) == 2
        assert int(archive["n_sites"]) == 1
        assert int(archive["substeps"]) == 8
