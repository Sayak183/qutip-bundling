"""Result 1's height table must measure all three systems the same way.

The committed table once compared System A's *time-averaged* error against
System B's and System C's error at the worst-time slice t*, and then read a
physical difference out of the resulting exponents: +0.64 for A against +0.55
for B. Measured consistently at t*, both chains give +0.61. The gap was the
convention, not the systems.

Nothing about that is visible in the document -- three rows of numbers with no
statement of which error they are -- so it is pinned here instead. The test
recomputes the table from the committed data and also asserts that the two
conventions genuinely disagree, so that if someone "simplifies" the measurement
back to a time average the failure names what changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
DATA = BENCHMARKS / "data"
sys.path.insert(0, str(BENCHMARKS))

common = pytest.importorskip(
    "common", reason="benchmark scripts require benchmarks/ on sys.path")
linregress = pytest.importorskip("scipy.stats").linregress

M_FIXED = 8

# The rows as published, at M=8, energy, evaluated at t*.
PUBLISHED = {
    "spin_chain": ([16, 32, 64, 128, 256],
                   [3.3e-2, 5.0e-2, 8.0e-2, 1.24e-1, 1.74e-1], +0.61),
    "mixed_chain": ([16, 32, 64, 128],
                    [2.6e-2, 3.4e-2, 5.5e-2, 8.8e-2], +0.61),
    "oscillator_bath": ([16, 32, 64],
                        [2.6e-3, 2.3e-3, 2.0e-3], -0.20),
}


def _samples(item):
    s = common.result1_samples(item, "energy")
    return s.reshape(1, -1) if s.ndim == 1 else s


def _load(system, dim):
    path = DATA / f"accuracy_vs_M_{system}_dim{dim}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _bias(system, dim, worst_time=True):
    """Bias of the M=8 bundle. worst_time=True is the published convention:
    the t* of the SMALLEST-M bundle, held for every M."""
    data = _load(system, dim)
    ref = common.result1_reference(data, "energy")
    sweep = data["slb_sweep"]
    item = next(i for i in sweep if i.get("M", i.get("bundles")) == M_FIXED)
    mean = np.mean(_samples(item), axis=0)
    if not worst_time:
        return float(np.mean(np.abs(mean - ref)))
    tstar = int(np.argmax(np.abs(np.mean(_samples(sweep[0]), axis=0) - ref)))
    return abs(float(mean[tstar]) - float(ref[tstar]))


@pytest.mark.parametrize("system", sorted(PUBLISHED))
def test_height_row_matches_the_data(system):
    dims, published, _exponent = PUBLISHED[system]
    measured = [_bias(system, d) for d in dims]
    # 3% covers the two significant figures the table prints.
    assert measured == pytest.approx(published, rel=0.03)


@pytest.mark.parametrize("system", sorted(PUBLISHED))
def test_height_exponent_matches_the_data(system):
    dims, _published, exponent = PUBLISHED[system]
    measured = [_bias(system, d) for d in dims]
    slope = linregress(np.log10(dims), np.log10(measured)).slope
    assert slope == pytest.approx(exponent, abs=0.005)


def test_the_two_chains_grow_at_the_same_rate():
    """The claim the corrected table makes. Worth its own test because the
    previous version of the section read a difference into the gap."""
    slopes = {}
    for system in ("spin_chain", "mixed_chain"):
        dims, _pub, _exp = PUBLISHED[system]
        slopes[system] = linregress(
            np.log10(dims), np.log10([_bias(system, d) for d in dims])).slope
    assert slopes["spin_chain"] == pytest.approx(slopes["mixed_chain"], abs=0.01)


def test_the_two_conventions_really_do_differ():
    """If this ever passes trivially, the distinction being pinned above has
    evaporated and the other tests here stop meaning anything."""
    worst = _bias("spin_chain", 64, worst_time=True)
    averaged = _bias("spin_chain", 64, worst_time=False)
    assert averaged < worst
    assert averaged / worst < 0.95, "time-average should sit ~10% below t*"
