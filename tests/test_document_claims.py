"""Numbers written in BENCHMARKS.md must still match the data files.

The repository's numbers live in two places. Data files regenerate themselves
when a sweep is re-run; sentences do not. Every defect found in the last three
sessions was the second drifting away from the first, and nearly all of them
survived a fully green test suite -- because the tests exercise code, and
nothing read the document.

Five of them were found by hand in one pass:

  * section 3.3 said Result 5 used M = 16/32/64 with 128 thermal realizations;
    it ran 8/16/32 with 16
  * Result 1's "Sizes" paragraph still said dimensions 16, 32 and 64 while the
    same section tabulated five sizes twenty lines below
  * Result 3's solver list said SLB at M = 16, 32 against a sweep of 2 to 256
  * Result 5's kernel table said "dim 4 to 32" while the paragraph under it
    reported dimension 256
  * System C's height row lost its exponent when dimension 128 was merged and
    was left reading "flat"

This file closes that gap for the claims that are recomputable: it PARSES the
published tables and sentences out of the Markdown and checks them against the
data, so extending a sweep without updating the prose fails here.

Note the difference from `tests/test_result1_heights.py`, which pins the same
height table against hard-coded constants. That catches the DATA changing under
a fixed document. This catches the DOCUMENT changing away from fixed data. Both
directions have now happened, so both are checked.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
DATA = BENCHMARKS / "data"
DOC = BENCHMARKS / "BENCHMARKS.md"
sys.path.insert(0, str(BENCHMARKS))

common = pytest.importorskip(
    "common", reason="benchmark scripts require benchmarks/ on sys.path")
linregress = pytest.importorskip("scipy.stats").linregress

# The document renders minus signs as U+2212 and exponents as superscripts.
MINUS = str.maketrans({"−": "-"})
SUPERSCRIPT = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹",
                            "-0123456789")

# Result 1's fitting window, from plot_R1_invariance.py.
MIN_M, MAX_M = 2, 32


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


# --- recomputation, matching plot_R1_invariance.py exactly ----------------

def _samples(item):
    a = common.result1_samples(item, "energy")
    return a.reshape(1, -1) if a.ndim == 1 else a


def _sweep(system: str, dim: int):
    path = DATA / f"accuracy_vs_M_{system}_dim{dim}.json"
    if not path.exists():
        pytest.fail(f"{path.name} is quoted in BENCHMARKS.md but not committed")
    data = json.loads(path.read_text(encoding="utf-8"))
    ref = common.result1_reference(data, "energy")
    sweep = data["slb_sweep"]
    tstar = int(np.argmax(np.abs(np.mean(_samples(sweep[0]), axis=0) - ref)))
    return sweep, ref, tstar


def bias_slope(system: str, dim: int) -> float:
    """Fitted bias exponent: log-log fit over bundle sizes whose bias clears
    twice its own s.e.m., at the worst-time slice."""
    sweep, ref, tstar = _sweep(system, dim)
    ms, biases = [], []
    for item in sweep:
        m = item.get("M", item.get("bundles"))
        if not MIN_M <= m <= MAX_M:
            continue
        col = _samples(item)[:, tstar]
        bias = abs(float(np.mean(col)) - float(ref[tstar]))
        sem = float(np.std(col, ddof=1)) / np.sqrt(len(col))
        if bias > 2 * sem:
            ms.append(m)
            biases.append(bias)
    assert len(ms) >= 3, f"{system} dim {dim}: too few points clear the floor"
    return linregress(np.log10(ms), np.log10(biases)).slope


def height(system: str, dim: int, m_fixed: int = 8) -> float:
    sweep, ref, tstar = _sweep(system, dim)
    item = next(i for i in sweep if i.get("M", i.get("bundles")) == m_fixed)
    return abs(float(np.mean(_samples(item), axis=0)[tstar]) - float(ref[tstar]))


def committed_dims(system: str) -> list[int]:
    """Dimensions with a CANONICAL Result 1 file -- _dim<D>.json and nothing
    after the number. A file like _dim128_r800.json is a deliberate side-run
    at a non-default realization count and is not a committed dimension; the
    old split("dim") crashed on it."""
    canonical = re.compile(rf"^accuracy_vs_M_{re.escape(system)}_dim(\d+)$")
    return sorted(int(m.group(1))
                  for p in DATA.glob(f"accuracy_vs_M_{system}_dim*.json")
                  if (m := canonical.match(p.stem)))


# --- 1. the bias-slope table ---------------------------------------------

def test_result1_slope_table_matches_the_data(doc):
    """Parses:

        | dim | 16 | 32 | 64 | 128 | 256 |
        |---|---|---|---|---|---|
        | bias slope | -0.91 | -0.97 | -0.98 | -0.98 | -0.97 |
    """
    match = re.search(r"\|\s*dim\s*\|([^\n]*)\|\n\|[-| ]+\|\n"
                      r"\|\s*bias slope\s*\|([^\n]*)\|", doc)
    assert match, "Result 1's bias-slope table is not where this test expects it"
    dims = [int(x) for x in match.group(1).split("|") if x.strip()]
    published = [float(x.translate(MINUS))
                 for x in match.group(2).split("|") if x.strip()]
    assert len(dims) == len(published)

    assert dims == committed_dims("spin_chain"), (
        f"the table lists dims {dims}; committed data has "
        f"{committed_dims('spin_chain')}")

    measured = [bias_slope("spin_chain", d) for d in dims]
    assert measured == pytest.approx(published, abs=0.005), (
        f"published {published} against measured "
        f"{[round(v, 3) for v in measured]}")


def test_system_b_slope_sentence_matches_the_data(doc):
    """Parses: System B gives $M^{-1.00}$, $M^{-1.00}$, ... across its four sizes."""
    match = re.search(r"System B gives\s+((?:\$M\^\{-?[\d.]+\}\$,?\s*)+)"
                      r"across its (\w+) sizes", doc)
    assert match, "System B's slope sentence has changed shape"
    published = [float(v) for v in re.findall(r"-?[\d.]+", match.group(1))]
    words = {"three": 3, "four": 4, "five": 5, "six": 6}
    assert len(published) == words[match.group(2)], (
        "the sentence lists a different number of slopes than it claims")

    dims = committed_dims("mixed_chain")
    assert len(published) == len(dims), (
        f"sentence quotes {len(published)} sizes; data has {len(dims)}: {dims}")
    measured = [bias_slope("mixed_chain", d) for d in dims]
    assert measured == pytest.approx(published, abs=0.005)


# --- 2. the height table and its exponents -------------------------------

def _decode(cell: str) -> float:
    """'8.8x10^-2' written with Unicode superscripts -> 8.8e-2.

    Matched rather than split, because the final cell of a row carries a
    trailing annotation like '(to dim 128)'.
    """
    return _decode_with_precision(cell)[0]


def _decode_with_precision(cell: str) -> tuple[float, float]:
    """The value and half a unit in its last printed digit: '1.07x10^-1' ->
    (0.107, 0.0005). A value must round to what is printed, so that is the
    tolerance. rel=0.03 used to stand in for it and let 1.07 stand for a
    measured 1.0647."""
    match = re.match(r"\s*([\d.]+)×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", cell)
    assert match, f"cannot read a value from {cell!r}"
    mantissa, exponent = match.group(1), int(match.group(2).translate(SUPERSCRIPT))
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return (float(mantissa) * 10 ** exponent,
            0.5 * 10 ** (exponent - decimals) * (1 + 1e-9))


@pytest.mark.parametrize("label,system", [("A** TFIM chain", "spin_chain"),
                                          ("B** mixed chain", "mixed_chain"),
                                          ("C** oscillator", "oscillator_bath")])
def test_height_row_and_exponent_match_the_data(doc, label, system):
    """Each row quotes a value per dimension and a fitted N exponent. Both are
    recomputed. The System C row lost its exponent once, to the word 'flat'."""
    # The first cell must contain an arrow: these rows read
    # "2.6x10-2 -> 3.4x10-2 -> ...". Without that guard the search binds to
    # whichever table using this label comes first in the document, which is
    # how section 5.2's solver grid silently captured the B and C rows once.
    row = re.search(
        rf"\|\s*\*\*{re.escape(label)}\s*\|([^|]*→[^|]*)\|([^|]*)\|", doc)
    assert row, f"height row for {label} not found"

    cells = [_decode_with_precision(c.strip()) for c in row.group(1).split("→")
             if "×10" in c]
    values = [v for v, _ in cells]
    dims = committed_dims(system)
    assert len(values) == len(dims), (
        f"{system}: row quotes {len(values)} values, data has {len(dims)}: {dims}")
    measured = [height(system, d) for d in dims]
    for d, m, (v, half) in zip(dims, measured, cells):
        assert abs(m - v) <= half, (
            f"{system} dim {d}: measured {m:.4e} does not round to the printed {v:.4e}")

    exponent = re.search(r"N\^([+-][\d.]+)", row.group(2).translate(MINUS))
    assert exponent, (
        f"{label}'s scaling cell quotes no exponent: {row.group(2).strip()!r}. "
        "A word like 'flat' is not checkable -- quote the fitted number.")
    fitted = linregress(np.log10(dims), np.log10(measured)).slope
    assert fitted == pytest.approx(float(exponent.group(1)), abs=0.005)


# --- 3. claims about which sizes exist at all ----------------------------

def test_sizes_paragraph_matches_the_committed_files(doc):
    """Parses: 'spans dimensions 16 to 512 on System A, 16 to 256 on
    System B and 16 to 128 on the oscillator'. This sentence was stale for
    weeks once, and its shape changed on 2026-09-11 when System B stopped
    sharing a range with the oscillator."""
    match = re.search(r"spans dimensions (\d+) to (\d+) on System A, "
                      r"(\d+) to (\d+) on\s+System B and (\d+) to (\d+) on "
                      r"the oscillator", doc)
    assert match, "Result 1's 'Sizes' sentence has changed shape"
    quoted = [int(g) for g in match.groups()]
    for system, (lo, hi) in zip(("spin_chain", "mixed_chain", "oscillator_bath"),
                                zip(quoted[0::2], quoted[1::2])):
        dims = committed_dims(system)
        assert (dims[0], dims[-1]) == (lo, hi), (
            f"{system} spans {dims[0]}-{dims[-1]}, sentence says {lo}-{hi}")


# --- 4. the two tables a cold reviewer caught, which nothing checked ---------

SIZE_AT_DIM64 = {"spin_chain": 6, "mixed_chain": 6, "oscillator_bath": 32}


def test_reference_table_quotes_the_sector_limit_not_global_gibbs(doc):
    """The §5 profile table must not present global Gibbs as the t→∞ state.

    It did, for weeks, in the section a reader meets first -- reintroducing the
    exact error Result 5 exists to correct. Where the connectivity graph of
    <e|X|e'> is disconnected the limit is Gibbs WITHIN the sector rho_0
    occupies, and at dimension 64 that is -5.6490 for System A against a global
    -5.5687. Recomputed here from plot_extreme_dimension's own helper, so the
    table cannot drift back.
    """
    plot_extreme = pytest.importorskip("plot_extreme_dimension")
    row = re.search(r"\|\s*\*\*Actual t→∞ limit\*\*[^|]*\|([^|]*)\|([^|]*)\|([^|]*)\|",
                    doc)
    assert row, ("the reference table has no 'Actual t→∞ limit' row -- if it was "
                 "removed, global Gibbs is being presented as the limit again")

    published = []
    for cell in row.groups():
        m = re.search(r"(-?[\d.]+)", cell.translate(MINUS))
        assert m, f"cannot read a number from {cell!r}"
        published.append(float(m.group(1)))

    for value, system in zip(published, ("spin_chain", "mixed_chain",
                                         "oscillator_bath")):
        sector, n_sectors = plot_extreme.sector_resolved_energy(
            {"meta": {"params": {"system": system,
                                 "size": SIZE_AT_DIM64[system]}}})
        if sector is None:                 # ergodic: the two targets coincide
            assert n_sectors == 1
            continue
        assert value == pytest.approx(sector, abs=5e-4), (
            f"{system}: table says {value}, sector-resolved limit is {sector:.4f}")


def test_result4_summary_row_is_at_the_largest_dimension(doc):
    """Result 4's summary said 'speedup at the largest dim' while quoting
    dimension-64 values, for as long as the sweep had reached 128.

    Checks the cheap, unambiguous half: the dimension each row claims, and the
    N_L range, against the committed points. The speedups themselves depend on
    an estimator convention this file deliberately does not re-implement.
    """
    rows = re.findall(r"\|\s*\*\*([ABC])\*\*[^|]*\(to (\d+)\)\s*\|"
                      r"\s*([\d,]+)\s*→\s*([\d,]+)\s*\|", doc)
    assert len(rows) == 3, f"expected three Result 4 rows, parsed {len(rows)}"

    systems = {"A": "spin_chain", "B": "mixed_chain", "C": "oscillator_bath"}
    for tag, dim_claimed, nl_lo, nl_hi in rows:
        path = DATA / f"isocost_vs_dim_{systems[tag]}.json"
        if not path.exists():
            pytest.skip(f"{path.name} not committed")
        points = json.loads(path.read_text(encoding="utf-8"))["points"]
        dims = sorted(p["dim"] for p in points)
        n_ls = [p["n_l"] for p in sorted(points, key=lambda q: q["dim"])]

        assert int(dim_claimed) == dims[-1], (
            f"System {tag}: row says 'to {dim_claimed}', data reaches {dims[-1]}")
        assert int(nl_lo.replace(",", "")) == n_ls[0]
        assert int(nl_hi.replace(",", "")) == n_ls[-1], (
            f"System {tag}: row says N_L to {nl_hi}, data has {n_ls[-1]} "
            f"at dim {dims[-1]}")


def test_result2_bundle_ladder_matches_the_figure(doc):
    """Result 2's M* ladder, reproduced the way the figure computes it.

    This took three attempts to get right, and the reason is worth pinning. The
    sweep entries carry an `rmse` field, but `plot_cost_scaling.get_metrics`
    ignores it: it reads `mse` and `sem_sq` and rebuilds

        bias^2 = mse - sem_sq,  var_single = sem_sq * N_ACC,
        single rmse = sqrt(bias^2 + var_single)

    with N_ACC from the file's own metadata. Reading `rmse` instead gives a
    different ladder, which is how the published one came to say 32 at dim 16
    where the figure says 16.
    """
    pcs = pytest.importorskip("plot_cost_scaling")

    match = re.search(r"ladder runs \$((?:\d+\\to)+\d+)\$", doc)
    assert match, "Result 2's ladder sentence has changed shape"
    published = [int(v) for v in match.group(1).split(r"\to")]

    path = DATA / "cost_scaling_mixed_chain.json"
    if not path.exists():
        pytest.skip("cost_scaling_mixed_chain.json not committed")
    document = json.loads(path.read_text(encoding="utf-8"))
    n_acc = document["meta"]["params"]["N_ACC"]
    m_star = pcs.derive_iso(document["points"], 0.02, "single", "rmse", n_acc)[0]

    measured = [int(v) for v in m_star if v == v]     # drop unreached (NaN)
    assert measured == published, (
        f"document quotes {published}, the figure's own derivation gives "
        f"{measured}")


def test_result5_sampling_row_matches_its_data(doc):
    """Parses section 3.3's row for Result 5. It claimed M = 16/32/64 with 128
    thermal realizations against a run of 8/16/32 with 16."""
    row = re.search(r"\|\s*extreme dim \(Result 5\)\s*\|([^|]*)\|([^|]*)\|", doc)
    assert row, "section 3.3's Result 5 row not found"
    published_m = [int(v) for v in re.findall(r"\d+", row.group(1))]
    published_r = int(re.search(r"\d+", row.group(2)).group())

    path = DATA / "extreme_dimension_mixed_chain_dim256.json"
    if not path.exists():
        pytest.skip("Result 5 data not committed")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert published_m == [s["M"] for s in data["sweep"]]
    assert published_r == data["thermal"]["n_realizations"]


def test_result5_convergence_table_matches_the_frontier_data(doc):
    """Parses Result 5's combined convergence table and recomputes every cell
    from the three frontier data files.

    Rows are keyed on N_L, which is unique across all three systems, so a row
    silently attributed to the wrong system fails rather than passing against
    the wrong file.

    The trends are checked separately from the values. The section's claim is
    that ONLY the oscillator improves -- it must fall by at least three orders
    of magnitude, and neither chain may fall at all. A table that matched every
    file while a trend had reversed would be a different result.
    """
    systems = {}
    for name in ("oscillator_bath", "mixed_chain", "spin_chain"):
        path = DATA / f"frontier_spins_{name}.json"
        if not path.exists():
            pytest.skip(f"frontier sweep for {name} not committed")
        for point in json.loads(path.read_text(encoding="utf-8"))["points"]:
            if point.get("self_convergence"):
                systems[point["n_l"]] = (name, point)

    superscripts = ("\u2070\u00b9\u00b2\u00b3\u2074"
                    "\u2075\u2076\u2077\u2078\u2079")
    sci = r"\**([\d.]+) \u00d7 10\u207b([" + superscripts + r"])\**"
    rows = re.findall(
        r"^\|[^|]*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*(\d+)\s*\|\s*"
        + sci + r"\s*\|\s*" + sci + r"\s*\|",
        doc, re.M)
    assert len(rows) == len(systems), (
        f"table has {len(rows)} rows against {len(systems)} measured points")

    seen = {}
    for dim, n_l, substeps, fro_m, fro_e, tr_m, tr_e in rows:
        n_l = int(n_l.replace(",", ""))
        assert n_l in systems, f"N_L={n_l} is in the table but in no data file"
        name, point = systems[n_l]

        assert int(dim.replace(",", "")) == point["dim"], (
            f"dimension wrong for N_L={n_l}")
        assert int(substeps) == point["substeps"], f"substeps wrong for N_L={n_l}"

        convergence = point["self_convergence"]
        # Three significant figures, so half a unit in the last digit is 0.5%.
        published_fro = float(fro_m) * 10.0 ** -superscripts.index(fro_e)
        published_tr = float(tr_m) * 10.0 ** -superscripts.index(tr_e)
        assert published_fro == pytest.approx(
            convergence["frobenius"], rel=5e-3), f"Frobenius wrong for N_L={n_l}"
        assert published_tr == pytest.approx(
            convergence["trace"], rel=5e-3), f"trace wrong for N_L={n_l}"

        seen.setdefault(name, []).append((point["dim"], convergence["frobenius"]))

    oscillator = [v for _, v in sorted(seen["oscillator_bath"])]
    assert oscillator == sorted(oscillator, reverse=True), (
        f"Result 5 claims the oscillator falls monotonically; got {oscillator}")
    assert oscillator[0] / oscillator[-1] > 1e3, (
        "Result 5 claims four orders of magnitude on the oscillator; measured "
        f"{oscillator[0] / oscillator[-1]:.3g}")

    for name in ("mixed_chain", "spin_chain"):
        values = [v for _, v in sorted(seen[name])]
        assert values[-1] >= values[0] * 0.5, (
            f"Result 5 claims only the oscillator improves, but {name} fell "
            f"from {values[0]:.3g} to {values[-1]:.3g}")
        assert min(values) > 100 * max(oscillator[-1], 1e-30), (
            f"{name} is claimed to stay near 1e-1 while the oscillator reaches "
            f"{oscillator[-1]:.3g}; got {values}")


def _rounding_tolerance(published):
    """Half a unit in the last decimal the document actually prints.

    A fixed relative tolerance cannot serve both "1.47 s" (three significant
    figures, so 0.3% of rounding on its own) and "33,725.5 s" (seven, so
    0.0001%). Deriving it from the printed string means the check is exactly as
    strict as the number claims to be, and gets stricter if the table is ever
    published to more decimals.
    """
    decimals = len(published.split(".")[1]) if "." in published else 0
    return 0.5 * 10.0 ** -decimals


def test_solver_timing_grid_matches_the_three_timing_files(doc):
    """Parses section 5.2's four-solver grid and recomputes every cell.

    Rows are keyed on (dim, N_L), which is unique across all three systems, so
    a row silently attributed to the wrong system fails rather than passing
    against the wrong file.

    The memory table below the grid is checked too, and against the formula
    rather than against itself: the whole point of `N_L * N^4 * 16` is that it
    was fitted to five OOM kills and then predicted three successes. A
    published "predicted" column that had drifted from the formula would make
    that claim unverifiable, which is the failure this file exists to catch.
    """
    measured = {}
    for name in ("spin_chain", "mixed_chain", "oscillator_bath"):
        path = DATA / f"solver_timing_{name}.json"
        if not path.exists():
            pytest.skip(f"solver timing for {name} not committed")
        for point in json.loads(path.read_text(encoding="utf-8"))["points"]:
            measured[(point["dim"], point["n_l"])] = point

    # dim | N_L | native | mesolve | slb | mcsolve-per-trajectory. Cells are
    # either a bolded/plain number with unit, or italic prose (a projection or
    # a divergence), and only the numeric ones are recomputable.
    cell = r"\s*\**\*?([\d,]+\.\d+) s\**\s*|\s*\*[^|]*\*\s*"
    rows = re.findall(
        r"^\|[^|]*\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|(" + cell + r")\|("
        + cell + r")\|(" + cell + r")\|(" + cell + r")\|",
        doc, re.M)
    assert len(rows) == len(measured), (
        f"grid has {len(rows)} rows against {len(measured)} measured points")

    for dim, n_l, _, native, _, mesolve, _, slb, _, mcsolve in rows:
        key = (int(dim), int(n_l.replace(",", "")))
        assert key in measured, f"row dim={dim} N_L={n_l} is in no data file"
        timings = measured[key]["timings"]

        for label, published in (("native", native), ("mesolve", mesolve),
                                 ("slb", slb)):
            entry = timings.get(label, {})
            if not published:                       # italic: projection or divergence
                assert entry.get("skipped") or "diverged" in entry, (
                    f"{label} at dim {dim} is prose in the document but the "
                    f"data holds a real measurement")
                continue
            assert "median_s" in entry, (
                f"{label} at dim {dim} is a number in the document but the "
                f"data records no measurement")
            assert float(published.replace(",", "")) == pytest.approx(
                entry["median_s"], abs=_rounding_tolerance(published)), (
                f"{label} wrong at dim {dim}")

        if mcsolve:
            assert float(mcsolve.replace(",", "")) == pytest.approx(
                timings["mcsolve"]["per_trajectory_s"],
                abs=_rounding_tolerance(mcsolve)), (
                f"mcsolve per-trajectory wrong at dim {dim}")

    # The memory model, checked against the formula it claims to be.
    memory = re.findall(
        r"^\| 19604\d{3} \| [ABC] \| (\d+) \| ([\d,]+) \| ([\d,]+) GB \|",
        doc, re.M)
    assert len(memory) == 3, f"expected 3 memory rows, found {len(memory)}"
    for dim, n_l, predicted in memory:
        expected = int(n_l.replace(",", "")) * int(dim) ** 4 * 16 / 1e9
        assert float(predicted.replace(",", "")) == pytest.approx(
            expected, rel=5e-3), (
            f"published prediction for dim {dim} is {predicted} GB; "
            f"N_L * N^4 * 16 gives {expected:.0f} GB")

# --- 5. section 2, checked against the CODE rather than against data ---------
#
# Everything above compares the document to a data file. These compare it to
# the code, which is the failure mode section 2 actually has: its numbers come
# from build_spin_chain's parameters, MIXED_FIELD_G, build_oscillator_bath's
# anharmonicity and the Davies grouping tolerance. Change one of those and this
# section goes stale with no job having run, so nothing above would notice.

SECTION2_BUILDERS = {
    # label in the document -> (builder, size at dim 16, size at dim 64)
    "A": ("spin_chain", 4, 6),
    "B": ("mixed_chain", 4, 6),
    "C": ("oscillator_bath", 8, 32),
}


def _build(system, size):
    import common
    return {"spin_chain": common.build_spin_chain,
            "mixed_chain": common.build_mixed_field_chain,
            "oscillator_bath": common.build_oscillator_bath}[system](size)


def test_section2_worked_example_matches_the_code(doc):
    """Section 2.6's dimension-16 table, every cell recomputed.

    Parses:

        | **System A** (...) | 62 of 256 (24%) | **13** | 4.8 |

    and recomputes the connected-pair count, N_L and the packing ratio through
    explain_structure.gap_census -- the same function that printed the numbers
    the table was written from.
    """
    import explain_structure as es

    rows = re.findall(
        r"^\|\s*\*\*System ([ABC])\*\*[^|]*\|\s*(\d+) of (\d+)\s*\((\d+)%\)"
        r"\s*\|\s*\*\*([\d,]+)\*\*\s*\|\s*([\d.]+)\s*\|",
        doc, re.M)
    assert len(rows) == 3, (
        f"section 2.6's breakdown table has {len(rows)} parseable rows, not 3")

    for label, connected, pairs, percent, n_l, packing in rows:
        system, size16, _ = SECTION2_BUILDERS[label]
        H, X, _ = _build(system, size16)
        total, _, n_connected, n_connected_gaps = es.gap_census(H, X)

        assert int(pairs) == total, (
            f"System {label}: table says {pairs} level pairs, code gives {total}")
        assert int(connected) == n_connected, (
            f"System {label}: table says {connected} connected pairs, "
            f"code gives {n_connected}")
        assert int(n_l.replace(",", "")) == n_connected_gaps, (
            f"System {label}: table says N_L = {n_l}, code gives "
            f"{n_connected_gaps}")

        assert round(100 * n_connected / total) == int(percent), (
            f"System {label}: table says {percent}% of pairs connect, "
            f"code gives {100 * n_connected / total:.1f}%")
        assert float(packing) == pytest.approx(
            n_connected / n_connected_gaps,
            abs=_rounding_tolerance(packing)), (
            f"System {label}: table says {packing} transitions per operator")


def test_section2_gap_collision_sentence_matches_the_code(doc):
    """Section 2.6 claims System A's 256 transitions carry only 81 distinct gaps
    before symmetry, and that 62 survive symmetry carrying 13.

    That "81" is the whole point of the section -- it is the degeneracy that
    makes System A compress -- and nothing else in the document checks it.
    """
    import explain_structure as es

    match = re.search(
        r"there are (\d+) possible transitions\.[^.]*?"
        r"produce only (\d+) distinct gaps rather than (\d+)", doc, re.S)
    assert match, "section 2.6's gap-collision sentence has changed shape"
    stated_pairs, distinct, repeated_pairs = (int(g) for g in match.groups())

    H, X, _ = _build("spin_chain", 4)
    total, total_gaps, _, _ = es.gap_census(H, X)
    assert stated_pairs == total == repeated_pairs, (
        f"sentence says {stated_pairs}/{repeated_pairs} pairs, code gives {total}")
    assert distinct == total_gaps, (
        f"sentence says {distinct} distinct gaps among all pairs, code gives "
        f"{total_gaps}")


def test_section2_locality_sentence_matches_the_code(doc):
    """Section 2.5: "At dimension 64 this gives 27% (A), 26% (B) and 3.1% (C)
    -- raw distances of 17.4, 16.9 and 2.0 levels."

    Recomputed at dimension 64, where the claim is made. Percentages are d_bar
    divided by N, which is how section 2.6 says to convert them.
    """
    import common
    import explain_structure as es

    match = re.search(
        r"At dimension 64 this gives ([\d.]+)% \(A\), ([\d.]+)% \(B\) and "
        r"([\d.]+)% \(C\)\s*\u2014 raw distances of\s*([\d.]+), ([\d.]+) and "
        r"([\d.]+) levels", doc, re.S)
    assert match, "section 2.5's locality sentence has changed shape"
    percents = match.groups()[:3]
    raws = match.groups()[3:]

    for label, percent, raw in zip("ABC", percents, raws):
        system, _, size64 = SECTION2_BUILDERS[label]
        H, X, _ = _build(system, size64)
        assert H.shape[0] == 64, f"System {label} builder did not give dim 64"
        c_ops = common.build_davies_operators(H, X)
        d_bar = es.mean_offdiagonal_distance(es.transition_weight(H, c_ops))

        assert float(raw) == pytest.approx(
            d_bar, abs=_rounding_tolerance(raw)), (
            f"System {label}: sentence says d_bar = {raw} levels, code gives "
            f"{d_bar:.2f}")
        assert float(percent) == pytest.approx(
            100 * d_bar / 64, abs=_rounding_tolerance(percent)), (
            f"System {label}: sentence says {percent}%, code gives "
            f"{100 * d_bar / 64:.2f}%")


def test_section2_operator_count_law_matches_the_code(doc):
    """Section 2.3 states N_L = n^2 - n + 1 for System A and quotes two values.

    The formula is the reason System A is the control, so it is worth checking
    as a formula and not only at the two sizes the sentence happens to name.
    """
    import common

    match = re.search(r"n\^2-n\+1\}?\$?\s*\((\d+) at dim (\d+), "
                      r"(\d+) at dim (\d+)\)", doc)
    assert match, "section 2.3's N_L formula sentence has changed shape"
    a_nl, a_dim, b_nl, b_dim = (int(g) for g in match.groups())

    for quoted_nl, dim in ((a_nl, a_dim), (b_nl, b_dim)):
        n = int(math.log2(dim))
        H, X, _ = common.build_spin_chain(n)
        assert H.shape[0] == dim
        measured = len(common.build_davies_operators(H, X))
        assert measured == quoted_nl, (
            f"dim {dim}: sentence says N_L = {quoted_nl}, code gives {measured}")
        assert measured == n * n - n + 1, (
            f"dim {dim}: N_L = {measured} breaks the stated n^2-n+1 law "
            f"({n * n - n + 1})")

def test_result3_dim1024_mcsolve_sentence_matches_the_data(doc):
    """Result 3's System A section quotes mcsolve at dimension 1024: an energy
    error, its s.e.m., and a wall-clock. All three recomputed from the file,
    the error through plot_method_comparison.method_errors -- the same scoring
    the figures use -- so the sentence cannot quote a number the figure would
    not."""
    import plot_method_comparison as pmc

    path = DATA / "method_comparison_spin_chain_dim1024.json"
    if not path.exists():
        pytest.skip("dim-1024 mcsolve run not committed")
    point = json.loads(path.read_text(encoding="utf-8"))["point"]
    mc = point["methods"]["mcsolve"]

    match = re.search(
        r"dimension 1024 \(10 spins\).*?energy error ([\d.]+)\u00d710\u207b\u00b2, "
        r"of which\s+([\d.]+)\u00d710\u207b\u00b2 is the sampling s\.e\.m\..*?"
        r"It took \*\*([\d,]+) s\*\*", doc, re.S)
    assert match, "Result 3's dim-1024 mcsolve sentence has changed shape"
    q_err, q_sem, q_wall = match.groups()

    error = [r for r in pmc.method_errors(point, "energy") if r[0] == "mcsolve"][0][2]
    sem = float(np.mean(np.asarray(mc["traj_std"]["energy"]) / np.sqrt(mc["ntraj"])))
    assert float(q_err) == pytest.approx(100 * error, abs=0.05), (
        f"sentence says {q_err}e-2, file gives {100 * error:.2f}e-2")
    assert float(q_sem) == pytest.approx(100 * sem, abs=0.05)
    assert int(q_wall.replace(",", "")) == pytest.approx(mc["wall_s"], abs=0.5)

def test_result3_dim2048_sentence_matches_the_data(doc):
    """Result 3's 11-spin paragraph quotes eight numbers from two files: the
    reference self-check deviation, mcsolve's energy error and s.e.m., both
    wall-clocks, their ratio, and the per-trajectory cost at 11 spins -- all
    from the dim-2048 file, the error through
    plot_method_comparison.method_errors as for the 10-spin sentence -- and
    the per-trajectory cost at 10 spins, from the dim-1024 file."""
    import plot_method_comparison as pmc

    path = DATA / "method_comparison_spin_chain_dim2048.json"
    if not path.exists():
        pytest.skip("dim-2048 run not committed")
    point = json.loads(path.read_text(encoding="utf-8"))["point"]
    mc, nat = point["methods"]["mcsolve"], point["methods"]["native"]

    match = re.search(
        r"dimension 2048 \(11 spins\).*?deviation of ([\d.]+)\u00d710\u207b\u2078"
        r".*?energy error is ([\d.]+)\u00d710\u207b\u00b2, of which ([\d.]+)\u00d710\u207b\u00b2 "
        r"is the sampling s\.e\.m\..*?It\s+took \*\*([\d,]+) s\*\*, against "
        r"\*\*([\d,]+) s\*\* for native RK4.*?\*\*([\d.]+)\u00d7 slower than the exact"
        r".*?\(([\d]+) s to ([\d]+) s\)", doc, re.S)
    assert match, "Result 3's dim-2048 paragraph has changed shape"
    q_dev, q_err, q_sem, q_mc, q_nat, q_ratio, q_traj10, q_traj11 = match.groups()

    dev = point["reference"]["selfcheck"]["max_abs_dev"]
    assert point["reference"]["selfcheck"]["passed"] is True
    assert float(q_dev) == pytest.approx(dev * 1e8, abs=0.05)
    error = [r for r in pmc.method_errors(point, "energy") if r[0] == "mcsolve"][0][2]
    sem = float(np.mean(np.asarray(mc["traj_std"]["energy"]) / np.sqrt(mc["ntraj"])))
    assert float(q_err) == pytest.approx(100 * error, abs=0.005)
    assert float(q_sem) == pytest.approx(100 * sem, abs=0.005)
    assert int(q_mc.replace(",", "")) == pytest.approx(mc["wall_s"], abs=0.5)
    assert int(q_nat.replace(",", "")) == pytest.approx(nat["wall_s"], abs=0.5)
    assert float(q_ratio) == pytest.approx(mc["wall_s"] / nat["wall_s"], abs=0.05)
    assert int(q_traj11) == pytest.approx(mc["wall_s"] / mc["ntraj"], abs=0.5)

    p10 = DATA / "method_comparison_spin_chain_dim1024.json"
    if p10.exists():
        mc10 = json.loads(p10.read_text(encoding="utf-8"))["point"]["methods"]["mcsolve"]
        assert int(q_traj10) == pytest.approx(mc10["wall_s"] / mc10["ntraj"], abs=0.5)


def test_result3_dim2048_growth_is_quoted_at_matched_substeps(doc):
    """The dim-2048 paragraph once set a 4-substep ratio (4.9x) against an
    8-substep one (3.5x at 10 spins) and read a growing gap into it. At
    matched substeps the gap shrank. This pins the corrected comparison AND
    the fact that makes it valid: both denominators really are 8 substeps --
    the dim-2048 reference solve and section 5.2's dim-1024 grid timing."""
    import plot_method_comparison as pmc  # noqa: F401  (same import path as above)

    p11 = DATA / "method_comparison_spin_chain_dim2048.json"
    p10 = DATA / "method_comparison_spin_chain_dim1024.json"
    pgrid = DATA / "solver_timing_spin_chain.json"
    if not (p11.exists() and p10.exists() and pgrid.exists()):
        pytest.skip("dim-1024 / dim-2048 / timing-grid files not all committed")
    point = json.loads(p11.read_text(encoding="utf-8"))["point"]
    mc11, ref11 = point["methods"]["mcsolve"], point["reference"]
    mc10 = json.loads(p10.read_text(encoding="utf-8"))["point"]["methods"]["mcsolve"]
    grid = {p["dim"]: p for p in
            json.loads(pgrid.read_text(encoding="utf-8"))["points"]}[1024]

    assert ref11["selfcheck"]["primary_substeps"] == 8
    assert grid["native_substeps"] == 8, "the 10-spin denominator is no longer 8 substeps"
    g10 = grid["timings"]["native"]["median_s"]

    match = re.search(
        r"8-substep reference solve,\s+\*\*([\d,]+) s\*\*, the ratio is "
        r"\*\*([\d.]+)\u00d7\*\*.*?\(([\d.]+)\u00d7 to ([\d.]+)\u00d7\)\. Its cost per\s+"
        r"trajectory rose ([\d.]+)\u00d7 \((\d+) s to (\d+) s\) while the 8-substep "
        r"exact solve rose ([\d.]+)\u00d7\s+\(([\d,]+) s to ([\d,]+) s\)", doc, re.S)
    assert match, "the dim-2048 matched-substep comparison has changed shape"
    (q_ref, q_r11, q_r10, q_r11b, q_traj_growth, q_t10, q_t11,
     q_exact_growth, q_g10, q_ref_b) = match.groups()

    num = lambda s: float(s.replace(",", ""))
    assert num(q_ref) == pytest.approx(ref11["wall_s"], abs=0.5)
    assert num(q_ref_b) == pytest.approx(ref11["wall_s"], abs=0.5)
    assert num(q_g10) == pytest.approx(g10, abs=0.5)
    assert float(q_r11) == pytest.approx(mc11["wall_s"] / ref11["wall_s"], abs=0.05)
    assert float(q_r11b) == pytest.approx(mc11["wall_s"] / ref11["wall_s"], abs=0.05)
    assert float(q_r10) == pytest.approx(mc10["wall_s"] / g10, abs=0.05)
    t10, t11 = mc10["wall_s"] / mc10["ntraj"], mc11["wall_s"] / mc11["ntraj"]
    assert int(q_t10) == pytest.approx(t10, abs=0.5)
    assert int(q_t11) == pytest.approx(t11, abs=0.5)
    assert float(q_traj_growth) == pytest.approx(t11 / t10, abs=0.05)
    assert float(q_exact_growth) == pytest.approx(ref11["wall_s"] / g10, abs=0.05)
    assert float(q_r11) < float(q_r10), "the paragraph says the gap shrank"
    first = re.search(r"That is not up from the ([\d.]+)\u00d7 at 10 spins", doc)
    assert first, "the dim-2048 paragraph no longer names the 10-spin ratio"
    assert float(first.group(1)) == pytest.approx(mc10["wall_s"] / g10, abs=0.05)


def test_result3_mixed_chain_dim256_paragraph_matches_the_data(doc):
    """Result 3's System B dim-256 paragraph: the reference's self-check and
    its agreement with Result 1's independent reference at the same size,
    both wall-clocks and both ratios (against the 4-substep native solve and
    the 8-substep reference), mcsolve's energy error and s.e.m., and the
    spread of its error/s.e.m. ratio across all six observables against the
    sqrt(2) line -- all through plot_method_comparison.method_errors, the
    scoring the figures use."""
    import plot_method_comparison as pmc

    path = DATA / "method_comparison_mixed_chain_dim256.json"
    r1path = DATA / "accuracy_vs_M_mixed_chain_dim256.json"
    if not (path.exists() and r1path.exists()):
        pytest.skip("System B dim-256 files not committed")
    point = json.loads(path.read_text(encoding="utf-8"))["point"]
    mc, nat, ref = point["methods"]["mcsolve"], point["methods"]["native"], point["reference"]
    r1 = json.loads(r1path.read_text(encoding="utf-8"))

    match = re.search(
        r"dimension 256 \(8 spins.*?deviation of ([\d.]+)\u00d710\u207b\u2079"
        r".*?Result 1's \(job (\d{8})\); the two agree on the energy at \$t=5\$ to all "
        r"eight\s+decimals Result 1's file stores"
        r".*?took \*\*([\d,]+) s\*\*, (\d+) s per trajectory, against \*\*([\d,]+) s\*\*"
        r".*?\*\*([\d.]+)\u00d7 slower than the exact\s+solve\*\* \(([\d.]+)\u00d7 against "
        r"the 8-substep reference solve, ([\d,]+) s\)\. Its energy\s+error is "
        r"([\d.]+)\u00d710\u207b\u00b2, of which ([\d.]+)\u00d710\u207b\u00b2 is the "
        r"sampling s\.e\.m\. \u2014 a ratio of\s+([\d.]+),.*?ratio runs from "
        r"([\d.]+) to ([\d.]+) \u2014 energy, sx and coherence above the line; zz, sz"
        r"\s+and zz_per_bond below it.*?the 38\u00d7 there and the ([\d.]+)\u00d7 here",
        doc, re.S)
    assert match, "Result 3's System B dim-256 paragraph has changed shape"
    (q_dev, q_r1job, q_mc, q_traj, q_nat, q_r_nat, q_r_ref, q_ref, q_err, q_sem,
     q_ratio, q_lo, q_hi, q_here) = match.groups()
    num = lambda s: float(s.replace(",", ""))

    assert ref["selfcheck"]["passed"] is True
    assert float(q_dev) == pytest.approx(ref["selfcheck"]["max_abs_dev"] * 1e9, abs=0.05)
    e3 = float(np.mean(np.atleast_2d(ref["curves"]["energy"]), axis=0)[-1])
    e1 = float(r1["reference"]["energy"][-1])
    # Result 1 stores 8 decimals; "agree to all of them" means this, exactly.
    assert round(e3, 8) == e1, f"Result 3 {e3!r} does not round to Result 1's {e1!r}"
    assert q_r1job == str(r1["meta"]["execution"]["slurm"]["job_id"])
    assert float(q_here) == pytest.approx(mc["wall_s"] / nat["wall_s"], abs=0.05)

    assert num(q_mc) == pytest.approx(mc["wall_s"], abs=0.5)
    assert num(q_traj) == pytest.approx(mc["wall_s"] / mc["ntraj"], abs=0.5)
    assert num(q_nat) == pytest.approx(nat["wall_s"], abs=0.5)
    assert num(q_ref) == pytest.approx(ref["wall_s"], abs=0.5)
    assert float(q_r_nat) == pytest.approx(mc["wall_s"] / nat["wall_s"], abs=0.05)
    assert float(q_r_ref) == pytest.approx(mc["wall_s"] / ref["wall_s"], abs=0.05)

    ratios = {}
    for obs in point["observables"]:
        row = [r for r in pmc.method_errors(point, obs) if r[0] == "mcsolve"][0]
        ratios[obs] = row[2] / row[5]
        if obs == "energy":
            assert float(q_err) == pytest.approx(100 * row[2], abs=0.005)
            assert float(q_sem) == pytest.approx(100 * row[5], abs=0.005)
    assert float(q_ratio) == pytest.approx(ratios["energy"], abs=0.005)
    assert float(q_lo) == pytest.approx(min(ratios.values()), abs=0.005)
    assert float(q_hi) == pytest.approx(max(ratios.values()), abs=0.005)
    above = {o for o, r in ratios.items() if r > math.sqrt(2)}
    assert above == {"energy", "sx", "coherence"}, (
        f"the paragraph names energy, sx and coherence above sqrt(2); data: {sorted(above)}")


def test_result5_reference_wall_sentence_matches_the_data(doc):
    """Result 5 opens by naming, per system, the largest dimension whose
    Result 1 file carries a certified exact reference, and the job behind it.
    This sentence said 512 / 128 / 128 while the data had moved to 1024 / 256
    -- System B's for eleven days -- because nothing read it."""
    match = re.search(
        r"largest dimension carrying an exact reference is \*\*(\d+) on System A\*\* "
        r"\(native RK4 at 8 substeps, job (\d{8})\), \*\*(\d+) on System B\*\* "
        r"\(job (\d{8})\), and \*\*(\d+) on the oscillator\*\*", doc)
    assert match, "Result 5's reference-wall sentence has changed shape"
    a_dim, a_job, b_dim, b_job, c_dim = match.groups()

    def wall(system):
        for dim in sorted(committed_dims(system), reverse=True):
            d = json.loads((DATA / f"accuracy_vs_M_{system}_dim{dim}.json")
                           .read_text(encoding="utf-8"))
            method = d.get("reference_method") or ""
            check = d.get("reference_selfcheck") or {}
            if method.startswith("mesolve") or check.get("passed"):
                job = d.get("meta", {}).get("execution", {}).get("slurm", {}).get("job_id")
                return dim, str(job) if job else None
        return None, None

    for system, q_dim, q_job in (("spin_chain", a_dim, a_job),
                                 ("mixed_chain", b_dim, b_job),
                                 ("oscillator_bath", c_dim, None)):
        dim, job = wall(system)
        assert int(q_dim) == dim, f"{system}: sentence says {q_dim}, data certifies {dim}"
        if q_job is not None:
            assert q_job == job, f"{system}: sentence names job {q_job}, file records {job}"


# --- 6. section 6, which had no guard until its data moved without it --------

SECTION6_PRETTY = {"Spin Chain": "spin_chain", "Oscillator Bath": "oscillator_bath"}
SECTION6_ROW = re.compile(
    r"^\|\s*(Spin Chain|Oscillator Bath)\s*\|\s*(?:dim\s*)?(\d+)\s*\|"
    r"\s*`M\^(-?[\d.]+)`\s*\|"
    r"\s*(?:`M\^(-?[\d.]+)`|\u2014)\s*\|"
    r"\s*(?:([\d.]+)(?:\u2013([\d.]+))?\u00d7|\u2014)\s*\|"
    r"\s*([a-z ]+?)\s*\|\s*$", re.M)


def _section6_panels():
    """(system, dim) -> file, with the dim read from the FILE, not the panel's
    label, so a mislabelled panel cannot check the table against the wrong
    data."""
    import plot_jackknife_rate_strip as strip
    panels = {}
    for key, cfg in strip.SYSTEMS.items():
        for fname, _label in cfg["panels"]:
            d = json.loads((BENCHMARKS / fname).read_text(encoding="utf-8"))
            panels[(key, int(d["dim"]))] = fname
    return strip, panels


def test_section6_jackknife_rate_table_matches_the_progress_files(doc):
    """Section 6's rate table, row by row, recomputed through
    plot_jackknife_rate_strip.fit_stats -- the function the strip figure draws
    with, so the table cannot disagree with its own figure.

    The size-set check is the one that matters most. Section 6 sat at dims
    16/32/64 on pre-0.6.4 data while every other section moved to 512 and
    beyond, and no test noticed, because none existed. This one fails in both
    directions: the plotter moving without the table, or the table without the
    plotter."""
    strip, panels = _section6_panels()
    rows = SECTION6_ROW.findall(doc.translate(MINUS))
    assert rows, "section 6's jackknife rate table is not where this test expects it"

    quoted = {(SECTION6_PRETTY[p], int(d)) for p, d, *_ in rows}
    assert quoted == set(panels), (
        f"the table quotes {sorted(quoted)} but plot_jackknife_rate_strip.py "
        f"draws {sorted(panels)}")

    for pretty, dim, b, jk, lo, hi, verdict in rows:
        fname = panels[(SECTION6_PRETTY[pretty], int(dim))]
        f = strip.fit_stats(json.loads((BENCHMARKS / fname).read_text(encoding="utf-8")))
        where = f"{pretty} dim {dim} ({fname})"
        assert f["b_slope"] == pytest.approx(float(b), abs=0.005), where
        if jk:
            assert f["quotable"], f"{where}: table quotes a slope the 2xSEM/3-point rule refuses"
            assert f["jk_slope"] == pytest.approx(float(jk), abs=0.005), where
        else:
            assert not f["quotable"], f"{where}: table withholds a quotable slope"
        if lo:
            assert f["gain_hi"] == pytest.approx(float(hi or lo), abs=0.05), where
            # A single printed reduction must be the whole range, not its top.
            assert f["gain_lo"] == pytest.approx(float(lo), abs=0.05), where
        assert verdict == f["verdict"], (
            f"{where}: table says {verdict!r}, fit_stats says {f['verdict']!r}")


def test_section6_uncorrected_exponent_range_matches_the_panels(doc):
    """'the fitted bias exponent sits at M^-a to M^-b across all sizes' --
    the min and max of the drawn panels' uncorrected slopes."""
    strip, panels = _section6_panels()
    match = re.search(r"fitted bias exponent sits at \$M\^\{(-[\d.]+)\}\$ to "
                      r"\$M\^\{(-[\d.]+)\}\$\s+across all sizes", doc)
    assert match, "section 6's exponent-range sentence has changed shape"
    slopes = [strip.fit_stats(json.loads((BENCHMARKS / f).read_text(encoding="utf-8")))["b_slope"]
              for f in panels.values()]
    shallow, steep = (float(x) for x in match.groups())
    assert shallow == pytest.approx(max(slopes), abs=0.005)
    assert steep == pytest.approx(min(slopes), abs=0.005)


def test_section6_prose_repeats_the_table_faithfully(doc):
    """The paragraph under section 6's table restates its exponents, the most
    floor-clearing panel, the corrected bias's clearance at dim 512, and the
    growth of both biases across dimension. The table test cannot see any of
    that, and a copy corrected in one place and not the other is this
    project's most common defect."""
    strip, panels = _section6_panels()
    F = {d: strip.fit_stats(json.loads((BENCHMARKS / panels[("spin_chain", d)])
                                       .read_text(encoding="utf-8")))
         for d in (128, 256, 512)}
    m = re.search(r"steepens to\s+\$M\^\{(-[\d.]+)\}\$, \$M\^\{(-[\d.]+)\}\$ and "
                  r"\$M\^\{(-[\d.]+)\}\$ at dims 128, 256 and 512, from\s+uncorrected rates "
                  r"of \$M\^\{(-[\d.]+)\}\$, \$M\^\{(-[\d.]+)\}\$ and \$M\^\{(-[\d.]+)\}\$, "
                  r"and falls up to\s+\$([\d.]+)\\times\$", doc)
    assert m, "section 6's restatement of the table has changed shape"
    q = [float(x) for x in m.groups()]
    for i, d in enumerate((128, 256, 512)):
        assert q[i] == pytest.approx(F[d]["jk_slope"], abs=0.005)
        assert q[3 + i] == pytest.approx(F[d]["b_slope"], abs=0.005)
    assert q[6] == pytest.approx(max(f["gain_hi"] for f in F.values()), abs=0.05)

    m = re.search(r"clears the floor on (\w+) points of\s+(\w+)", doc)
    words = {"three": 3, "four": 4, "five": 5, "six": 6}
    assert m and words[m.group(1)] == F[512]["n_above"] and words[m.group(2)] == len(F[512]["M"])

    m = re.search(r"corrected bias clears its own s\.e\.m\. by \$(\d+)\\times\$ at \$M=2\$ "
                  r"and\s+>?\s*\$(\d+)\\times\$ at \$M=8\$", doc)
    assert m, "section 6's corrected-bias clearance sentence has changed shape"
    ratio = F[512]["bjk"] / F[512]["sem"]
    M = list(F[512]["M"])
    assert int(m.group(1)) == round(ratio[M.index(2)])
    assert int(m.group(2)) == round(ratio[M.index(8)])

    m = re.search(r"uncorrected bias grows ([\d.]+)\u2013([\d.]+)\u00d7 and the corrected "
                  r"bias grows\s+([\d.]+)\u2013([\d.]+)\u00d7", doc)
    assert m, "section 6's growth-across-dimension sentence has changed shape"
    a, c = F[128], F[512]
    both = [M_ for M_ in a["M"] if a["bjk"][list(a["M"]).index(M_)] > 2 * a["sem"][list(a["M"]).index(M_)]
            and c["bjk"][list(c["M"]).index(M_)] > 2 * c["sem"][list(c["M"]).index(M_)]]
    unc = [c["bias"][list(c["M"]).index(x)] / a["bias"][list(a["M"]).index(x)] for x in both]
    jk = [c["bjk"][list(c["M"]).index(x)] / a["bjk"][list(a["M"]).index(x)] for x in both]
    q = [float(x) for x in m.groups()]
    assert (q[0], q[1]) == pytest.approx((min(unc), max(unc)), abs=0.05)
    assert (q[2], q[3]) == pytest.approx((min(jk), max(jk)), abs=0.05)
    assert min(jk) >= min(unc), "the sentence says the corrected bias grows at least as fast"


def test_result1_chains_are_compared_on_their_common_range(doc):
    """Result 1 once called the two chains' height growth the same by setting
    System A's exponent over seven sizes against System B's over five. Each
    exponent falls as its range lengthens, so only a common range compares.
    This pins the matched-range numbers the section now quotes."""
    m = re.search(r"dims 16 to 256, System A grows as \$N\^\{\+([\d.]+)\}\$ and System B as "
                  r"\$N\^\{\+([\d.]+)\}\$ \u2014 a\s+gap of \$([\d.]+)\$", doc)
    assert m, "Result 1's matched-range comparison has changed shape"
    common = sorted(set(committed_dims("spin_chain")) & set(committed_dims("mixed_chain")))
    assert (common[0], common[-1]) == (16, 256), f"common range is now {common}"
    fit = lambda s: linregress(np.log10(common),
                               np.log10([height(s, d) for d in common])).slope
    a, b = fit("spin_chain"), fit("mixed_chain")
    assert float(m.group(1)) == pytest.approx(a, abs=0.005)
    assert float(m.group(2)) == pytest.approx(b, abs=0.005)
    assert float(m.group(3)) == pytest.approx(a - b, abs=0.005)


def test_result1_worst_panel_table_matches_the_figures(doc):
    """The table under Result 1's opening figures -- each system's worst
    plotted panel as a percentage of its span at M = 2, 8, 32 and the top rung,
    and whether the deviation clears three standard errors -- recomputed
    through plot_convergence_dynamics.worst_panel_rows, the module that draws
    those figures, at the size it draws them. The table had no guard; when the
    figures moved from Result 3's data to Result 1's, every number in it
    changed and one verdict flipped."""
    import plot_convergence_dynamics as P

    rows_by_label = {"A": "spin_chain", "B": "mixed_chain", "C": "oscillator_bath"}
    pattern = re.compile(
        r"^\|\s*\*\*([ABC])\*\*[^|]*dim (\d+)\s*\|\s*\**([\d.]+)%\**\s*\|"
        r"\s*\**([\d.]+)%\**\s*\|\s*\**([\d.]+)%\**\s*\|\s*\**([\d.]+)%\**"
        r" at \$M=(\d+)\$\s*\|\s*([^|]+?)\s*\|$", re.M)
    found = pattern.findall(doc)
    assert len(found) == 3, f"expected the three worst-panel rows, found {len(found)}"

    def half_unit(printed):
        decimals = len(printed.split(".")[1]) if "." in printed else 0
        return 0.5 * 10 ** (-decimals) * (1 + 1e-9)

    for label, dim, p2, p8, p32, ptop, mtop, verdict in found:
        system = rows_by_label[label]
        assert int(dim) == P.largest_dim(system), (
            f"{label}: table is at dim {dim}, the figure is drawn at {P.largest_dim(system)}")
        rows = {r["M"]: r for r in P.worst_panel_rows(system, int(dim))}
        top = max(rows)
        assert int(mtop) == top, f"{label}: top rung {mtop}, data tops out at {top}"
        for printed, M in ((p2, 2), (p8, 8), (p32, 32), (ptop, top)):
            assert abs(rows[M]["percent"] - float(printed)) <= half_unit(printed), (
                f"{label} M={M}: measured {rows[M]['percent']:.3f}% does not round to {printed}%")
        # Matched exactly: a substring test once accepted "NOT resolved at
        # every rung" as if it said "resolved at every rung".
        all_resolved = all(r["resolved"] for r in rows.values())
        none_resolved = not any(r["resolved"] for r in rows.values())
        if verdict == "resolved at every rung":
            assert all_resolved, f"{label}: table says resolved at every rung"
        elif verdict == "**never** \u2014 always inside its own scatter":
            assert none_resolved, f"{label}: table says never resolved"
        else:
            pytest.fail(f"{label}: unrecognised verdict {verdict!r}")


def test_result1_worst_panel_prose_matches_the_figures(doc):
    """The paragraphs under the worst-panel table quote numbers the table
    does not: A's N_L and its top-rung coherence, B's smallest clearance, the
    oscillator's largest worst-panel clearance, its energy clearance range, and
    the energy's share of its span. All recomputed from the same files, through
    plot_convergence_dynamics, so the prose cannot drift from its figures."""
    import plot_convergence_dynamics as P

    A = {r["M"]: r for r in P.worst_panel_rows("spin_chain")}
    B = {r["M"]: r for r in P.worst_panel_rows("mixed_chain")}
    C = {r["M"]: r for r in P.worst_panel_rows("oscillator_bath")}
    dA, _ = P.load("spin_chain", P.largest_dim("spin_chain"))

    m = re.search(r"here \$N_L = (\d+)\$ while the ladder stops at\s+(\d+)", doc)
    assert m and int(m.group(1)) == dA["n_l"] and int(m.group(2)) == max(A)
    m = re.search(r"the coherence is still (\d+)% of its span off at \$M=(\d+)\$", doc)
    assert m and A[int(m.group(2))]["panel"] == "coherence"
    assert int(m.group(1)) == round(A[int(m.group(2))]["percent"])

    m = re.search(r"own scatter \u2014 ([\d.]+) standard errors even at \$M=(\d+)\$", doc)
    assert m and int(m.group(2)) == max(B)
    assert float(m.group(1)) == pytest.approx(min(r["z"] for r in B.values()), abs=0.05)
    assert min(B.values(), key=lambda r: r["z"])["M"] == max(B)

    m = re.search(r"200-realization scatter \u2014 ([\d.]+) at most, at \$M=(\d+)\$", doc)
    assert m, "the oscillator's clearance sentence has changed shape"
    top = max(C.values(), key=lambda r: r["z"])
    assert float(m.group(1)) == pytest.approx(top["z"], abs=0.05) and int(m.group(2)) == top["M"]

    dC, _ = P.load("oscillator_bath", P.largest_dim("oscillator_bath"))
    zs, pcts = [], []
    for item in dC["slb_sweep"]:
        ref, s = P._curves(dC, item, "energy")
        mean, sem = s.mean(0), s.std(0, ddof=1) / np.sqrt(s.shape[0])
        dev = np.abs(mean - ref)
        i = int(np.argmax(dev))
        zs.append(dev[i] / sem[i])
        pcts.append(100 * dev[i] / (ref.max() - ref.min()))
    m = re.search(r"Its energy bias \*is\* resolved, at (\d+) to (\d+)\s+standard errors", doc)
    assert m and (int(m.group(1)), int(m.group(2))) == (round(min(zs)), round(max(zs)))
    m = re.search(r"no more\s+than ([\d.]+)% of the energy's span", doc)
    assert m and max(pcts) <= float(m.group(1)) + 0.00005


def test_result1_oscillator_resolution_claims_match_the_decomposition(doc):
    """Result 1 says the oscillator's bias is resolved at dim 128 (53-61x its
    s.e.m. at 200 realizations, 109-120x at 800) and at the floor at dim 64
    (0.2-2.0x). All three ranges recomputed through
    plot_accuracy_vs_M.peak_error_anatomy -- the function that draws the
    error-decomposition figure -- at its t*, so the sentence describes the
    panel that is actually shown.

    This caveat once described dim 64 while the figure showed dim 128. Both
    were true; they were about different panels. Pinning the numbers to the
    plotter's own t* keeps them attached to the right one.
    """
    import plot_accuracy_vs_M as P

    def resolution_range(path):
        d = json.loads(path.read_text(encoding="utf-8"))
        if "reference_energy" in d and "reference" not in d:   # older schema
            d["reference"] = {"energy": d["reference_energy"]}
            for r in d["slb_sweep"]:
                r["samples"] = {"energy": r["samples_energy"]}
        ref = np.asarray(d["reference"]["energy"], dtype=float)
        _, _, bias, fluct = P.peak_error_anatomy(d, "energy", ref, relative=False)
        n = np.asarray(d["slb_sweep"][0]["samples"]["energy"]).shape[0]
        ratio = bias / (fluct / np.sqrt(n))
        return float(ratio.min()), float(ratio.max())

    match = re.search(
        r"resolved at every \$M\$ \u2014 \*\*(\d+) to (\d+)\*\* times its own standard "
        r"error across 200\s+realizations, and \*\*(\d+) to (\d+)\*\* times across 800"
        r".*?\*\*([\d.]+) to ([\d.]+)\*\* standard errors", doc, re.S)
    assert match, "Result 1's oscillator resolution sentence has changed shape"
    q = [float(g) for g in match.groups()]

    small = re.search(r"dims 16 and 32 fare little better at that\s+instant, at "
                      r"([\d.]+) to ([\d.]+) and ([\d.]+) to ([\d.]+)", doc)
    assert small, "Result 1's dims-16-and-32 resolution clause has changed shape"
    s = [float(g) for g in small.groups()]

    for (lo_q, hi_q), name, tol in (
            ((q[0], q[1]), "accuracy_vs_M_oscillator_bath_dim128.json", 0.5),
            ((q[2], q[3]), "accuracy_vs_M_oscillator_bath_dim128_r800.json", 0.5),
            ((q[4], q[5]), "accuracy_vs_M_oscillator_bath_dim64.json", 0.05),
            ((s[0], s[1]), "accuracy_vs_M_oscillator_bath_dim16.json", 0.05),
            ((s[2], s[3]), "accuracy_vs_M_oscillator_bath_dim32.json", 0.05)):
        path = DATA / name
        if not path.exists():
            pytest.skip(f"{name} not committed")
        lo, hi = resolution_range(path)
        assert lo_q == pytest.approx(lo, abs=tol), (
            f"{name}: sentence says min {lo_q}x, decomposition gives {lo:.2f}x")
        assert hi_q == pytest.approx(hi, abs=tol), (
            f"{name}: sentence says max {hi_q}x, decomposition gives {hi:.2f}x")
