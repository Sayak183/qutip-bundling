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
    match = re.match(r"\s*([\d.]+)×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", cell)
    assert match, f"cannot read a value from {cell!r}"
    return float(match.group(1)) * 10 ** int(match.group(2).translate(SUPERSCRIPT))


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

    values = [_decode(c.strip()) for c in row.group(1).split("→")
              if "×10" in c]
    dims = committed_dims(system)
    assert len(values) == len(dims), (
        f"{system}: row quotes {len(values)} values, data has {len(dims)}: {dims}")
    measured = [height(system, d) for d in dims]
    assert measured == pytest.approx(values, rel=0.03)

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

def test_result1_oscillator_resolution_claims_match_the_decomposition(doc):
    """Result 1 says the oscillator's bias is resolved at dim 128 (52-61x its
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

    for (lo_q, hi_q), name, tol in (
            ((q[0], q[1]), "accuracy_vs_M_oscillator_bath_dim128.json", 0.5),
            ((q[2], q[3]), "accuracy_vs_M_oscillator_bath_dim128_r800.json", 0.5),
            ((q[4], q[5]), "accuracy_vs_M_oscillator_bath_dim64.json", 0.05)):
        path = DATA / name
        if not path.exists():
            pytest.skip(f"{name} not committed")
        lo, hi = resolution_range(path)
        assert lo_q == pytest.approx(lo, abs=tol), (
            f"{name}: sentence says min {lo_q}x, decomposition gives {lo:.2f}x")
        assert hi_q == pytest.approx(hi, abs=tol), (
            f"{name}: sentence says max {hi_q}x, decomposition gives {hi:.2f}x")
