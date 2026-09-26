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
    <e|X|e'> is disconnected the limit is Gibbs WITHIN each sector rho_0
    occupies, and at dimension 64 that is -5.6490 for System A against a global
    -5.5687. Recomputed here from plot_extreme_dimension's own helper, so the
    table cannot drift back.
    """
    plot_extreme = pytest.importorskip("plot_extreme_dimension")
    row = re.search(r"\|\s*\*\*Actual t→∞ limit\*\*[^|]*\|([^|]*)\|([^|]*)\|([^|]*)\|",
                    doc)
    assert row, ("the reference table has no 'Actual t→∞ limit' row -- if it was "
                 "removed, global Gibbs is being presented as the limit again")

    for cell, system in zip(row.groups(), REFERENCE_SYSTEMS):
        m = re.search(r"(-?[\d.]+)", cell.translate(MINUS))
        assert m, f"cannot read a number from {cell!r}"
        sector, n_sectors = plot_extreme.sector_resolved_energy(
            _limit_doc(system))
        if sector is None:                 # ergodic: the two targets coincide
            assert n_sectors == 1
            continue
        limit = plot_extreme.sector_limit(_limit_doc(system))
        tol = _rounding_tolerance(m.group(1)) * (1 + 1e-9)
        assert abs(float(m.group(1)) - sector) <= tol, (
            f"{system}: table says {m.group(1)}, sector-resolved limit is "
            f"{sector:.6f}")

        # "(rho0 splits 50/50 over 2 of 7 sectors)": the sector count, and how
        # many of them rho0 touches -- the cell once gave the total count under
        # a label that said "the sector rho0 occupies".
        counts = re.search(r"(\d+) of (\d+) sectors", cell)
        assert counts, f"{system}: the cell no longer says which sectors: {cell!r}"
        occupied = sorted((w for w in limit["weights"] if w > 1e-12), reverse=True)
        assert (int(counts.group(1)), int(counts.group(2))) == (
            len(occupied), n_sectors)
        split = re.search(r"(\d+)/(\d+)", cell)
        if split:                          # whole percent, so half a percent
            assert [float(g) for g in split.groups()] == pytest.approx(
                [100 * w for w in occupied], abs=0.5 + 1e-9)


def _limit_doc(system):
    return {"meta": {"params": {"system": system,
                                "size": SIZE_AT_DIM64[system]}}}


REFERENCE_SYSTEMS = ("spin_chain", "mixed_chain", "oscillator_bath")
# The row prints <label> = value; these are common's names for the operators.
COMPONENT_OPS = {"ZZ": "zz", "X": "sx", "Z": "sz",
                 "n": "n", "n²": "n2", "σᶻ": "sz"}


def test_reference_table_components_are_the_limit_not_global_gibbs(doc):
    """The row under the t→∞ energy once quoted the GLOBAL Gibbs components for
    A and B (<ZZ> = 4.05, <X> = 2.54 for A), which add up to the global energy
    -5.5687, not the limit -5.6490 printed one row above. Every component is
    recomputed here as an expectation value in plot_extreme_dimension's limit
    state -- Gibbs within each sector rho0 occupies, and plain Gibbs for the
    one-sector oscillator."""
    plot_extreme = pytest.importorskip("plot_extreme_dimension")
    import qutip
    row = re.search(r"\|\s*\*\*Components at t→∞\*\*[^|]*\|([^|]*)\|([^|]*)\|"
                    r"([^|]*)\|", doc)
    assert row, "the reference table has no 'Components at t→∞' row"

    for cell, system in zip(row.groups(), REFERENCE_SYSTEMS):
        printed = re.findall(r"⟨([^⟩]+)⟩\s*=\s*(-?[\d.]+)", cell.translate(MINUS))
        assert len(printed) == 3, f"{system}: cannot read three components: {cell!r}"
        H = _build(system, SIZE_AT_DIM64[system])[0]
        labels, ops, _ = common.observable_set(system, H)
        state = plot_extreme.sector_limit(_limit_doc(system))["state"]
        for name, value in printed:
            op = ops[labels.index(COMPONENT_OPS[name])]
            measured = float(np.real(qutip.expect(op, state)))
            assert abs(measured - float(value)) <= (
                _rounding_tolerance(value) * (1 + 1e-9)), (
                f"{system}: table says <{name}> = {value}, the limit gives "
                f"{measured:.4f}")


def test_reference_paragraph_names_the_sectors_rho0_starts_in(doc):
    """The paragraph under the table once said "-5.6490 for A across 7
    sectors", which reads as if rho0 spans all seven. It starts in two."""
    plot_extreme = pytest.importorskip("plot_extreme_dimension")
    pattern = (r"A has (\d+) sectors and starts half in a (\d+)-level one and "
               r"half in a (\d+)-level one; its limit is \$(-[\d.]+)\$\. "
               r"B has (\d+) sectors and starts wholly in the larger one, "
               r"(\d+) of its (\d+) levels; its limit is \$(-[\d.]+)\$")
    match = re.search(pattern.replace(" ", r"\s+"), doc)
    assert match, "the sector sentence under the reference table has changed shape"
    a_n, a_big, a_small, a_e, b_n, b_size, b_dim, b_e = match.groups()

    a = plot_extreme.sector_limit(_limit_doc("spin_chain"))
    a_occupied = [(s, w) for s, w in zip(a["sizes"], a["weights"]) if w > 1e-12]
    assert int(a_n) == a["n_sectors"]
    assert sorted((s for s, _ in a_occupied), reverse=True) == [
        int(a_big), int(a_small)]
    assert [w for _, w in a_occupied] == pytest.approx([0.5, 0.5], abs=1e-9)
    assert abs(a["energy"] - float(a_e)) <= _rounding_tolerance(a_e) * (1 + 1e-9)

    b = plot_extreme.sector_limit(_limit_doc("mixed_chain"))
    b_occupied = [(s, w) for s, w in zip(b["sizes"], b["weights"]) if w > 1e-12]
    assert int(b_n) == b["n_sectors"]
    assert len(b_occupied) == 1 and b_occupied[0][1] == pytest.approx(1.0, abs=1e-9)
    assert b_occupied[0][0] == max(b["sizes"]) == int(b_size)
    assert sum(b["sizes"]) == int(b_dim)
    assert abs(b["energy"] - float(b_e)) <= _rounding_tolerance(b_e) * (1 + 1e-9)


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
    assert path.exists(), f"BENCHMARKS.md quotes {path.name} but it is not committed"
    point = json.loads(path.read_text(encoding="utf-8"))["point"]
    mc = point["methods"]["mcsolve"]

    text = _p6f_region(doc)
    match = re.search(
        r"dimension 1024 \(10 spins\), scored against the archived certified "
        r"reference: energy error ([\d.]+)×10⁻² against a sampling s\.e\.m\. of "
        r"([\d.]+)×10⁻² — ", text)
    assert match, "Result 3's dim-1024 mcsolve sentence has changed shape"
    wall = re.search(r"It\s+took\s+\*\*([\d,]+) s\*\*, against [\d,]+ s for native RK4 "
                     r"at \d+ substeps at the same size", text)
    assert wall, "Result 3's dim-1024 wall-clock sentence has changed shape"
    q_err, q_sem = match.groups()
    q_wall = wall.group(1)

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
    assert path.exists(), f"BENCHMARKS.md quotes {path.name} but it is not committed"
    point = json.loads(path.read_text(encoding="utf-8"))["point"]
    mc, nat = point["methods"]["mcsolve"], point["methods"]["native"]

    text = _p6f_region(doc)
    match = re.search(
        r"dimension 2048 \(11 spins\) — the first exact solve at that size in this "
        r"project\. The reference is native RK4 at \d+ substeps, certified against its "
        r"\d+-substep partner at a deviation of ([\d.]+)×10⁻⁸ \(tolerance "
        r"10⁻⁴\)\. `mcsolve`'s energy error is ([\d.]+)×10⁻² against a sampling "
        r"s\.e\.m\. of ([\d.]+)×10⁻² — ratio ([\d.]+) on the energy; sz and coherence "
        r"land just past the \$\\sqrt\{2\}\$ line \(([\d.]+) and ([\d.]+)\), which for an "
        r"unbiased method is noise, not bias\. It\s+took\s+\*\*([\d,]+) s\*\*, against "
        r"\*\*([\d,]+) s\*\* for native RK4 at \d+ substeps on the same grid in the same "
        r"allocation, so `mcsolve` is \*\*([\d.]+)× slower than the exact solve\*\*",
        text)
    assert match, "Result 3's dim-2048 paragraph has changed shape"
    (q_dev, q_err, q_sem, q_r_energy, q_r_sz, q_r_coh, q_mc, q_nat,
     q_ratio) = match.groups()
    growth = re.search(r"`mcsolve`'s cost per trajectory rose [\d.]+× "
                       r"\((\d+) s to (\d+) s\)", text)
    assert growth, "Result 3's dim-2048 per-trajectory growth has changed shape"
    q_traj10, q_traj11 = growth.groups()

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

    ratios = {}
    for obs in point["observables"]:
        row = next(r for r in pmc.method_errors(point, obs) if r[0] == "mcsolve")
        ratios[obs] = row[2] / row[5]
    assert _near(q_r_energy, ratios["energy"]) and _near(q_r_sz, ratios["sz"])
    assert _near(q_r_coh, ratios["coherence"])
    past = {o for o, r in ratios.items()
            if o != "zz_per_bond" and r > pmc.BIAS_LIMITED_RATIO}
    assert past == {"sz", "coherence"}, f"past the sqrt(2) line: {sorted(past)}"

    p10 = DATA / "method_comparison_spin_chain_dim1024.json"
    assert p10.exists(), f"BENCHMARKS.md quotes {p10.name} but it is not committed"
    mc10 = json.loads(p10.read_text(encoding="utf-8"))["point"]["methods"]["mcsolve"]
    assert int(q_traj10) == pytest.approx(mc10["wall_s"] / mc10["ntraj"], abs=0.5)


def test_result3_dim2048_growth_is_quoted_at_matched_substeps(doc):
    """The dim-2048 paragraph quotes 4.9x against the 4-substep native solve in
    its own job, and 2.5x against its own 8-substep reference. It names the
    10-spin 3.5x (also against 8 substeps) but claims no trend from it: that
    ratio divides by a solve timed in another job on another node, and the
    3.5x-to-2.5x change (1.4x) is under section 7's ~1.5x resolution for single
    timings. The 1.4x is exactly the gap between the two quoted growth factors.
    An earlier version said the gap "shrank"; this pins the reasons it may not,
    so the claim cannot come back while they hold."""
    p11 = DATA / "method_comparison_spin_chain_dim2048.json"
    p10 = DATA / "method_comparison_spin_chain_dim1024.json"
    pgrid = DATA / "solver_timing_spin_chain.json"
    for path in (p11, p10, pgrid):
        assert path.exists(), f"BENCHMARKS.md quotes {path.name} but it is not committed"
    d11 = json.loads(p11.read_text(encoding="utf-8"))
    d10 = json.loads(p10.read_text(encoding="utf-8"))
    dgrid = json.loads(pgrid.read_text(encoding="utf-8"))
    point = d11["point"]
    mc11, nat11, ref11 = (point["methods"]["mcsolve"], point["methods"]["native"],
                          point["reference"])
    mc10 = d10["point"]["methods"]["mcsolve"]
    grid = {p["dim"]: p for p in dgrid["points"]}[1024]
    g10 = grid["timings"]["native"]["median_s"]

    match = re.search(
        r"That ([\d.]+)× is not comparable with the ([\d.]+)× at 10 spins, which "
        r"was taken against an (\d+)-substep solve\. Against this job's own (\d+)-substep "
        r"reference solve, \*\*([\d,]+) s\*\*, the ratio is \*\*([\d.]+)×\*\*\. Even so, "
        r"no trend from 10 spins is claimed \(([\d.]+)× to ([\d.]+)×\): the 10-spin "
        r"([\d.]+)× divides by an exact solve timed in another job on another node "
        r"\(§5\.3\), and a ([\d.]+)× change is inside the ~([\d.]+)× that single "
        r"timings on a shared node cannot resolve \(§7\)\. That ([\d.]+)× is the gap "
        r"between two growth factors, each spanning two jobs: `mcsolve`'s cost per "
        r"trajectory rose ([\d.]+)× \((\d+) s to (\d+) s\), and the (\d+)-substep exact "
        r"solve rose ([\d.]+)× \(([\d,]+) s to ([\d,]+) s\)\.", _flat(doc))
    assert match, "the dim-2048 matched-substep comparison has changed shape"
    (q_r4, q_r10, q_sub10, q_sub11, q_ref, q_r11, q_r10b, q_r11b, q_r10c, q_change,
     q_res, q_change_b, q_traj_growth, q_t10, q_t11, q_sub_exact, q_exact_growth,
     q_g10, q_ref_b) = match.groups()

    # 4.9x: mcsolve over the 4-substep native solve, same job.
    assert d11["meta"]["substeps"] == 4
    assert _near(q_r4, mc11["wall_s"] / nat11["wall_s"])
    # Both 8-substep denominators really are 8 substeps.
    assert int(q_sub10) == int(q_sub11) == int(q_sub_exact) == 8
    assert grid["native_substeps"] == 8, "the 10-spin denominator is no longer 8 substeps"
    assert ref11["selfcheck"]["primary_substeps"] == 8
    r10, r11 = mc10["wall_s"] / g10, mc11["wall_s"] / ref11["wall_s"]
    assert _near(q_r10, r10) and _near(q_r10b, r10) and _near(q_r10c, r10)
    assert _near(q_r11, r11) and _near(q_r11b, r11)
    assert _near(q_ref, ref11["wall_s"]) and _near(q_ref_b, ref11["wall_s"])
    assert _near(q_g10, g10)
    # Why no trend is claimed: another job on another node, one timing each,
    # and a change smaller than section 7's resolution.
    job = lambda d: (d["meta"]["execution"]["slurm"]["job_id"],
                     d["meta"]["execution"]["hostname"])
    assert job(dgrid)[0] != job(d10)[0] and job(dgrid)[1] != job(d10)[1]
    assert dgrid["meta"]["params"]["repeats"] == 1
    assert len(grid["timings"]["native"]["samples_s"]) == 1
    assert len(mc10["wall_s_repeats"]) == len(mc11["wall_s_repeats"]) == 1
    assert _near(q_change, r10 / r11) and _near(q_change_b, r10 / r11)
    res = re.search(r"ratios below ~([\d.]+)x are not resolved\s+without repeats", _flat(doc))
    assert res, "section 7's timing-resolution note has changed shape"
    assert float(q_res) == float(res.group(1))
    assert r10 / r11 < float(q_res), "the change now exceeds the stated resolution"
    t10, t11 = mc10["wall_s"] / mc10["ntraj"], mc11["wall_s"] / mc11["ntraj"]
    assert _near(q_t10, t10) and _near(q_t11, t11)
    assert _near(q_traj_growth, t11 / t10)
    assert _near(q_exact_growth, ref11["wall_s"] / g10)
    # the 1.4x is the ratio of the two growth factors, not a separate number
    assert math.isclose((ref11["wall_s"] / g10) / (t11 / t10), r10 / r11, rel_tol=1e-9)
    region = _flat(doc).split("#### System A — TFIM chain (dim 64")[1].split("### Result 4")[0]
    assert "shrank" not in region, "Result 3's System A text reads a trend into cross-job timings"


def test_result3_mixed_chain_dim256_paragraph_matches_the_data(doc):
    """Result 3's System B dim-256 paragraph: the reference's self-check and
    its agreement with Result 1's independent reference at the same size,
    both wall-clocks and both ratios (against the 4-substep native solve and
    the 8-substep reference), mcsolve's energy error and s.e.m., and the
    spread of its error/s.e.m. ratio across the five distinct observables
    (zz_per_bond repeats zz) against the sqrt(2) line -- all through
    plot_method_comparison.method_errors, the scoring the figures use. The
    result comes first and its provenance second (review part 6, U4)."""
    import plot_method_comparison as pmc

    path = DATA / "method_comparison_mixed_chain_dim256.json"
    r1path = DATA / "accuracy_vs_M_mixed_chain_dim256.json"
    assert path.exists() and r1path.exists(), "the dim-256 paragraph names both files"
    document = json.loads(path.read_text(encoding="utf-8"))
    point = document["point"]
    mc, nat, ref = point["methods"]["mcsolve"], point["methods"]["native"], point["reference"]
    r1 = json.loads(r1path.read_text(encoding="utf-8"))

    # The result first, then where it came from (review part 6, U4).
    match = re.search(
        r"At dimension 256 \(8 spins, \$N_L = ([\d{},]+)\$\), `mcsolve` at (\d+) "
        r"trajectories took \*\*([\d,]+) s\*\*, (\d+) s per trajectory, against "
        r"\*\*([\d,]+) s\*\* for native RK4 at (\d+) substeps in the same allocation: "
        r"\*\*([\d.]+)× slower than the exact solve\*\* \(([\d.]+)× against "
        r"the (\d+)-substep reference solve, ([\d,]+) s\)\. Its energy error is "
        r"([\d.]+)×10⁻² and its sampling s\.e\.m\. "
        r"([\d.]+)×10⁻², a ratio of ([\d.]+), just past the "
        r"\$\\sqrt\{2\}\$ line above\. Across the (\w+) distinct observables the ratio "
        r"runs from ([\d.]+) to ([\d.]+): energy, sx and coherence above the line, zz "
        r"and sz below it\. `mcsolve` has essentially no bias \(§4\), so a point above the "
        r"line here is noise that landed more than one s\.e\.m\. from the reference, "
        r"not a bias\. No SLB ran at this size, so no SLB/`mcsolve` ratio is quoted\. "
        r"These numbers come from job (\d{8}), which also ran a fresh certified "
        r"reference: native RK4 at (\d+) substeps, certified against its (\d+)-substep "
        r"partner at a deviation of ([\d.]+)×10⁻⁹\. It is the second "
        r"certified exact solve at this size, after Result 1's \(job (\d{8})\); the two "
        r"agree on the energy at \$t=5\$ to all eight decimals Result 1's file stores\. "
        r"The job ran on (\d+) threads where dims 4–128 ran on (\d+) \(job (\d{8})\), "
        r"so its wall-clocks are not drawn on the figures above and no growth from dim "
        r"128 is quoted: the 38× there and the ([\d.]+)× here",
        _flat(doc))
    assert match, "Result 3's System B dim-256 paragraph has changed shape"
    (q_nl, q_ntraj, q_mc, q_traj, q_nat, q_subs, q_r_nat, q_r_ref, q_ref_subs, q_ref,
     q_err, q_sem, q_ratio, q_count, q_lo, q_hi, q_job, q_ref_subs2, q_pair, q_dev,
     q_r1job, q_threads, q_threads128, q_job128, q_here) = match.groups()
    assert _printed(q_nl) == point["n_l"] and int(q_ntraj) == mc["ntraj"]
    assert int(q_subs) == document["meta"]["substeps"]
    assert int(q_ref_subs) == int(q_ref_subs2) == document["meta"]["params"]["ref_substeps"]
    assert ref["selfcheck"]["substeps_pair"] == sorted([int(q_pair), int(q_ref_subs)])
    assert q_job == str(document["meta"]["execution"]["slurm"]["job_id"])
    assert int(q_threads) == int(document["meta"]["execution"]["threads"]["OMP_NUM_THREADS"])
    for dim in (4, 8, 16, 32, 64, 128):
        other = json.loads((DATA / f"method_comparison_mixed_chain_dim{dim}.json")
                           .read_text(encoding="utf-8"))["meta"]["execution"]
        assert str(other["slurm"]["job_id"]) == q_job128, dim
        assert int(other["threads"]["OMP_NUM_THREADS"]) == int(q_threads128), dim
    assert {"five": 5}[q_count] == len(point["observables"]) - 1

    # The superseded wording (provenance first, "of which", six observables)
    # must not come back.
    superseded = re.search(
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
    assert superseded is None, "the superseded dim-256 wording is back"
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
    assert ratios["zz_per_bond"] == pytest.approx(ratios["zz"], rel=1e-9), (
        "Result 3 says zz_per_bond repeats zz's ratio")
    distinct = {o: r for o, r in ratios.items() if o != "zz_per_bond"}
    above = {o for o, r in distinct.items() if r > math.sqrt(2)}
    assert above == {"energy", "sx", "coherence"}, (
        f"the paragraph names energy, sx and coherence above sqrt(2); data: {sorted(above)}")
    assert set(distinct) - above == {"zz", "sz"}, "the paragraph names zz and sz below it"


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


def test_intro_run_script_safety_sentence_matches_the_code(doc):
    """The intro says no run_*.py replaces existing JSON without --overwrite,
    and that all but one also need --system/--all and support --dry-run. Two
    runners once started a System B run when given no arguments, and that run
    would overwrite committed data without asking, while this sentence said
    otherwise. So it is checked against the source rather than trusted."""
    flat = " ".join(doc.split())      # immune to re-wrapping the paragraph
    match = re.search(
        r"None of them will replace an existing JSON file unless "
        r"`--overwrite` is given\. All but one also need an explicit "
        r"`--system` or `--all` and support `--dry-run`\. The exception is "
        r"`(run_\w+\.py)`: it runs only the spin chain, takes a required "
        r"`--dim` instead, and has no dry run\.", flat)
    assert match, "the intro's run-script safety sentence has changed shape"
    exception = match.group(1)

    runners = sorted(p.name for p in BENCHMARKS.glob("run_*.py"))
    source = {name: (BENCHMARKS / name).read_text(encoding="utf-8")
              for name in runners}
    guarded = [name for name in runners
               if "add_safety_arguments(" in source[name]
               and "preflight_run(" in source[name]]
    assert sorted(set(runners) - set(guarded)) == [exception], (
        "every run_*.py except the one the intro names must use the "
        "benchmark_cli guard")

    # The exception's own promises: an --overwrite check, a required --dim,
    # no --dry-run, and the spin chain only.
    text = source[exception]
    assert '"--overwrite"' in text and "pass --overwrite" in text
    assert '"--dim", type=int, required=True' in text
    assert "--dry-run" not in text
    assert "build_spin_chain" in text
    assert "build_mixed_field_chain" not in text
    assert "build_oscillator_bath" not in text

    # smoke_test.py runs every guarded runner with no scope and with
    # --all --dry-run; a runner missing from its list is never exercised.
    smoke = pytest.importorskip("smoke_test")
    assert sorted(smoke.RUNNERS) == guarded, (
        "smoke_test.RUNNERS must list every guarded runner")


# --- 7. section 5's opening summary, and the Result 2 copies of its numbers --
#
# The summary at the top of section 5 restates numbers from every Result. It
# printed System A's mesolve slope as System B's, an exponent from superseded
# shared-node timings, a speed ratio with no substep caveat, a memory wall on
# the wrong system and gibibytes labelled as gigabytes -- all of it unpinned.
# Every number below is recomputed through the module that draws its figure.

def _flat(doc: str) -> str:
    """The document with line breaks and blockquote markers folded to single
    spaces, so a sentence matches however it is wrapped."""
    return re.sub(r"\s*\n(?:>\s*)?", " ", doc)


def _printed(value: str) -> float:
    """'1{,}013' or '3,249' -> 1013.0 / 3249.0."""
    return float(value.replace("{,}", "").replace(",", ""))


def _near(printed: str, measured: float) -> bool:
    """True when `measured` rounds to `printed` in its last printed digit."""
    return abs(_printed(printed) - measured) <= _rounding_tolerance(printed) * (1 + 1e-9)


def _cost_curves(system: str):
    """(document, dims, native, full, iso, m_star) exactly as
    plot_cost_scaling.figure builds them for its default single-run view."""
    pcs = pytest.importorskip("plot_cost_scaling")
    path = DATA / f"cost_scaling_{system}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    document = json.loads(path.read_text(encoding="utf-8"))
    points = document["points"]
    as_array = common.as_array
    dims = as_array([p["dim"] for p in points])
    native = as_array([p.get("t_native_ref") for p in points])
    full = as_array([p.get("t_full") for p in points])
    m_star, iso = pcs.derive_iso(points, pcs.TARGET_BY_SYSTEM[system],
                                 pcs.ESTIMATE_TYPE, pcs.ERROR_TYPE,
                                 document["meta"]["params"]["N_ACC"])[:2]
    return document, dims, native, full, iso, m_star


def _fit(dims, times):
    """plot_cost_scaling.fit_slope, plus the dimensions it used: the fit runs
    over a suffix of the finite points, so they are the last n of them."""
    import plot_cost_scaling as pcs
    slope, n = pcs.fit_slope(dims, times)
    used = dims[np.isfinite(times)][-n:]
    return slope, n, int(used[0]), int(used[-1])


# "Result 3's jobs time both on this chain: 231 s against 115 s at dim 64 and
# 4,819 s against 2,413 s at dim 128, 2.0x each" -- the reference and the native
# solve from the same job, so the reader can see the factor the ratios carry.
_MARGIN_CLAUSE = (r"Result 3's jobs time both on this chain: ([\d,]+) s against "
                  r"([\d,]+) s at dim 64 and ([\d,]+) s against ([\d,]+) s at dim "
                  r"128, ([\d.]+)x each")


def _substep_margin(dim: int) -> float:
    """Result 3's measured cost of the 8-substep reference over the 4-substep
    native solve, same job, System B -- the factor Result 2's ratios carry."""
    path = DATA / f"method_comparison_mixed_chain_dim{dim}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    document = json.loads(path.read_text(encoding="utf-8"))
    point = document["point"]
    assert point["reference"]["selfcheck"]["primary_substeps"] == 8
    assert document["meta"]["substeps"] == 4
    return point["reference"]["wall_s"] / point["methods"]["native"]["wall_s"]


def test_result2_mixed_chain_exponents_match_the_figure(doc):
    """The summary quoted N^4.6 for System B's exact solver -- a leftover of
    the shared-node timings section 5.3 retired -- and N^6.0, which is System
    A's mesolve slope. Result 2 repeated the 4.6. Pinned here through
    plot_cost_scaling.fit_slope, the function that writes the figure's legend,
    over the same dimensions it fits, and with the matched-range comparison
    that makes the two exponents comparable at all."""
    document, dims, native, full, iso, _ = _cost_curves("mixed_chain")
    # The figure draws native only where certified; on this file that is
    # every point, so the fit below is the legend's fit.
    for p in document["points"]:
        if p.get("t_native_ref") is not None:
            assert (p.get("reference_method") == "mesolve"
                    or (p.get("native_ref_selfcheck") or {}).get("passed")), (
                f"dim {p['dim']}: an uncertified native point the legend omits")

    text = _flat(doc)
    import plot_cost_scaling as pcs
    target = re.search(
        r"SLB at matched accuracy \(energy RMSE ([\d.]+) for one run, estimated "
        r"from (\d+) realizations\) grows as", text)
    assert target, "section 5's Result 2 accuracy target has changed shape"
    assert float(target.group(1)) == pcs.TARGET_BY_SYSTEM["mixed_chain"]
    assert int(target.group(2)) == document["meta"]["params"]["N_ACC"]
    assert pcs.ESTIMATE_TYPE == "single", "the figure no longer costs one run"
    head = re.search(
        r"realizations\) grows as \$N\^\{([\d.]+)\}\$ over dims (\d+) to "
        r"(\d+), against \$N\^\{([\d.]+)\}\$ for the certified native exact solver "
        r"over the same sizes \(\$N\^\{([\d.]+)\}\$ over dims (\d+) to (\d+), the "
        r"slope in the figure's legend\)\. Superoperator `mesolve` has a two-point "
        r"local slope of \$N\^\{([\d.]+)\}\$ \(dims (\d+) and (\d+); smaller sizes "
        r"fall under the ([\d.]+) s fitting floor\)", text)
    assert head, "section 5's Result 2 exponent sentence has changed shape"
    body = re.search(
        r"energy iso-accuracy cost near \$N\^\{([\d.]+)\}\$ over dims (\d+) to "
        r"(\d+), against \$N\^\{([\d.]+)\}\$ for the exact solve over the same "
        r"sizes \(\$N\^\{([\d.]+)\}\$ over dims (\d+) to (\d+)\)", text)
    assert body, "Result 2's System B iso-accuracy sentence has changed shape"

    s_iso, _, iso_lo, iso_hi = _fit(dims, iso)
    s_nat, n_nat, nat_lo, nat_hi = _fit(dims, native)
    window = (dims >= iso_lo) & (dims <= iso_hi)
    s_same, n_same = pcs.fit_slope(dims[window], native[window])
    assert n_same == int(window.sum()), "the matched-range fit dropped a point"
    s_full, n_full, full_lo, full_hi = _fit(dims, full)
    assert n_full == 2, "mesolve now has more than two fitted points; say so"

    for match in (head, body):
        g = match.groups()
        assert _near(g[0], s_iso), f"iso exponent {g[0]} against {s_iso:.3f}"
        assert (int(g[1]), int(g[2])) == (iso_lo, iso_hi)
        assert _near(g[3], s_same), f"matched-range exact {g[3]} against {s_same:.3f}"
        assert _near(g[4], s_nat), f"full-range exact {g[4]} against {s_nat:.3f}"
        assert (int(g[5]), int(g[6])) == (nat_lo, nat_hi)
    g = head.groups()
    assert _near(g[7], s_full), f"mesolve slope {g[7]} against {s_full:.3f}"
    assert (int(g[8]), int(g[9])) == (full_lo, full_hi)
    assert float(g[10]) == pcs.FIT_FLOOR_SECONDS


def test_result2_mesolve_slopes_are_attributed_to_the_right_system(doc):
    """'N^6.0 on the chain' was System A's number under a sentence a reader
    took to mean System B. All three are now named, each through fit_slope."""
    m = re.search(
        r"steepest on each plot: \$N\^\{([\d.]+)\}\$ on System A, \$N\^\{([\d.]+)\}\$ "
        r"on System B and \$N\^\{([\d.]+)\}\$ on the oscillator — all two-point "
        r"local slopes, through dims (\d+) and (\d+)\.", _flat(doc))
    assert m, "Result 2's mesolve-slope sentence has changed shape"
    for printed, system in zip(m.groups()[:3],
                               ("spin_chain", "mixed_chain", "oscillator_bath")):
        _, dims, _, full, _, _ = _cost_curves(system)
        slope, n, lo, hi = _fit(dims, full)
        assert n == 2, f"{system}: mesolve fit uses {n} points, not two"
        assert (lo, hi) == (int(m.group(4)), int(m.group(5))), f"{system}: dims {lo}-{hi}"
        assert _near(printed, slope), f"{system}: printed {printed}, fit {slope:.3f}"


def test_result2_speed_ratios_state_their_substeps(doc):
    """353x and 1,013x compare an 8-substep exact solve with a 4-substep SLB
    solve. Both copies now say so and give the matched-substep estimate, using
    Result 3's same-job measurement of what the extra substeps cost."""
    document, dims, native, _, iso, m_star = _cost_curves("mixed_chain")
    meta = document["meta"]
    assert meta["params"]["NATIVE_REF_SUBSTEPS"] == 8 and meta["substeps"] == 4
    by_dim = {p["dim"]: p for p in document["points"]}
    ratio = {d: by_dim[d]["t_native_ref"] / by_dim[d]["t_slb_fixed"] for d in (64, 128)}
    margin = {d: _substep_margin(d) for d in (64, 128)}

    text = _flat(doc)
    head = re.search(
        r"Measured directly, the certified exact solve \(native RK4 at (\d+) "
        r"substeps\) costs \$(\d+)\\times\$ one SLB solve \(\$M=(\d+)\$, (\d+) "
        r"substeps\) at dim 64 and \$([\d{},]+)\\times\$ at dim 128\. Twice the "
        r"substeps doubles the exact solve's cost \(" + _MARGIN_CLAUSE + r"\), so at "
        r"matched substeps the gap is about \$(\d+)\\times\$ and \$(\d+)\\times\$\.", text)
    assert head, "section 5's Result 2 speed-ratio sentence has changed shape"
    (q_ref, q64, q_m, q_slb, q128, *q_secs, q_margin, q_m64, q_m128) = head.groups()
    assert int(q_ref) == meta["params"]["NATIVE_REF_SUBSTEPS"]
    assert int(q_slb) == meta["substeps"] and int(q_m) == meta["params"]["M_REP"]

    body = re.search(
        r"certified exact solve \(native RK4 at 8 substeps\) takes \*\*(\d+) "
        r"minutes\*\* against \*\*([\d.]+) s\*\* for one SLB solve at \$M=8\$ and 4 "
        r"substeps — \*\*\$([\d{},]+)\\times\$\*\*, widening from "
        r"\$(\d+)\\times\$ at dim 64\. Twice the substeps doubles the exact solve's "
        r"cost \(" + _MARGIN_CLAUSE + r"\), so at matched "
        r"substeps the gap is about \$(\d+)\\times\$ at dim 128 and "
        r"\$(\d+)\\times\$ at dim 64\.", text)
    assert body, "Result 2's System B speed-ratio bullet has changed shape"
    (b_min, b_slb, b128, b64, *b_secs, b_margin, b_m128, b_m64) = body.groups()
    for secs in (q_secs, b_secs):
        for (ref_s, nat_s), dim in zip((secs[0:2], secs[2:4]), (64, 128)):
            point = json.loads((DATA / f"method_comparison_mixed_chain_dim{dim}.json")
                               .read_text(encoding="utf-8"))["point"]
            assert _near(ref_s, point["reference"]["wall_s"]), (ref_s, dim)
            assert _near(nat_s, point["methods"]["native"]["wall_s"]), (nat_s, dim)

    for printed_64, printed_128 in ((q64, q128), (b64, b128)):
        assert _near(printed_64, ratio[64]), f"{printed_64} against {ratio[64]:.2f}"
        assert _near(printed_128, ratio[128]), f"{printed_128} against {ratio[128]:.2f}"
    for printed in (q_margin, b_margin):
        assert all(_near(printed, m) for m in margin.values()), (
            f"margin printed {printed}, measured {margin}")
    for printed_64, printed_128 in ((q_m64, q_m128), (b_m64, b_m128)):
        assert _near(printed_64, ratio[64] / margin[64])
        assert _near(printed_128, ratio[128] / margin[128])
    assert _near(b_min, by_dim[128]["t_native_ref"] / 60)
    assert _near(b_slb, by_dim[128]["t_slb_fixed"])
    job = re.search(r"Complete to dimension 128 \(job (\d{8}), run on an exclusive node\)", text)
    assert job and job.group(1) == str(meta["execution"]["slurm"]["job_id"])

    # Section 5.3 quotes the same ratios, and the oscillator's, before and after
    # the re-timing; it says they carry the 2x margin. True on both systems.
    assert re.search(r"All of these ratios run the exact solve at twice SLB's "
                     r"substeps \(§5\.1\); at matched substeps each gap is "
                     r"about half as wide\.", text), "section 5.3's caveat is gone"
    oscillator = json.loads((DATA / "cost_scaling_oscillator_bath.json")
                            .read_text(encoding="utf-8"))["meta"]
    assert oscillator["params"]["NATIVE_REF_SUBSTEPS"] == 2 * oscillator["substeps"]

    # The iso-accuracy cost at dim 128 against the same exact solve.
    iso_line = re.search(
        r"At dim 128 one SLB run at \$M\^\\ast=(\d+)\$ \(its error estimated from "
        r"(\d+) realizations\) costs ([\d.]+) s against ([\d,]+) s for the 8-substep "
        r"exact solve: \$(\d+)\\times\$, or about \$(\d+)\\times\$ at matched "
        r"substeps\.", text)
    assert iso_line, "Result 2's System B dim-128 iso-cost sentence has changed shape"
    q_mstar, q_nacc, q_iso, q_nat, q_r, q_rm = iso_line.groups()
    assert dims[-1] == 128
    assert int(q_mstar) == int(m_star[-1])
    assert int(q_nacc) == meta["params"]["N_ACC"]
    assert _near(q_iso, iso[-1]) and _near(q_nat, native[-1])
    assert _near(q_r, native[-1] / iso[-1])
    assert _near(q_rm, native[-1] / iso[-1] / margin[128])


def test_section5_summary_result3_ratios_match_the_data(doc):
    """The summary called 914x and 8.3x an 'advantage' without saying they are
    error ratios at a fixed budget, and moved from dim 64 at M=16 to dim 128 at
    M=256 in one step. Every ratio is recomputed through
    plot_method_comparison.method_errors, the figures' own scoring."""
    import plot_method_comparison as pmc

    m = re.search(
        r"SLB at \$M=(\d+)\$ with (\d+) realizations, `mcsolve` with (\d+) "
        r"trajectories\), SLB ranges from (\d+)x more accurate than `mcsolve` "
        r"\(oscillator energy\) to ([\d.]+)x less accurate \(mixed-chain "
        r"coherence\)\. These are error ratios, not speedups\. The weak case is a "
        r"setting rather than a property: at dim 128 that coherence gap is "
        r"([\d.]+)x at \$M=16\$ and ([\d.]+)x at \$M=256\$, while the energy goes "
        r"from ([\d.]+)x worse to ([\d.]+)x better", _flat(doc))
    assert m, "section 5's Result 3 summary item has changed shape"
    q_m, q_runs, q_traj, q_osc, q_coh64, q_coh16, q_coh256, q_e16, q_e256 = m.groups()

    def pair(system, dim, observable, bundles):
        path = DATA / f"method_comparison_{system}_dim{dim}.json"
        if not path.exists():
            pytest.skip(f"{path.name} not committed")
        point = json.loads(path.read_text(encoding="utf-8"))["point"]
        rows = pmc.method_errors(point, observable)
        mc = next(r for r in rows if r[0] == "mcsolve")
        slb = next(r for r in rows if r[0] == "slb" and r[3] == f"M={bundles}")
        assert mc[4] == int(q_traj), f"{system} dim {dim}: mcsolve ran {mc[4]} trajectories"
        assert slb[4] == int(q_runs), f"{system} dim {dim}: SLB ran {slb[4]} realizations"
        return slb[2], mc[2]

    assert int(q_m) == 16
    slb, mc = pair("oscillator_bath", 64, "energy", 16)
    assert _near(q_osc, mc / slb)
    slb, mc = pair("mixed_chain", 64, "coherence", 16)
    assert _near(q_coh64, slb / mc)
    slb, mc = pair("mixed_chain", 128, "coherence", 16)
    assert _near(q_coh16, slb / mc)
    slb, mc = pair("mixed_chain", 128, "coherence", 256)
    assert _near(q_coh256, slb / mc)
    slb, mc = pair("mixed_chain", 128, "energy", 16)
    assert _near(q_e16, slb / mc)
    slb, mc = pair("mixed_chain", 128, "energy", 256)
    assert _near(q_e256, mc / slb)


def test_section5_summary_result4_speedups_match_the_figures(doc):
    """394x and 850x are against mcsolve's PROJECTED cost -- the trajectory
    count the 3% target needs times a measured per-trajectory cost -- with SLB
    as a 16-realization ensemble. Recomputed through plot_isocost_vs_dim.derive
    exactly as its main() calls it."""
    P = pytest.importorskip("plot_isocost_vs_dim")
    from isocost_config import run_counts

    m = re.search(
        r"reach the \*same\* accuracy, (\d+)% of every observable's span\? At dim "
        r"(\d+), the largest size, SLB \((\d+) realizations\) is (\d+)x cheaper "
        r"than `mcsolve` on the mixed chain and (\d+)x cheaper on the oscillator\. "
        r"The `mcsolve` side is projected, not run: the (\d+) and (\d+) "
        r"trajectories the target needs, times its measured cost per trajectory\. "
        r"Neither ratio is at matched step counts: SLB takes (\d+) fixed RK4 "
        r"substeps on the mixed chain and (\d+) on the oscillator at dim 128, while "
        r"`mcsolve` steps adaptively\. "
        r"From dim (\d+) up the gap widens at every doubling on both systems, "
        r"while System A never reaches the target at all\.", _flat(doc))
    assert m, "section 5's Result 4 summary item has changed shape"
    q_pct, q_dim, q_runs, q_b, q_c, q_nb, q_nc, q_sub_b, q_sub_c, q_from = m.groups()
    # The SLB substeps behind each ratio, from the runner that produced the
    # file: (size, substeps) at the largest size.
    import run_isocost_vs_dim as R
    for name, q_sub in (("mixed_chain", q_sub_b), ("oscillator_bath", q_sub_c)):
        build, points = R.SYSTEMS[name]
        size, substeps = points[-1]
        assert build(size)[0].shape[0] == int(q_dim) == 128
        assert int(q_sub) == substeps, f"{name}: SLB ran {substeps} substeps"
    assert int(q_pct) == round(100 * P.TARGET_REL)
    assert P.ESTIMATE_TYPE == "ensemble"

    def derived(name):
        path = DATA / f"isocost_vs_dim_{name}.json"
        if not path.exists():
            pytest.skip(f"{path.name} not committed")
        n_runs = run_counts(name)
        out = P.derive(json.loads(path.read_text(encoding="utf-8")),
                       P.TARGET_RMSE, n_runs, P.ESTIMATE_TYPE)
        return out, out["slb"][max(n_runs)], max(n_runs)

    for name, q_speed, q_ntraj in (("mixed_chain", q_b, q_nb),
                                   ("oscillator_bath", q_c, q_nc)):
        out, slb, n_runs = derived(name)
        assert int(out["dims"][-1]) == int(q_dim), f"{name} reaches {out['dims'][-1]}"
        assert n_runs == int(q_runs)
        assert slb["ok"].all() and out["mc_ok"][-1], f"{name}: target not met everywhere"
        speedup = out["mc_cost"] / slb["cost"]
        assert _near(q_speed, speedup[-1]), f"{name}: {q_speed} against {speedup[-1]:.1f}"
        assert _near(q_ntraj, out["mc_star"][-1])
        tail = speedup[out["dims"] >= int(q_from)]
        assert np.all(np.diff(tail) > 0), f"{name}: gap does not widen at every step {tail}"

    out, slb, _ = derived("spin_chain")
    assert not slb["ok"].any(), "System A is said never to reach the target"


def test_section5_summary_mesolve_memory_walls_match_the_formula(doc):
    """The summary put the chain's wall at dim 128 for both chains; on 32 GB
    the mixed chain stops at dim 32. Checked against section 5.2's own model,
    N_L * N^4 * 16 bytes, with N_L from the committed files, and against the
    large-memory runs that went one size further."""
    m = re.search(
        r"On a 32 GB machine it runs to dim (\d+) on the transverse-field chain and "
        r"to dim (\d+) on the mixed chain and the oscillator, because the next size "
        r"needs (\d+), (\d+) and (\d+) GB\. The large-memory nodes of §5\.2 each "
        r"ran one size further; the size after that needs ([\d.]+), ([\d.]+) and "
        r"([\d.]+) TB, more than any node here has\.", _flat(doc))
    assert m, "section 5's memory-wall summary item has changed shape"
    q = m.groups()
    ceilings = {"spin_chain": q[0], "mixed_chain": q[1], "oscillator_bath": q[1]}
    next_gb = dict(zip(("spin_chain", "mixed_chain", "oscillator_bath"), q[2:5]))
    after_tb = dict(zip(("spin_chain", "mixed_chain", "oscillator_bath"), q[5:8]))

    largest_node = []   # available_bytes the timing runs recorded on their skips
    for system in ceilings:
        n_l, big_node = {}, {}
        for name in (f"cost_scaling_{system}.json", f"solver_timing_{system}.json"):
            path = DATA / name
            if not path.exists():
                pytest.skip(f"{name} not committed")
            for p in json.loads(path.read_text(encoding="utf-8"))["points"]:
                n_l[p["dim"]] = p["n_l"]
                if "timings" in p:
                    entry = p["timings"].get("mesolve", {})
                    big_node[p["dim"]] = "median_s" in entry
                    if "available_bytes" in entry:
                        largest_node.append(entry["available_bytes"])
        need = {d: n_l[d] * d ** 4 * 16 for d in n_l}
        # 32 GB read as GB or as GiB gives the same ceiling, or the claim is fragile.
        for limit in (32e9, 32 * 1024 ** 3):
            ceiling = max(d for d in need if need[d] <= limit)
            assert ceiling == int(ceilings[system]), f"{system}: ceiling {ceiling}"
        ceiling = int(ceilings[system])
        assert _near(next_gb[system], need[2 * ceiling] / 1e9)
        assert big_node.get(2 * ceiling), f"{system}: mesolve did not run at dim {2 * ceiling}"
        assert _near(after_tb[system], need[4 * ceiling] / 1e12)
        after_next = need[4 * ceiling]
        assert largest_node and after_next > max(largest_node), (
            f"{system}: dim {4 * ceiling} needs {after_next / 1e12:.2f} TB; "
            f"a node recorded {max(largest_node or [0]) / 1e12:.2f} TB")


def test_extreme_dimension_operator_list_is_quoted_in_both_units(doc):
    """The Result 5 operator list is 34.2e9 bytes: 34 GB, 31.9 GiB. The
    document printed 31.9 'GB' in two places against 34 GB in section 5.2.
    plot_extreme_dimension.derive computes the GiB value its figure prints."""
    P = pytest.importorskip("plot_extreme_dimension")
    path = DATA / "extreme_dimension_mixed_chain_dim256.json"
    if not path.exists():
        pytest.skip("Result 5 data not committed")
    data = json.loads(path.read_text(encoding="utf-8"))
    size = data["operator_list_bytes"]
    assert size == data["n_l"] * data["dim"] ** 2 * 16, "not a dense complex128 list"
    gib = P.derive(data)["list_gb"]

    text = _flat(doc)
    found = (re.findall(r"operator list alone would be (\d+) GB \(([\d.]+) GiB\)", text)
             + re.findall(r"would consume \*\*(\d+) GB\*\* \(([\d.]+) GiB,", text))
    assert len(found) == 2, f"expected both copies in GB and GiB, found {found}"
    for q_gb, q_gib in found:
        assert _near(q_gb, size / 1e9) and _near(q_gib, gib)
    assert "31.9 GB and no exact solve" not in _flat(doc)


# --- section 5.1's cost-accuracy reading, and its copies in Result 3 --------
#
# Section 5.1 once scored `mcsolve` on its bias alone -- 620x and 19.3x --
# after Result 3 had retired that scoring for bias (+) s.e.m. and printed 914x
# and 33.2x for the same comparison. Nothing read section 5.1, so the two
# sections disagreed for as long as it took someone to notice. These tests
# pin both scorings to the file and to the words that say which one is meant.

def _flat_ws(doc: str) -> str:
    """The prose with every run of whitespace collapsed to one space, so a
    sentence regex does not depend on where the paragraph was hard-wrapped."""
    return re.sub(r"\s+", " ", doc)


def _latex_with_precision(mantissa: str, exponent: str) -> tuple[float, float]:
    """'6.7', '-2' from $6.7\\times10^{-2}$ -> (0.067, 0.0005): the value and
    half a unit in its last printed digit, as _decode_with_precision does for
    the Unicode form used in tables."""
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    exp = int(exponent)
    return float(mantissa) * 10 ** exp, 0.5 * 10 ** (exp - decimals) * (1 + 1e-9)


def _assert_rounds_to(measured: float, printed: str, what: str):
    """Plain decimal like '14.2', '7,118' or '0.99': half a unit in the last
    printed digit, via _rounding_tolerance."""
    value = float(printed.replace(",", ""))
    assert abs(measured - value) <= _rounding_tolerance(printed.replace(",", "")) * (1 + 1e-9), (
        f"{what}: measured {measured:.6g} does not round to the printed {printed}")


def _assert_latex_rounds_to(measured: float, mantissa: str, exponent: str, what: str):
    value, half = _latex_with_precision(mantissa, exponent)
    assert abs(measured - value) <= half, (
        f"{what}: measured {measured:.4e} does not round to the printed {value:.4e}")


LATEX = r"\$([\d.]+)\\times10\^\{(-?\d+)\}\$"


def _deviation(estimate, reference) -> float:
    """Time-averaged |estimate - reference| of ONE estimate, through the bias
    term of common.tavg_bias_sem_rmse. One sample has no s.e.m.: the NaN s.e.m.
    and RMSE terms are discarded, and numpy's ddof warning with them."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return common.tavg_bias_sem_rmse(np.atleast_2d(estimate), reference)[0]


def _oscillator64():
    """Everything section 5.1 quotes, from the one file it quotes: errors
    through plot_method_comparison.method_errors (Result 3's scoring) and
    distances from the reference through common.tavg_bias_sem_rmse."""
    import plot_method_comparison as pmc

    path = DATA / "method_comparison_oscillator_bath_dim64.json"
    if not path.exists():
        pytest.fail(f"{path.name} is quoted in BENCHMARKS.md but not committed")
    document = json.loads(path.read_text(encoding="utf-8"))
    point = document["point"]
    out = {"substeps": document["meta"]["substeps"],
           "native_wall": point["methods"]["native"]["wall_s"]}
    for obs in ("energy", "coherence"):
        rows = pmc.method_errors(point, obs)
        mc = next(r for r in rows if r[0] == "mcsolve")
        slb = {int(r[3].split("=")[1]): r for r in rows if r[0] == "slb"}
        out[obs] = {"mc": mc, "slb": slb}
    reference = pmc.mean_curve(point["reference"]["curves"]["energy"])
    out["span"] = float(reference.max() - reference.min())
    out["mc_deviation"] = _deviation(
        pmc.mean_curve(point["methods"]["mcsolve"]["curves"]["energy"]), reference)
    index = point["observables"].index("energy")
    slb16 = next(r for r in point["methods"]["slb"] if r["M"] == 16)
    samples = np.asarray(slb16["samples"], dtype=float)[:, index, :]
    out["n_runs"] = samples.shape[0]
    out["single"] = [_deviation(s, reference) for s in samples]
    out["mean_of_16"] = common.tavg_bias_sem_rmse(samples, reference)[0]
    out["slb16_run_wall"] = slb16["wall_s"] / slb16["n_runs"]
    return out


def test_section51_which_number_table_uses_result3_scoring(doc):
    """The table that says which number is which: the ensemble row must carry
    Result 3's ratios (bias (+) s.e.m. on both sides), and those must be the
    ratios Result 3's own table prints; the one-realization row carries the
    no-s.e.m. ratio section 5.1 quotes."""
    d = _oscillator64()
    row = re.search(r"^\| \*\*([\d.]+)x, ([\d.]+)x\*\* \| how many times smaller SLB's "
                    r"error is than `mcsolve`'s on the oscillator at dim 64 \(energy, "
                    r"coherence\): the 16-realization `M=16` ensemble", doc, re.M)
    assert row, "section 5.1's Result 3 row has changed shape"
    for printed, obs in zip(row.groups(), ("energy", "coherence")):
        mc, slb = d[obs]["mc"], d[obs]["slb"][16]
        assert slb[4] == 16 and mc[4] == 500
        _assert_rounds_to(mc[2] / slb[2], printed, f"section 5.1 {obs} ratio")
        in_result3 = re.search(rf"^\| `{obs}` \|[^\n]*\| \**{re.escape(printed)}x better\**",
                               doc, re.M)
        assert in_result3, f"section 5.1 quotes {printed}x for {obs}; Result 3's table does not"

    single = re.search(r"^\| \*\*(\d+)x\*\* \| the same energy comparison for \*one\* SLB "
                       r"realization against the 500-trajectory mean", doc, re.M)
    assert single, "section 5.1's one-realization row has changed shape"
    _assert_rounds_to(d["mc_deviation"] / np.mean(d["single"]), single.group(1),
                      "one-realization ratio")


def test_section51_sample_count_paragraph_matches_the_data(doc):
    """'The two stochastic methods need very different sample counts': the
    span, mcsolve's one-trajectory spread and its 500-mean's distance, their
    ratio against sqrt(500), and SLB's one-realization distance (mean, best,
    worst over the 16 run) against the 16-realization mean's."""
    d = _oscillator64()
    m = re.search(
        r"\(([\d.]+) on the oscillator at dimension 64\)\. No s\.e\.m\. is added on either "
        r"side\. Result 3's tables add it and score the whole 16-realization SLB ensemble, "
        r"which is why they print ([\d.]+)x where this subsection prints ([\d.]+)x\. .*?"
        r"one trajectory scatters by " + LATEX + r" of the span, and the mean of all 500 "
        r"sits " + LATEX + r" from the reference — a factor of (\d+), against "
        r"\$\\sqrt\{500\}=(\d+)\$\..*?One realization at \$M=16\$ sits " + LATEX +
        r" from the reference \(the average over the (\d+) realizations run; the best was "
        + LATEX + r", the worst " + LATEX + r"\)\. The mean of all 16 sits " + LATEX +
        r" from it, only ([\d.]+)x closer", _flat_ws(doc))
    assert m, "section 5.1's sample-count paragraph has changed shape"
    (q_span, q_ens, q_single, s_m, s_e, dev_m, dev_e, q_factor, q_root, one_m, one_e,
     q_runs, best_m, best_e, worst_m, worst_e, mean_m, mean_e, q_closer) = m.groups()

    span, energy = d["span"], d["energy"]
    mc = energy["mc"]
    spread = mc[5] * np.sqrt(mc[4])          # S: the s.e.m. times sqrt(ntraj)
    single = np.asarray(d["single"])
    _assert_rounds_to(span, q_span, "energy span")
    _assert_rounds_to(mc[2] / energy["slb"][16][2], q_ens, "Result 3 energy ratio")
    _assert_rounds_to(d["mc_deviation"] / single.mean(), q_single, "one-realization ratio")
    _assert_latex_rounds_to(spread / span, s_m, s_e, "one-trajectory spread")
    _assert_latex_rounds_to(d["mc_deviation"] / span, dev_m, dev_e, "500-mean distance")
    _assert_rounds_to(spread / d["mc_deviation"], q_factor, "one-to-500 factor")
    _assert_rounds_to(np.sqrt(mc[4]), q_root, "sqrt(ntraj)")
    assert int(q_runs) == d["n_runs"] == 16
    _assert_latex_rounds_to(single.mean() / span, one_m, one_e, "one realization, mean")
    _assert_latex_rounds_to(single.min() / span, best_m, best_e, "one realization, best")
    _assert_latex_rounds_to(single.max() / span, worst_m, worst_e, "one realization, worst")
    _assert_latex_rounds_to(d["mean_of_16"] / span, mean_m, mean_e, "16-realization mean")
    _assert_rounds_to(single.mean() / d["mean_of_16"], q_closer, "one vs sixteen")


def test_section51_parallel_limit_table_and_paragraph_match_the_data(doc):
    """The two-row cost table and 'So parallelism does not close the gap':
    per-sample times, both distances (span-normalized and absolute), the 6x
    and 488x, SLB's substep count, and the Result 3 scoring beside them."""
    d = _oscillator64()
    energy, span = d["energy"], d["span"]
    mc = energy["mc"]
    single = float(np.mean(d["single"]))

    table = re.search(
        r"^\| SLB, `M=16` \| \*\*1\*\* realization \| ([\d.]+) s \| ([\d.]+×10[⁻¹²³⁴⁵⁶⁷⁸⁹⁰]+) \|\n"
        r"\| `mcsolve` \| \*\*(\d+)\*\* trajectories \| ([\d,]+) s in series \(([\d.]+) s per "
        r"trajectory\) \| ([\d.]+×10[⁻¹²³⁴⁵⁶⁷⁸⁹⁰]+) \|$", doc, re.M)
    assert table, "section 5.1's cost table has changed shape"
    q_run, q_one, q_ntraj, q_total, q_per, q_mc = table.groups()
    _assert_rounds_to(d["slb16_run_wall"], q_run, "one SLB realization, wall")
    value, half = _decode_with_precision(q_one)
    assert abs(single / span - value) <= half, "one SLB realization, span-normalized"
    assert int(q_ntraj) == mc[4]
    _assert_rounds_to(mc[1], q_total, "mcsolve total wall")
    _assert_rounds_to(mc[1] / mc[4], q_per, "mcsolve per trajectory")
    value, half = _decode_with_precision(q_mc)
    assert abs(d["mc_deviation"] / span - value) <= half, "mcsolve, span-normalized"

    p = re.search(
        r"`mcsolve` finishes in ([\d.]+) s at " + LATEX + r" span-normalized energy error "
        r"\(" + LATEX + r" absolute: how far its 500-trajectory mean sits from the "
        r"reference\), while one SLB realization finishes in ([\d.]+) s at " + LATEX +
        r" \(" + LATEX + r" absolute\) — still (\d+)x faster and (\d+)x more accurate on "
        r"the energy\. The \d+x is not at matched step counts: SLB takes (\d+) fixed RK4 "
        r"substeps per output interval, while `mcsolve` steps adaptively\. Scored as in "
        r"Result 3, with each ensemble's s\.e\.m\. folded in, the 16-realization SLB "
        r"ensemble \(" + LATEX + r"\) beats the 500-trajectory `mcsolve` mean \(" + LATEX +
        r"\) by (\d+)x\.", _flat_ws(doc))
    assert p, "section 5.1's parallel-limit paragraph has changed shape"
    (q_mc_t, mcs_m, mcs_e, mca_m, mca_e, q_slb_t, ss_m, ss_e, sa_m, sa_e, q_fast,
     q_acc, q_sub, e16_m, e16_e, emc_m, emc_e, q_914) = p.groups()
    _assert_rounds_to(mc[1] / mc[4], q_mc_t, "mcsolve per trajectory")
    _assert_latex_rounds_to(d["mc_deviation"] / span, mcs_m, mcs_e, "mcsolve, span-normalized")
    _assert_latex_rounds_to(d["mc_deviation"], mca_m, mca_e, "mcsolve, absolute")
    _assert_rounds_to(d["slb16_run_wall"], q_slb_t, "one SLB realization, wall")
    _assert_latex_rounds_to(single / span, ss_m, ss_e, "SLB, span-normalized")
    _assert_latex_rounds_to(single, sa_m, sa_e, "SLB, absolute")
    _assert_rounds_to((mc[1] / mc[4]) / d["slb16_run_wall"], q_fast, "per-sample speed ratio")
    _assert_rounds_to(d["mc_deviation"] / single, q_acc, "one-realization accuracy ratio")
    assert int(q_sub) == d["substeps"], "SLB's substep count is not what the file records"
    _assert_latex_rounds_to(energy["slb"][16][2], e16_m, e16_e, "SLB ensemble, Result 3 score")
    _assert_latex_rounds_to(mc[2], emc_m, emc_e, "mcsolve, Result 3 score")
    _assert_rounds_to(mc[2] / energy["slb"][16][2], q_914, "Result 3 energy ratio")


def test_section51_thread_sentence_matches_the_files(doc):
    """Section 5.1 used to call these runs 'single core'. Every Result 3 file
    records the BLAS/OpenMP thread count its job set; the sentence quoting
    them must match all of them."""
    m = re.search(r"Each job set (\d+) BLAS threads on Systems A and C and (\d+) on System "
                  r"B \((\d+) at A's dimensions (\d+) and (\d+) and at B's (\d+)\)", _flat_ws(doc))
    assert m, "section 5.1's thread-count sentence has changed shape"
    ac, b, big, a1, a2, b1 = (int(g) for g in m.groups())
    expected = {"spin_chain": ac, "oscillator_bath": ac, "mixed_chain": b}
    exceptions = {("spin_chain", a1), ("spin_chain", a2), ("mixed_chain", b1)}
    paths = sorted(DATA.glob("method_comparison_*_dim*.json"))
    assert paths, "no Result 3 files found"
    for path in paths:
        system, dim = re.match(r"method_comparison_(\w+?)_dim(\d+)$", path.stem).groups()
        threads = json.loads(path.read_text(encoding="utf-8"))["meta"]["execution"]["threads"]
        want = big if (system, int(dim)) in exceptions else expected[system]
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert int(threads[var]) == want, (
                f"{path.name}: {var}={threads[var]}, section 5.1 says {want}")


def test_section51_exact_baseline_sentence_matches_the_data(doc):
    """'On the choice of exact baseline': the largest mesolve/native cost
    ratio across Result 3's files, the two quoted pairs of wall-clocks, the
    agreement of both solvers with the certified reference at those sizes
    (through plot_method_comparison.method_errors), and native's substeps."""
    import plot_method_comparison as pmc

    m = re.search(
        r"differ in cost by up to (\d+)x on the same problem — ([\d.]+) s against "
        r"([\d.]+) s on the TFIM chain at dimension 64, ([\d.]+) s against ([\d.]+) s on the "
        r"oscillator at dimension 32 —.*?both sit within " + LATEX + r" of the certified "
        r"reference on every observable\. The times are not at matched step counts: native "
        r"RK4 runs at SLB's own substeps \((\d+) on the chain, (\d+) on the oscillator\)",
        _flat_ws(doc))
    assert m, "section 5.1's exact-baseline paragraph has changed shape"
    (q_max, q_mes_a, q_nat_a, q_mes_c, q_nat_c, dev_m, dev_e,
     q_sub_a, q_sub_c) = m.groups()

    ratios = []
    for path in DATA.glob("method_comparison_*_dim*.json"):
        methods = json.loads(path.read_text(encoding="utf-8"))["point"]["methods"]
        if "wall_s" in methods.get("mesolve", {}) and "wall_s" in methods.get("native", {}):
            ratios.append(methods["mesolve"]["wall_s"] / methods["native"]["wall_s"])
    _assert_rounds_to(max(ratios), q_max, "largest mesolve/native ratio")

    worst = 0.0
    for name, q_mes, q_nat, q_sub in (("spin_chain_dim64", q_mes_a, q_nat_a, q_sub_a),
                                      ("oscillator_bath_dim32", q_mes_c, q_nat_c, q_sub_c)):
        document = json.loads((DATA / f"method_comparison_{name}.json")
                              .read_text(encoding="utf-8"))
        point = document["point"]
        _assert_rounds_to(point["methods"]["mesolve"]["wall_s"], q_mes, f"{name} mesolve")
        _assert_rounds_to(point["methods"]["native"]["wall_s"], q_nat, f"{name} native")
        assert int(q_sub) == document["meta"]["substeps"], f"{name}: substeps"
        for obs in point["observables"]:
            for row in pmc.method_errors(point, obs):
                if row[0] in ("native", "mesolve"):
                    worst = max(worst, row[2])
    _assert_latex_rounds_to(worst, dev_m, dev_e, "exact solvers against the reference")


def test_result3_oscillator_trajectory_projection_uses_result4_rule(doc):
    """Result 3's projected trajectory count once scaled mcsolve's bias-only
    error. It now uses Result 4's rule, N* = (S/target)^2 with S the spread
    across trajectories -- here S = s.e.m. * sqrt(ntraj) from method_errors,
    the time mean of the per-time spread, as run_isocost_vs_dim records it --
    and SLB's 16-realization errors at M = 32 and M = 16 as the targets."""
    d = _oscillator64()
    mc, slb = d["energy"]["mc"], d["energy"]["slb"]
    m = re.search(
        r"\(\$S = ([\d.]+)\$ on the oscillator's energy at dimension 64\)\. Reaching SLB's "
        r"16-realization error of " + LATEX + r" there — the \$M=32\$ setting.*?would take "
        r"about \$([\d.]+)\\times10\^\{(\d+)\}\$ trajectories against the 500 it was run "
        r"with\. Matching the \$M=16\$ error of " + LATEX + r" would take about "
        r"\$([\d.]+)\\times10\^\{(\d+)\}\$", _flat_ws(doc))
    assert m, "Result 3's trajectory-projection sentence has changed shape"
    q_s, t32_m, t32_e, n32_m, n32_e, t16_m, t16_e, n16_m, n16_e = m.groups()
    spread = mc[5] * np.sqrt(mc[4])
    assert mc[4] == 500 and slb[32][4] == 16 and slb[16][4] == 16
    _assert_rounds_to(spread, q_s, "S")
    _assert_latex_rounds_to(slb[32][2], t32_m, t32_e, "SLB M=32 error")
    _assert_latex_rounds_to(slb[16][2], t16_m, t16_e, "SLB M=16 error")
    _assert_latex_rounds_to((spread / slb[32][2]) ** 2, n32_m, n32_e, "N* for M=32")
    _assert_latex_rounds_to((spread / slb[16][2]) ** 2, n16_m, n16_e, "N* for M=16")


def test_result3_oscillator_one_realization_paragraph_matches_the_data(doc):
    """Result 3's System C paragraph on one realization: its wall-clock and
    distance, the exact solve's wall-clock and substeps, the 54x and 3.4x
    against it, mcsolve's total and the 3,100x against it, the 488x and the
    distance it divides by, and the table's 914x it is set beside."""
    d = _oscillator64()
    mc, span = d["energy"]["mc"], d["span"]
    single = float(np.mean(d["single"]))
    m = re.search(
        r"One SLB realization at \$M=16\$ costs ([\d.]+) s and sits " + LATEX + r" of the "
        r"energy's span from the reference \(averaged over time and over the (\d+) "
        r"realizations run\), against (\d+) s for the exact full-dissipator solve at the "
        r"same (\d+) substeps \(\*\*(\d+)x cheaper\*\*, or ([\d.]+)x for the full "
        r"ensemble\) and ([\d,]+) s for `mcsolve` at (\d+) trajectories \(\*\*([\d,]+)x "
        r"cheaper\*\*, and \*\*(\d+)x more accurate\*\* on the energy\)\. The \d+x sets that "
        r"one realization's distance from the reference against the distance of "
        r"`mcsolve`'s 500-trajectory mean, " + LATEX + r" of the span, with no s\.e\.m\. on "
        r"either side\. The table's (\d+)x compares", _flat_ws(doc))
    assert m, "Result 3's System C one-realization paragraph has changed shape"
    (q_run, one_m, one_e, q_runs, q_nat, q_sub, q_54, q_34, q_mc, q_ntraj, q_3100,
     q_acc, dev_m, dev_e, q_914) = m.groups()
    _assert_rounds_to(d["slb16_run_wall"], q_run, "one SLB realization, wall")
    _assert_latex_rounds_to(single / span, one_m, one_e, "one realization, span-normalized")
    assert int(q_runs) == d["n_runs"]
    _assert_rounds_to(d["native_wall"], q_nat, "native wall")
    assert int(q_sub) == d["substeps"]
    _assert_rounds_to(d["native_wall"] / d["slb16_run_wall"], q_54, "one run vs native")
    ensemble_wall = d["slb16_run_wall"] * d["n_runs"]
    _assert_rounds_to(d["native_wall"] / ensemble_wall, q_34, "ensemble vs native")
    _assert_rounds_to(mc[1], q_mc, "mcsolve total wall")
    assert int(q_ntraj) == mc[4]
    # '3,100x' is printed to two significant figures: its trailing zeros are
    # not digits, so half a unit in the last significant one is 50.
    printed = int(q_3100.replace(",", ""))
    trailing = len(str(printed)) - len(str(printed).rstrip("0"))
    assert abs(mc[1] / d["slb16_run_wall"] - printed) <= 0.5 * 10 ** trailing, (
        f"mcsolve against one run is {mc[1] / d['slb16_run_wall']:.0f}x, printed {q_3100}x")
    _assert_rounds_to(d["mc_deviation"] / single, q_acc, "one-realization accuracy ratio")
    _assert_latex_rounds_to(d["mc_deviation"] / span, dev_m, dev_e, "mcsolve 500-mean distance")
    _assert_rounds_to(mc[2] / d["energy"]["slb"][16][2], q_914, "Result 3 energy ratio")


# --- section 5.2's walls table and oscillator stiffness bullets --------------
#
# The walls table called native RK4 "O(N^5)" on every system, and the stiffness
# bullets named the wrong node for Result 2's oscillator panel. Nothing read
# this part of section 5.2, so these pin it to the files it quotes.

def _p2a_load(name: str) -> dict:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p2a_text(doc: str) -> str:
    """Section 5.2's bullets are indented, which _flat keeps; fold that too."""
    return _flat_ws(_flat(doc))


def test_section52_native_wall_cell_matches_the_timing_files(doc):
    """The native RK4 row's wall: O(N_L N^3) per step, which is O(N^5) only
    where N_L ~ N^2/2 (System B). Pins System A's N_L range, the measured
    growth per doubling on System B (one job, one substep count, so the
    wall-clocks are comparable), the 9.4 h, the operator list's size, and the
    oscillator's native divergence."""
    m = re.search(
        r"\*\*CPU Wall, \$O\(N_L N\^3\)\$ per step:\*\* that is \$O\(N\^5\)\$ on System B, "
        r"where \$N_L \\approx N\^2/2\$, but close to \$O\(N\^3\)\$ on System A, whose \$N_L\$ "
        r"only grows from (\d+) at dim (\d+) to (\d+) at dim (\d+)\. Measured on System B "
        r"\(job (\d+), (\d+) substeps on the (\d+)-point grid, one (\d+)-thread node\), "
        r"the time grows "
        r"\$([\d.]+)\\times\$ and \$([\d.]+)\\times\$ per doubling from dim (\d+) to (\d+), "
        r"not the nominal \$(\d+)\\times\$; dim (\d+) takes \*\*([\d.]+) h\*\*, and its dense "
        r"operator list alone is (\d+) GB\. The oscillator diverges at dim (\d+) at (\d+) "
        r"substeps on the (\d+)-point grid", _p2a_text(doc))
    assert m, "section 5.2's native RK4 wall cell has changed shape"
    (a_nl_lo, a_dim_lo, a_nl_hi, a_dim_hi, job, sub, b_grid, threads, g1, g2,
     d_lo, d_hi, nominal, d_wall, hours, gb, c_dim, c_sub, c_grid) = m.groups()

    # System A: N_L at the two ends of the native row's range.
    spin = {p["dim"]: p["n_l"] for p in _p2a_load("solver_timing_spin_chain.json")["points"]}
    assert spin[int(a_dim_lo)] == int(a_nl_lo), f"System A N_L at dim {a_dim_lo}"
    top = _p2a_load(f"method_comparison_spin_chain_dim{a_dim_hi}.json")["point"]
    assert top["dim"] == int(a_dim_hi) and top["n_l"] == int(a_nl_hi)
    assert top["reference"]["selfcheck"]["passed"], "System A's top native size is not certified"

    # System B: one job, one substep count, one thread count.
    b = _p2a_load("solver_timing_mixed_chain.json")
    meta = b["meta"]
    assert meta["execution"]["slurm"]["job_id"] == job
    assert meta["params"]["native_substeps"] == int(sub)
    # The 9.4 h is the 40-point grid's; Result 1's 80-point reference is
    # twice the steps (20.5 h), and quoting it without its grid once
    # under-priced the dim-512 run.
    assert meta["tlist"]["n"] == int(b_grid)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert int(meta["execution"]["threads"][var]) == int(threads), var
    pts = {p["dim"]: p for p in b["points"]}
    lo, hi = int(d_lo), int(d_hi)
    assert sorted(pts) == [lo, 2 * lo, hi] and hi == 4 * lo
    for p in pts.values():
        assert p["native_substeps"] == int(sub)
        # N_L ~ N^2/2: 2,017 / 8,193 / 32,637 against 2,048 / 8,192 / 32,768.
        assert abs(p["n_l"] / (p["dim"] ** 2 / 2) - 1) < 0.02, p["dim"]
    native = {d: pts[d]["timings"]["native"]["median_s"] for d in pts}
    assert _near(g1, native[2 * lo] / native[lo]), "first doubling"
    assert _near(g2, native[hi] / native[2 * lo]), "second doubling"
    # Nominal growth per doubling: N^3 gives 8x, N_L gives ~4x.
    per_doubling_nl = (pts[hi]["n_l"] / pts[lo]["n_l"]) ** 0.5
    assert round(8 * per_doubling_nl) == int(nominal)
    assert int(d_wall) == hi and _near(hours, native[hi] / 3600)
    # One dense complex128 copy of every operator, as native_solver.py holds.
    assert _near(gb, pts[hi]["n_l"] * hi ** 2 * 16 / 1e9)

    # System C: the native solve at dim 256 diverged at the grid's substeps.
    osc = _p2a_load("solver_timing_oscillator_bath.json")
    assert osc["meta"]["tlist"]["n"] == int(c_grid), "the oscillator timing grid"
    c = {p["dim"]: p for p in osc["points"]}
    point = c[int(c_dim)]
    assert point["native_substeps"] == int(c_sub)
    assert point["timings"]["native"].get("diverged"), "the oscillator's native solve ran"


def test_section52_result1_oscillator_grid_and_substeps_match_the_files(doc):
    """Result 1's oscillator substeps are on an 80-point grid, Result 2's on a
    40-point one, so the same count is a step about half as long in Result 1."""
    m = re.search(
        r"All of these use a (\d+)-point time grid\. Result 1, a separate sweep, uses an "
        r"(\d+)-point grid, so its steps are about half as long at the same count: it runs the "
        r"oscillator at (\d+) substeps to dim (\d+), (\d+) at dim (\d+) and (\d+) at dim "
        r"(\d+)\.", _p2a_text(doc))
    assert m, "section 5.2's Result 1 substep sentence has changed shape"
    n_coarse, n_fine, s1, d1, s2, d2, s3, d3 = (int(g) for g in m.groups())
    assert n_fine == 2 * n_coarse
    for name in ("cost_scaling_oscillator_bath.json", "osc_dim256_reach_substeps128.json"):
        assert _p2a_load(name)["meta"]["tlist"]["n"] == n_coarse, name
    dims = committed_dims("oscillator_bath")
    assert dims[-1] == d3, (
        f"Result 1's oscillator now reaches dim {dims[-1]}; the sentence stops at {d3}")
    want = {d2: s2, d3: s3}
    for dim in dims:
        meta = _p2a_load(f"accuracy_vs_M_oscillator_bath_dim{dim}.json")["meta"]
        assert meta["tlist"]["n"] == n_fine, f"dim {dim} grid"
        expected = s1 if dim <= d1 else want[dim]
        assert meta["substeps"] == expected, f"dim {dim}: {meta['substeps']} substeps"


def test_section52_oscillator_reach_point_matches_its_file(doc):
    """The dim-256 reach point (job, substeps, one M=8 solve, N_L, Davies
    time), the instability Result 2 recorded at that size, and the caveat on
    the dim-256 entry's leftover timing list."""
    text = _p2a_text(doc)
    m = re.search(
        r"job (\d+) tested the third\. Given (\d+) substeps SLB runs at dimension (\d+) -- "
        r"one \$M=(\d+)\$ solve in ([\d.]+) s, \$N_L = ([\d{},]+)\$, Davies construction "
        r"([\d.]+) s -- with no divergence\. So the `slb_unstable_at_substeps: (\d+)` "
        r"recorded against that dimension in Result 2", text)
    assert m, "section 5.2's reach bullet has changed shape"
    job, sub, dim, m_rep, t_slb, n_l, t_dav, unstable_at = m.groups()
    reach = _p2a_load("osc_dim256_reach_substeps128.json")
    assert reach["meta"]["execution"]["slurm"]["job_id"] == job
    assert reach["meta"]["substeps"] == int(sub)
    assert reach["meta"]["params"]["M_REP"] == int(m_rep)
    (point,) = reach["points"]
    assert point["dim"] == int(dim) and point["n_l"] == int(_printed(n_l))
    assert _near(t_slb, point["t_slb_fixed"]), "reach solve wall-clock"
    assert _near(t_dav, point["t_davies"]), "reach Davies construction"

    panel = _p2a_load("cost_scaling_oscillator_bath.json")
    by_dim = {p["dim"]: p for p in panel["points"]}
    top = by_dim[int(dim)]
    assert top["slb_unstable_at_substeps"] == int(unstable_at) == panel["meta"]["substeps"]

    m = re.search(
        r"The same entry also lists three `t_native_ref_repeats` timings\. They are dim "
        r"(\d+)'s, carried over by a bug in `run_cost_scaling\.py`, fixed since; no "
        r"reference ran at dim "
        r"(\d+), and its `t_native_ref` is null\.", text)
    assert m, "section 5.2's leftover-timings caveat has changed shape"
    assert int(m.group(2)) == int(dim)
    assert top["t_native_ref"] is None and top["reference_method"] is None
    assert len(top["t_native_ref_repeats"]) == 3
    assert top["t_native_ref_repeats"] == by_dim[int(m.group(1))]["t_native_ref_repeats"], (
        "the dim-256 entry no longer carries dim 128's timings; drop the caveat sentence")


def test_section52_reach_point_absence_names_the_right_node_and_ratio(doc):
    """Result 2's oscillator panel ran on landau41 (job 19599672), not
    landau42. And the 327x it would give up is the 64-substep reference over
    one 32-substep SLB solve from that same job; at 128 SLB substeps it is
    about a quarter of that."""
    text = _p2a_text(doc)
    reach = _p2a_load("osc_dim256_reach_substeps128.json")["meta"]
    panel = _p2a_load("cost_scaling_oscillator_bath.json")
    m = re.search(r"it is integrated at \$(\d+)\\times\$ that panel's substeps, and it ran on "
                  r"(\w+) while the panel ran on (\w+)\.", text)
    assert m, "section 5.2's absence bullet has changed shape"
    factor, reach_host, panel_host = m.groups()
    assert reach["execution"]["hostname"] == reach_host
    assert panel["meta"]["execution"]["hostname"] == panel_host
    assert reach["substeps"] == int(factor) * panel["meta"]["substeps"]

    m = re.search(
        r"At dim (\d+) the certified reference \(([\d,.]+) s at (\d+) substeps\) is "
        r"\$(\d+)\\times\$ one \$M=(\d+)\$ SLB solve from the same job \(([\d.]+) s at (\d+) "
        r"substeps, half the reference's, so not at matched substeps\)\. With SLB at (\d+) "
        r"substeps, twice the reference's, it would be about \$(\d+)\\times\$\.", text)
    assert m, "section 5.2's 327x sentence has changed shape"
    dim, t_ref, ref_sub, ratio, m_rep, t_slb, slb_sub, new_sub, reduced = m.groups()
    point = {p["dim"]: p for p in panel["points"]}[int(dim)]
    params = panel["meta"]["params"]
    assert point["native_ref_selfcheck"]["passed"]
    assert point["reference_method"] == f"native_rk4_substeps{ref_sub}"
    assert params["NATIVE_REF_SUBSTEPS"] == int(ref_sub)
    assert panel["meta"]["substeps"] == int(slb_sub) and params["M_REP"] == int(m_rep)
    assert 2 * int(slb_sub) == int(ref_sub), "SLB at half the reference's substeps"
    assert _near(t_ref, point["t_native_ref"]) and _near(t_slb, point["t_slb_fixed"])
    measured = point["t_native_ref"] / point["t_slb_fixed"]
    assert _near(ratio, measured), f"reference over one SLB solve is {measured:.1f}x"
    assert int(new_sub) == 2 * int(ref_sub) == reach["substeps"]
    assert _near(reduced, measured * int(slb_sub) / int(new_sub))


def test_section52_uncommitted_certification_job_is_sourced_to_its_log(doc):
    """Job 19559986 wrote no data file, so its 2.4 days must say it comes from
    the job's log. If a file carrying that job is ever committed, quote it
    instead. The check's two resolutions are half and twice the primary's, on
    the grid run_method_comparison.py integrates on; the upward check costs
    twice the primary; and Result 1's 80-point grid, where the half-substep
    check is stable, prices the certified reference at 1.5 solves."""
    text = _p2a_text(doc)
    m = re.search(
        r"A dim-256 reference at (\d+) substeps does run: job (\d+) finished one in about "
        r"([\d.]+) days\. That time is from the job's log, not a data file; the job wrote none",
        text)
    assert m, "section 5.2's certification bullet has changed shape"
    primary, job, days = int(m.group(1)), m.group(2), m.group(3)
    carriers = [p.name for p in DATA.rglob("*.json")
                if f'"{job}"' in p.read_text(encoding="utf-8", errors="ignore")]
    assert not carriers, f"job {job} is now recorded in {carriers}; cite the file"
    m = re.search(r"The cheap check, at half the substeps \((\d+)\), is itself unstable at "
                  r"this size on the (\d+)-point grid\..*?`certified_reference` still tries "
                  r"the half-substep check first, but when it diverges it now checks upward, "
                  r"at twice the substeps \((\d+)\), for about twice the primary's cost, "
                  r"instead of discarding the reference\. On this grid that puts a "
                  r"\*certified\* dim-256 oscillator reference at more than a week: "
                  r"([\d.]+) days, the downward check until it diverges, then about "
                  r"([\d.]+) days more\. Result 1's runner avoids the problem with its "
                  r"finer (\d+)-point grid, where the (\d+)-substep check is stable: it "
                  r"prices a certified dim-256 reference at about ([\d.]+) days "
                  r"\(`run_accuracy_vs_M\.py`\)\.", text)
    assert m, "section 5.2's certification check sentence has changed shape"
    half, grid, up, primary_days, check_days, fine, stable, r1_days = m.groups()
    assert int(half) == primary // 2 and int(up) == 2 * primary
    assert len(common.TLIST) == int(grid), "run_method_comparison integrates on common.TLIST"
    assert primary_days == days and _near(check_days, 2 * float(days))
    # Result 1's grid: its oscillator dim-256 point runs SLB at the stable 64
    # substeps and checks the 128-substep reference against them, 1.5 solves.
    import run_accuracy_vs_M as R1
    import run_frontier_spins as F
    assert len(common.TLIST_FINE) == int(fine)
    size, _ladder, slb_sub = R1.SYSTEMS["oscillator_bath"][1][-1]
    H = common.build_oscillator_bath(size)[0]
    assert H.shape[0] == 256 and slb_sub == int(stable) == primary // 2
    dt = float(np.min(np.diff(common.TLIST_FINE)))
    assert F.spectral_bound(H) * dt / slb_sub < F.RK4_STABILITY_LIMIT
    per_solve = float(days) * (len(common.TLIST_FINE) - 1) / (len(common.TLIST) - 1)
    assert _near(r1_days, 1.5 * per_solve), f"{1.5 * per_solve:.2f} days"


# --- section 5.2's four-solver grid: projections, step sizes, construction --
#
# The grid's oscillator row printed "64 substeps" for dim 256 where the grid's
# own exact solve diverged at 64: the 64 was the frontier's count on its finer
# 101-point grid. A substep count is only a step size once its grid is named,
# so these tests check the step each sentence prints, not the count alone.

def _p2b_timing(system: str) -> dict:
    path = DATA / f"solver_timing_{system}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p2b_frontier(system: str) -> dict:
    path = DATA / f"frontier_spins_{system}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p2b_step(tlist: dict, substeps: int) -> float:
    """The RK4 step rk4_mesolve takes: one output interval over `substeps`."""
    return (tlist["t1"] - tlist["t0"]) / (tlist["n"] - 1) / substeps


def test_section52_grid_mesolve_projections_match_the_formula_and_files(doc):
    """Every italic `mesolve` cell is N_L * N^4 * 16 bytes through
    run_solver_timing.mesolve_bytes, every skip was the --max-full-dim cap the
    prose names, and the oscillator file's older Liouvillian-only figures are
    quoted as what that file records."""
    rst = pytest.importorskip("run_solver_timing")
    text = _flat(doc)

    m = re.search(r"Each job capped `mesolve` with `--max-full-dim` \(dim (\d+) on "
                  r"System A, dim (\d+) on B and C\)", text)
    assert m, "the grid's cap sentence has changed shape"
    caps = {"spin_chain": int(m.group(1)), "mixed_chain": int(m.group(2)),
            "oscillator_bath": int(m.group(2))}
    points = {}
    for system, cap in caps.items():
        document = _p2b_timing(system)
        assert document["meta"]["max_full_dim"] == cap, f"{system}: cap"
        for p in document["points"]:
            points[(p["dim"], p["n_l"])] = (system, p)
            entry = p["timings"]["mesolve"]
            if entry.get("skipped"):
                assert entry["reason"] == f"dim {p['dim']} > --max-full-dim {cap}", (
                    f"{system} dim {p['dim']} was skipped for another reason: "
                    f"{entry['reason']}")

    units = {"TB": 1e12, "PB": 1e15}
    cells = re.findall(r"^\|[^|]*\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|[^|]*\|"
                       r"\s*\*([\d.,]+) (TB|PB)\*\s*\|", doc, re.M)
    skipped = [k for k, (_, p) in points.items()
               if p["timings"]["mesolve"].get("skipped")]
    assert len(cells) == len(skipped), (
        f"{len(cells)} italic projections against {len(skipped)} skipped runs")
    for dim, n_l, printed, unit in cells:
        key = (int(dim), int(n_l.replace(",", "")))
        system, p = points[key]
        need = rst.mesolve_bytes(*key)
        assert _near(printed, need / units[unit]), (
            f"{system} dim {dim}: printed {printed} {unit}, formula gives "
            f"{need / units[unit]:.4g}")
        recorded = p["timings"]["mesolve"].get("projected_bytes")
        if recorded is not None:
            assert recorded == need, f"{system} dim {dim}: file disagrees with formula"

    m = re.search(r"records only the size of one Liouvillian, ([\d.]+) GB at dim "
                  r"(\d+) and ([\d.]+) GB at dim (\d+); the table's ([\d.]+) TB and "
                  r"([\d.]+) TB are the formula", text)
    assert m, "the oscillator file's Liouvillian sentence has changed shape"
    oscillator = {p["dim"]: p for p in _p2b_timing("oscillator_bath")["points"]}
    for gb, dim, tb in ((m.group(1), int(m.group(2)), m.group(5)),
                        (m.group(3), int(m.group(4)), m.group(6))):
        entry = oscillator[dim]["timings"]["mesolve"]
        assert "projected_bytes" not in entry, (
            "the oscillator file now records the formula; drop the caveat")
        assert entry["projected_liouvillian_bytes"] == (dim ** 2) ** 2 * 16
        assert _near(gb, entry["projected_liouvillian_bytes"] / 1e9)
        assert _near(tb, rst.mesolve_bytes(dim, oscillator[dim]["n_l"]) / 1e12)


def test_section52_oscillator_dim256_substeps_are_quoted_with_their_grid(doc):
    """The dim-256 divergence sentence and its Result 5 copy: every substep
    count is tied to its grid and printed as a step, and the counts are the
    ones the timing, reach and frontier files ran."""
    text = _flat(doc)
    assert "SLB reaches that size at 64 substeps" not in text

    timing = _p2b_timing("oscillator_bath")
    grid = timing["meta"]["tlist"]
    row = next(p for p in timing["points"] if p["dim"] == 256)
    reach_path = DATA / "osc_dim256_reach_substeps128.json"
    if not reach_path.exists():
        pytest.skip(f"{reach_path.name} not committed")
    reach = json.loads(reach_path.read_text(encoding="utf-8"))
    frontier = _p2b_frontier("oscillator_bath")
    fgrid = frontier["meta"]["tlist"]
    fpoint = next(p for p in frontier["points"] if p["dim"] == 256)

    m = re.search(
        r"On this (\d+)-point grid SLB ran at (\d+) substeps, a step of " + LATEX
        + r", and native RK4 at (\d+), a step of " + LATEX + r"; both diverged\. "
        r"SLB does run at this size with (\d+) substeps on the same grid, a step "
        r"of " + LATEX + r" \(job (\d+)\)\. The frontier's (\d+) substeps at this "
        r"size \(Result 5\) are on its finer (\d+)-point grid, a step of "
        + LATEX + r"\.", text)
    assert m, "section 5.2's dim-256 divergence sentence has changed shape"
    (n, slb, slb_m, slb_e, native, nat_m, nat_e, ok, ok_m, ok_e, job,
     fsub, fn, f_m, f_e) = m.groups()

    assert int(n) == grid["n"] == reach["meta"]["tlist"]["n"]
    assert int(slb) == row["slb_substeps"] and int(native) == row["native_substeps"]
    for name, sub in (("slb", slb), ("native", native)):
        entry = row["timings"][name]
        assert entry.get("diverged") and f"substeps={sub}" in entry["error"], name
    assert int(ok) == reach["meta"]["substeps"]
    assert job == reach["meta"]["execution"]["slurm"]["job_id"]
    reached = next(p for p in reach["points"] if p["dim"] == 256)
    assert reached["t_slb_fixed"] is not None, "the reach run did not complete"
    assert int(fsub) == fpoint["substeps"] and int(fn) == fgrid["n"]
    _assert_latex_rounds_to(_p2b_step(grid, int(slb)), slb_m, slb_e, "SLB step")
    _assert_latex_rounds_to(_p2b_step(grid, int(native)), nat_m, nat_e, "native step")
    _assert_latex_rounds_to(_p2b_step(grid, int(ok)), ok_m, ok_e, "reach step")
    _assert_latex_rounds_to(_p2b_step(fgrid, int(fsub)), f_m, f_e, "frontier step")

    m = re.search(
        r"These runs use a (\d+)-point time grid, so each substep count below is a "
        r"step ([\d.]+) times smaller than the same count on §5\.2's (\d+)-point "
        r"grid\. The oscillator's (\d+) substeps at Fock 128 are a step of " + LATEX
        + r"; on §5\.2's grid SLB ran stably with (\d+) substeps, a step of "
        + LATEX + r", while native RK4 diverged at (\d+), a step of " + LATEX
        + r":", text)
    assert m, "Result 5's grid sentence has changed shape"
    (fn, ratio, n, fsub, f_m, f_e, ok, ok_m, ok_e, native, nat_m, nat_e) = m.groups()
    assert int(fn) == fgrid["n"] and int(n) == grid["n"]
    assert _near(ratio, _p2b_step(grid, 1) / _p2b_step(fgrid, 1))
    assert int(fsub) == fpoint["substeps"] and int(ok) == reach["meta"]["substeps"]
    assert int(native) == row["native_substeps"] and row["timings"]["native"]["diverged"]
    _assert_latex_rounds_to(_p2b_step(fgrid, int(fsub)), f_m, f_e, "frontier step")
    _assert_latex_rounds_to(_p2b_step(grid, int(ok)), ok_m, ok_e, "reach step")
    _assert_latex_rounds_to(_p2b_step(grid, int(native)), nat_m, nat_e, "native step")
    for p in (_p2b_frontier(s) for s in ("oscillator_bath", "mixed_chain", "spin_chain")):
        assert p["meta"]["tlist"]["n"] == int(fn), "a frontier file left the 101-point grid"


def test_section52_construction_residual_matches_the_frontier_fit(doc):
    """'5.0 s of propagation against the 52.6 s measured ... residual 47.6 s':
    the frontier's M=16 and M=64 propagation times at System B dim 256, fitted
    fixed-plus-linear in M, evaluated at the grid's M and rescaled by RK4 step
    count; the printed construction range and the 13% gap from the same file."""
    text = _flat(doc)
    m = re.search(
        r"reproduces its untouched \$M=32\$ point to (\d+)% — and evaluating it at "
        r"\$M=(\d+)\$, scaled from the frontier's (\d+) RK4 steps \((\d+) intervals "
        r"of (\d+) substeps\) to this grid's (\d+) \((\d+) of (\d+)\), predicts "
        r"\*\*([\d.]+) s of propagation against the ([\d.]+) s measured\*\*\. The "
        r"residual, ([\d.]+) s, is bundle construction: combining ([\d,]+) operators "
        r"into (\d+)\. The frontier timed construction directly at that dimension: "
        r"([\d.]+) to ([\d.]+) s at \$M\$ = 16, 32 and 64\. The estimate here is "
        r"(\d+)% above the top of that range", text)
    assert m, "section 5.2's construction-residual sentence has changed shape"
    (m32, m_grid, fsteps, fint, fsub, gsteps, gint, gsub, pred, meas, resid,
     n_l, m_again, lo, hi, gap) = m.groups()

    frontier = _p2b_frontier("mixed_chain")
    point = next(p for p in frontier["points"] if p["dim"] == 256)
    runs = point["m_runs"]
    timing = _p2b_timing("mixed_chain")
    row = next(p for p in timing["points"] if p["dim"] == 256)

    assert int(fint) == frontier["meta"]["tlist"]["n"] - 1
    assert int(fsub) == point["substeps"] and int(fsteps) == int(fint) * int(fsub)
    assert int(gint) == timing["meta"]["tlist"]["n"] - 1
    assert int(gsub) == row["slb_substeps"] and int(gsteps) == int(gint) * int(gsub)
    assert int(m_grid) == int(m_again) == row["m_rep"]
    assert int(n_l.replace(",", "")) == row["n_l"] == point["n_l"]

    slope, intercept = np.polyfit([16, 64], [runs["16"]["t_dyn"], runs["64"]["t_dyn"]], 1)
    assert _near(m32, 100 * abs((intercept + 32 * slope) / runs["32"]["t_dyn"] - 1))
    predicted = (intercept + int(m_grid) * slope) * int(gsteps) / int(fsteps)
    measured = row["timings"]["slb"]["median_s"]
    assert _near(pred, predicted) and _near(meas, measured)
    assert _near(resid, measured - predicted)

    prep = [runs[k]["t_bundle_prep"] for k in ("16", "32", "64")]
    assert _near(lo, min(prep)) and _near(hi, max(prep))
    assert _near(gap, 100 * ((measured - predicted) / max(prep) - 1))


def test_section52_construction_oddity_is_narrowed_by_the_frontier(doc):
    """Flat construction time on the in-memory path at System B dims 128 and
    256, near-proportional growth on the streamed path at dim 512 -- every
    printed time from frontier_spins_mixed_chain.json, and each path taken
    from the file's own `streaming` flag."""
    text = _flat(doc)
    m = re.search(
        r"no \$M\$ dependence at all — ([\d.]+), ([\d.]+) and ([\d.]+) s at "
        r"\$M = 16\$, 32 and 64\. The frontier's other System B sizes narrow it "
        r"down\. At dimension (\d+), where the operator list is also held in "
        r"memory, construction is flat too: ([\d.]+), ([\d.]+) and ([\d.]+) s\. At "
        r"dimension (\d+), where the operators are streamed instead, it grows "
        r"almost in proportion to \$M\$: ([\d,]+), ([\d,]+) and ([\d,]+) s\.", text)
    assert m, "section 5.2's construction-oddity paragraph has changed shape"
    g = m.groups()
    points = {p["dim"]: p for p in _p2b_frontier("mixed_chain")["points"]}

    for dim, printed, streamed in ((256, g[0:3], False), (int(g[3]), g[4:7], False),
                                   (int(g[7]), g[8:11], True)):
        point = points[dim]
        assert point["streaming"] is streamed, f"dim {dim}: wrong path named"
        prep = [point["m_runs"][k]["t_bundle_prep"] for k in ("16", "32", "64")]
        for value, measured in zip(printed, prep):
            assert _near(value, measured), f"dim {dim}: {value} against {measured:.4g}"
        growth = prep[2] / prep[0]              # M goes up 4x from 16 to 64
        if streamed:
            assert 3.0 < growth < 4.5, f"dim {dim}: {growth:.2f}x is not ~4x"
        else:
            assert growth < 1.5, f"dim {dim}: {growth:.2f}x is not flat"


# --- section 5.2's memory, ratio and control paragraphs ---------------------
#
# The control paragraph priced SLB at M = 91 as "2.6x slower" than the exact
# solve. That set 1,600 RK4 steps (the frontier's 101-point grid at 16
# substeps) against 624 (the 40-point grid at 16): per step the two cost the
# same. The same paragraph quoted M* = 91 "at 10 spins", where Result 4 has no
# point, and a mesolve-denominated 1,891x printed in no table. The ratio
# paragraph called a 14% gap "within 10%", printed 641 for 641.5, and called
# System B's gap unexplained after the section had explained it. The memory
# paragraph read a 1.62 TB limit as 1.55 TB. Every number is pinned below.

def _p2c_timing_points():
    """Section 5.2's grid: (system, dim) -> (point, meta), all three files."""
    out = {}
    for name in ("spin_chain", "mixed_chain", "oscillator_bath"):
        path = DATA / f"solver_timing_{name}.json"
        if not path.exists():
            pytest.skip(f"{path.name} not committed")
        document = json.loads(path.read_text(encoding="utf-8"))
        for point in document["points"]:
            out[(name, point["dim"])] = (point, document["meta"])
    return out


def _p2c_rk4_steps(tlist_meta, substeps):
    """RK4 steps one solve takes: native_solver.rk4_mesolve runs `substeps`
    steps in each of the n - 1 intervals of tlist."""
    return (tlist_meta["n"] - 1) * substeps


def test_section52_mesolve_consequence_quotes_the_recorded_memory_limit(doc):
    """3.9 TB against the 1.62 TB the timing jobs recorded (available_bytes),
    not 1.55 TB; and the 7-spin ceiling is the largest measured mesolve."""
    m = re.search(
        r"System A cannot reach dim (\d+) on any node here: it needs ([\d.]+) TB, "
        r"and the timing jobs recorded a memory limit of ([\d.]+) TB\. So its "
        r"`mesolve` ceiling is (\d+) spins, not the (\d+) a dimension-only "
        r"argument gives\.", _flat(doc))
    assert m, "section 5.2's mesolve-consequence sentence has changed shape"
    q_dim, q_need, q_have, q_ceiling, q_dim_only = m.groups()
    grid = _p2c_timing_points()

    entry = grid[("spin_chain", int(q_dim))][0]["timings"]["mesolve"]
    assert entry.get("skipped"), f"mesolve did run at dim {q_dim}"
    assert _near(q_need, entry["projected_bytes"] / 1e12)
    limits = {p["timings"]["mesolve"]["available_bytes"]
              for p, _ in grid.values() if "available_bytes" in p["timings"]["mesolve"]}
    assert len(limits) == 1, f"the timing jobs recorded different limits: {limits}"
    limit = limits.pop()
    assert _near(q_have, limit / 1e12), f"recorded limit is {limit / 1e12:.3f} TB"
    # The reference table at the top of 5.2 quotes the same node size.
    node = re.search(r"On a ([\d.]+) TB node the measured ceilings are one step "
                     r"higher", _flat(doc))
    assert node, "section 5.2's mesolve table row has changed shape"
    assert _near(node.group(1), limit / 1e12)
    assert "1.55 TB" not in doc, "1.55 TB is the recorded limit in MiB read as TB"
    assert entry["projected_bytes"] > limit

    ran = [p for (name, _), (p, _) in grid.items()
           if name == "spin_chain" and "median_s" in p["timings"]["mesolve"]]
    largest = max(ran, key=lambda p: p["dim"])
    assert largest["size"] == int(q_ceiling)
    # Dimension alone -- one Liouvillian, N^4 * 16 bytes -- would fit at q_dim.
    nxt = grid[("spin_chain", int(q_dim))][0]
    assert nxt["size"] == int(q_dim_only) and int(q_dim) ** 4 * 16 < limit


def test_section52_peak_rss_is_labelled_as_job_log_data(doc):
    """The peak-RSS column and the 519 GB OOM figure are in no data file.
    The document must say so; if a timing file ever records RSS, this fails
    so the sentence can point at it instead."""
    text = _flat(doc)
    assert re.search(
        r"No data file stores peak memory use\. The peak-RSS column, the error "
        r"column built from it, the five OOM kills above and the 519 GB below all "
        r"come from the cluster's job records, so they cannot be checked from this "
        r"repository\.", text), (
        "section 5.2 no longer says where its peak-RSS figures come from")
    assert "The last row is the same solve that was OOM-killed" not in text
    for (name, dim), (point, _) in _p2c_timing_points().items():
        assert "rss" not in json.dumps(point).lower(), (
            f"{name} dim {dim} now records RSS; cite the file, not the job logs")


def test_section52_operator_count_ratios_match_the_grid(doc):
    """2 N_L / M against native / SLB from the grid: 22.8 vs 20.6 (A, 11%
    high), 421.5 vs 370 (C, 14% high), 8,159 vs 642 (B)."""
    text = _flat(doc)
    m = re.search(
        r"On System A at dim (\d+) that is \$([\d.]+)\$ against \$([\d.]+)\$ "
        r"measured; on System C at dim (\d+), \$([\d.]+)\$ against \$([\d.]+)\$\. "
        r"The prediction runs high on both, by (\d+)% on A and (\d+)% on C\.", text)
    assert m, "section 5.2's 2N_L/M sentence has changed shape"
    b = re.search(
        r"\*\*System B does not follow it\*\*: at dim (\d+) the prediction is "
        r"\$([\d{},]+)\$ and the measurement is \$([\d{},]+)\$\.", text)
    assert b, "section 5.2's System B ratio sentence has changed shape"
    grid = _p2c_timing_points()

    def ratios(name, dim):
        point = grid[(name, int(dim))][0]
        assert point["native_substeps"] == 2 * point["slb_substeps"], (
            f"{name}: the 2N_L/M model assumes SLB at half the substeps")
        t = point["timings"]
        return (2 * point["n_l"] / point["m_rep"],
                t["native"]["median_s"] / t["slb"]["median_s"])

    q_a_dim, q_a_pred, q_a_meas, q_c_dim, q_c_pred, q_c_meas, q_a_pct, q_c_pct = m.groups()
    for name, dim, q_pred, q_meas, q_pct in (
            ("spin_chain", q_a_dim, q_a_pred, q_a_meas, q_a_pct),
            ("oscillator_bath", q_c_dim, q_c_pred, q_c_meas, q_c_pct)):
        pred, meas = ratios(name, dim)
        assert _near(q_pred, pred), f"{name}: 2N_L/M is {pred}"
        assert _near(q_meas, meas), f"{name}: native/SLB is {meas:.2f}"
        assert _near(q_pct, 100 * (pred / meas - 1)), (
            f"{name}: prediction is {100 * (pred / meas - 1):.1f}% high")

    q_b_dim, q_b_pred, q_b_meas = b.groups()
    pred, meas = ratios("mixed_chain", q_b_dim)
    assert _near(q_b_pred, pred) and _near(q_b_meas, meas), (
        f"System B: 2N_L/M {pred:.2f}, native/SLB {meas:.2f}")
    c = re.search(
        r"Most of that gap is the construction cost estimated above: about nine "
        r"tenths of SLB's ([\d.]+) s there is spent combining ([\d,]+) operators "
        r"into (\d+) bundles, a cost the operator count leaves out\.", text)
    assert c, "section 5.2's System B construction sentence has changed shape"
    point_b = grid[("mixed_chain", int(q_b_dim))][0]
    assert _near(c.group(1), point_b["timings"]["slb"]["median_s"])
    assert int(c.group(2).replace(",", "")) == point_b["n_l"]
    assert int(c.group(3)) == point_b["m_rep"]
    assert "Both within 10%" not in text
    assert "is not explained here" not in text


def test_section52_control_prices_slb_per_rk4_step(doc):
    """Result 4 never reaches the target on System A (M = N_L = 43 at 7 spins,
    73 at 9, misses 1.1x to 4x, through plot_isocost_vs_dim.derive), has no
    10-spin point, and at M = N_L SLB and the exact solve cost the same per
    RK4 step: 8.6 s each, from the grid and the frontier."""
    P = pytest.importorskip("plot_isocost_vs_dim")
    from isocost_config import run_counts
    text = _flat(doc)
    m = re.search(
        r"On System A, Result 4 never reaches the (\d+)% target\. Even at "
        r"\$M = N_L\$, the largest bundle count it tries \((\d+) at (\d+) spins, "
        r"(\d+) at (\d+) spins, its largest size\), the error misses by "
        r"\$([\d.]+)\\times\$ and \$([\d.]+)\\times\$ at those two sizes, and by "
        r"\$([\d.]+)\\times\$ to \$([\d.]+)\\times\$ across all eight \((\d+) "
        r"realizations per size\)\. At (\d+) spins, where Result 4 has no point, that bundle count "
        r"would be \$N_L = (\d+)\$\.", text)
    assert m, "section 5.2's control paragraph has changed shape"
    (q_pct, q_m7, q_s7, q_m9, q_s9, q_miss7, q_miss9, q_lo, q_hi, q_runs, q_s10,
     q_nl10) = m.groups()

    path = DATA / "isocost_vs_dim_spin_chain.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    document = json.loads(path.read_text(encoding="utf-8"))
    n_runs = run_counts("spin_chain")
    out = P.derive(document, P.TARGET_RMSE, n_runs, P.ESTIMATE_TYPE)
    slb = out["slb"][max(n_runs)]
    assert int(q_runs) == max(n_runs), f"Result 4 averages {max(n_runs)} realizations"
    assert int(q_pct) == round(100 * P.TARGET_REL)
    assert not slb["ok"].any(), "System A now reaches the target somewhere"
    assert list(slb["mstar"]) == list(out["n_ls"]), "M* is no longer N_L everywhere"
    mstar = dict(zip(out["dims"].astype(int), slb["mstar"]))
    assert mstar[2 ** int(q_s7)] == int(q_m7)
    assert int(out["dims"][-1]) == 2 ** int(q_s9) and mstar[2 ** int(q_s9)] == int(q_m9)
    assert 2 ** int(q_s10) not in mstar, "Result 4 now has a 10-spin point"
    misses = []
    for point, bias_sq, noise_sq, binding in zip(
            document["points"], slb["bias_sq"], slb["noise_sq"], slb["binding"]):
        labels = list(P._obs_axis(point)[1])
        target = P.observable_targets(point)[labels.index(binding)]
        misses.append(float(np.sqrt(bias_sq + noise_sq) / target))
    assert _near(q_lo, min(misses)) and _near(q_hi, max(misses)), misses
    assert len(misses) == 8, "the sentence says all eight sizes"
    by_dim = dict(zip(out["dims"].astype(int), misses))
    assert _near(q_miss7, by_dim[2 ** int(q_s7)]) and _near(q_miss9, by_dim[2 ** int(q_s9)])
    per = re.search(r"That is per realization\. Result 4's errors are means of (\d+), so "
                    r"at the accuracy it measured SLB costs about (\d+) times the exact "
                    r"solve per step, and still misses the target\.", text)
    assert per and int(per.group(1)) == int(per.group(2)) == max(n_runs)

    grid = _p2c_timing_points()
    point10, meta = grid[("spin_chain", 2 ** int(q_s10))]
    assert point10["size"] == int(q_s10) and point10["n_l"] == int(q_nl10)

    s = re.search(
        r"Measured at dim (\d+) on (\d+)-thread nodes, the exact solve takes "
        r"([\d.]+) s per step \(([\d,.]+) s for ([\d,]+) steps, in the grid above\)\. "
        r"SLB takes ([\d.]+) s per step at \$M=(\d+)\$ and ([\d.]+) s at \$M=(\d+)\$ "
        r"\(the frontier's ([\d,]+)-step solves, job (\d+), on another node\)\. "
        r"Scaled in proportion to \$M=(\d+)\$, that is ([\d.]+) s, the exact "
        r"solve's rate, before SLB pays for bundle construction\. The two jobs ran "
        r"different step counts, so only the per-step costs compare\.", text)
    assert s, "section 5.2's per-step comparison has changed shape"
    (q_dim, q_threads, q_exact_rate, q_exact_wall, q_exact_steps, q_rate_a, q_ma,
     q_rate_b, q_mb, q_front_steps, q_job, q_m91, q_scaled) = s.groups()
    assert 2 ** int(q_s10) == int(q_dim)
    exact_steps = _p2c_rk4_steps(meta["tlist"], point10["native_substeps"])
    exact_wall = point10["timings"]["native"]["median_s"]
    assert int(q_exact_steps.replace(",", "")) == exact_steps
    assert _near(q_exact_wall, exact_wall)
    assert _near(q_exact_rate, exact_wall / exact_steps)

    fpath = DATA / "frontier_spins_spin_chain.json"
    if not fpath.exists():
        pytest.skip(f"{fpath.name} not committed")
    frontier = json.loads(fpath.read_text(encoding="utf-8"))
    fpoint = next(p for p in frontier["points"] if p["dim"] == int(q_dim))
    front_steps = _p2c_rk4_steps(frontier["meta"]["tlist"], fpoint["substeps"])
    assert int(q_front_steps.replace(",", "")) == front_steps
    rate = {int(k): v["t_dyn"] / front_steps for k, v in fpoint["m_runs"].items()}
    assert _near(q_rate_a, rate[int(q_ma)]) and _near(q_rate_b, rate[int(q_mb)])
    assert int(q_m91) == fpoint["n_l"] == point10["n_l"]
    scaled = rate[int(q_mb)] * fpoint["n_l"] / int(q_mb)
    assert _near(q_scaled, scaled), f"M=91 scaled rate is {scaled:.3f} s"
    assert _near(q_scaled, exact_wall / exact_steps), "the two rates no longer agree"

    fe, ge = frontier["meta"]["execution"], meta["execution"]
    assert fe["slurm"]["job_id"] == q_job
    assert fe["threads"]["OMP_NUM_THREADS"] == ge["threads"]["OMP_NUM_THREADS"] == q_threads
    assert fe["hostname"] != ge["hostname"], "the paragraph says another node"

    assert not re.search(r"2\.6(?:\\times\$|×) slower", doc), (
        "the 2.6x penalty compared 1,600 steps against 624; it must not return")
    assert "1{,}891" not in doc and "1,891" not in doc


def test_section52_grid_ratio_names_its_denominator_and_substeps(doc):
    """The grid's 20.6x at dim 1024 is native RK4 at 8 substeps over one SLB
    solve at M = 8 and 4 substeps: printed with both, as a cost."""
    m = re.search(
        r"The grid's own ratio at dim (\d+), native RK4 over SLB at \$M=(\d+)\$, is "
        r"\$([\d.]+)\\times\$ \(([\d,.]+) s against ([\d.]+) s\)\. SLB there takes "
        r"(\d+) substeps to the exact solve's (\d+), so half the steps, and (\d+) "
        r"bundles, not the (\d+) above\.", _flat(doc))
    assert m, "section 5.2's grid-ratio sentence has changed shape"
    q_dim, q_m, q_ratio, q_nat, q_slb, q_sub_slb, q_sub_nat, q_m2, q_nl = m.groups()
    point = _p2c_timing_points()[("spin_chain", int(q_dim))][0]
    t = point["timings"]
    assert int(q_m) == int(q_m2) == point["m_rep"]
    assert _near(q_nat, t["native"]["median_s"]) and _near(q_slb, t["slb"]["median_s"])
    assert _near(q_ratio, t["native"]["median_s"] / t["slb"]["median_s"])
    assert int(q_sub_slb) == point["slb_substeps"]
    assert int(q_sub_nat) == point["native_substeps"] == 2 * point["slb_substeps"]
    assert int(q_nl) == point["n_l"]


def test_result3_control_slb_sentence_matches_section52(doc):
    """Result 3's copy of the control's SLB price: same per RK4 step as the
    exact solve at M = N_L, not 2.6x slower."""
    m = re.search(
        r"SLB fares no better: at \$M = N_L = (\d+)\$ one realization costs the same "
        r"per RK4 step as the exact solve \(§5\.2\), and Result 4's errors average "
        r"(\d+) realizations, so at its accuracy SLB costs about (\d+) times as "
        r"much\.", _flat(doc))
    assert m, "Result 3's control SLB sentence has changed shape"
    from isocost_config import run_counts
    assert int(m.group(2)) == int(m.group(3)) == max(run_counts("spin_chain"))
    assert int(m.group(1)) == _p2c_timing_points()[("spin_chain", 1024)][0]["n_l"]


def test_section52_mcsolve_probe_projections_match_the_500_trajectory_runs(doc):
    """8-trajectory probe x 500 against the two 500-trajectory runs: 2.9 h vs
    2.6 h (A, dim 1024) and 4.2 vs 3.8 days (B, dim 256), 11% high on both."""
    m = re.search(
        r"At the (\d+) trajectories Result 3 runs, the largest cells project to "
        r"([\d.]+) h \(System A, dim (\d+)\), ([\d.]+) days \(System B, dim (\d+)\) "
        r"and \*\*(\d+) days\*\* \(System C, dim (\d+)\).*?The first two have since "
        r"been run at (\d+): \*\*([\d.]+) h\*\* \(job (\d+)\) and \*\*([\d.]+) days\*\* "
        r"\(job (\d+)\), so the probe projected (\d+)% high on both\.", _flat(doc))
    assert m, "section 5.2's mcsolve projection paragraph has changed shape"
    (q_n, q_a_h, q_a_dim, q_b_d, q_b_dim, q_c_d, q_c_dim, q_n2, q_a_run, q_a_job,
     q_b_run, q_b_job, q_pct) = m.groups()
    assert int(q_n) == int(q_n2)
    grid = _p2c_timing_points()
    per_traj = {name: grid[(name, int(dim))][0]["timings"]["mcsolve"]["per_trajectory_s"]
                for name, dim in (("spin_chain", q_a_dim), ("mixed_chain", q_b_dim),
                                  ("oscillator_bath", q_c_dim))}
    projected = {name: int(q_n) * t for name, t in per_traj.items()}
    assert _near(q_a_h, projected["spin_chain"] / 3600)
    assert _near(q_b_d, projected["mixed_chain"] / 86400)
    assert _near(q_c_d, projected["oscillator_bath"] / 86400)
    for name, dim, q_run, q_job, unit in (("spin_chain", q_a_dim, q_a_run, q_a_job, 3600),
                                          ("mixed_chain", q_b_dim, q_b_run, q_b_job, 86400)):
        path = DATA / f"method_comparison_{name}_dim{dim}.json"
        if not path.exists():
            pytest.skip(f"{path.name} not committed")
        document = json.loads(path.read_text(encoding="utf-8"))
        mc = document["point"]["methods"]["mcsolve"]
        assert mc["ntraj"] == int(q_n)
        assert document["meta"]["execution"]["slurm"]["job_id"] == q_job
        assert _near(q_run, mc["wall_s"] / unit)
        assert _near(q_pct, 100 * (projected[name] / mc["wall_s"] - 1)), (
            f"{name}: probe projected {100 * (projected[name] / mc['wall_s'] - 1):.1f}% high")


# --- section 5.3: the provenance intro and table -------------------------
#
# The table's dates were written from memory: Result 4 "ended" a day before its
# mixed-chain file was written, Result 5's frontier "started" a day after its
# oscillator sweep did, and the certified references were dated to August
# although every file says Jul 31. Job 19603810 sat among Result 2's figure
# jobs although its file is named so the plotter cannot load it, and the intro
# said "every file" records the 0.6.4 tolerance while three July files in
# data/ record none. These pin all of it to the files' own metadata.

_P3A_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
               "Oct", "Nov", "Dec")


def _p3a_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p3a_stamp(document: dict):
    import datetime
    return datetime.datetime.fromisoformat(document["meta"]["timestamp"])


def _p3a_walls(obj) -> float:
    """Seconds of solving a file records: every number under a key starting
    't_', every entry of a 'samples_s' list, every entry of a 'cost' list."""
    total = 0.0
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "meta":
                continue
            if key.startswith("t_") and isinstance(value, (int, float)):
                total += float(value or 0.0)
            elif key in ("samples_s", "cost") and isinstance(value, list):
                total += sum(float(v) for v in value if v)
            else:
                total += _p3a_walls(value)
    elif isinstance(obj, list):
        total += sum(_p3a_walls(v) for v in obj)
    return total


def _p3a_day(moment) -> str:
    return f"{_P3A_MONTHS[moment.month - 1]} {moment.day}"


def _p3a_span(start, end) -> str:
    """'Jul 31', 'Sep 16 – 18' or 'Aug 7 – Sep 18', as the table prints them."""
    start, end = start.date(), end.date()
    if start == end:
        return _p3a_day(start)
    if (start.year, start.month) == (end.year, end.month):
        return f"{_p3a_day(start)} – {end.day}"
    return f"{_p3a_day(start)} – {_p3a_day(end)}"


def _p3a_end_stamped(paths):
    """Files written when their run finished: the stamps bound the run."""
    stamps = [_p3a_stamp(_p3a_json(p)) for p in paths]
    return min(stamps), max(stamps)


def _p3a_start_stamped(paths):
    """Files stamped when their run started: the end adds the recorded solves."""
    import datetime
    starts, ends = [], []
    for p in paths:
        document = _p3a_json(p)
        starts.append(_p3a_stamp(document))
        ends.append(starts[-1] + datetime.timedelta(seconds=_p3a_walls(document)))
    return min(starts), max(ends)


def _p3a_table(doc: str) -> dict:
    """Section 5.3's provenance rows: bold label -> (jobs cell, dates cell)."""
    start = doc.index("| Result | Slurm jobs | dates |")
    block = doc[start:doc.index("\n\n", start)]
    rows = re.findall(r"^\|\s*\*\*([^*]+)\*\*[^|]*\|([^|]*)\|([^|]*)\|\s*$",
                      block, re.M)
    assert rows, "section 5.3's provenance table has changed shape"
    return {label.strip(): (jobs.strip(), dates.strip()) for label, jobs, dates in rows}


def _p3a_jobs(paths) -> set[str]:
    return {str(_p3a_json(p)["meta"]["execution"]["slurm"]["job_id"]) for p in paths}


def test_section53_intro_version_and_tolerance_match_the_files(doc):
    """'every file those Results read records degeneracy_tol = 1e-10' under
    0.6.4: every file smoke_test attributes to Results 1-5 carries both."""
    m = re.search(r"\*\*Results 1 through 5 run on data regenerated under ([\d.]+)\*\*, "
                  r"so every operator count matches the shipped code, and every file "
                  r"those Results read records `degeneracy_tol = ([\de.+-]+)`, the "
                  r"shipped default\.", _flat(doc))
    assert m, "section 5.3's opening sentence has changed shape"
    version, tol = m.group(1), float(m.group(2))
    assert common.QUTIP_BUNDLING_VERSION == version
    assert common.DAVIES_DEGENERACY_TOL == tol
    import smoke_test
    checked = 0
    for result in "12345":
        for pattern in smoke_test.RESULT_DATA_GLOBS[result]:
            for path in sorted(DATA.glob(pattern)):
                meta = _p3a_json(path)["meta"]
                assert meta.get("qutip_bundling") == version, path.name
                assert meta.get("davies", {}).get("degeneracy_tol") == tol, path.name
                checked += 1
    assert checked, "no Result 1-5 files found"


def test_section53_names_the_stale_frontier_files(doc):
    """The three July oscillator frontier files in data/: no tolerance,
    version or job; operator counts that the shipped code no longer builds;
    read by no Result, only by plot_frontier.py and the CSV export."""
    import fnmatch
    import smoke_test
    m = re.search(
        r"Three older files still sit in `data/` and must not be quoted: "
        r"`frontier_oscillator_bath_dim(\d+)\.json`, `_dim(\d+)\.json` and "
        r"`_dim(\d+)\.json`, written on (\w+ \d+) before ([\d.]+)\. They record no "
        r"tolerance, package version or job, and at dims (\d+) and (\d+) they hold "
        r"([\d,]+) and ([\d,]+) operators where the shipped code builds ([\d,]+) "
        r"and ([\d,]+)\. No Result reads them; only the superseded "
        r"`plot_frontier\.py` and the CSV export do\.", _flat(doc))
    assert m, "section 5.3's stale-file sentence has changed shape"
    (d1, d2, d3, day, version, da, db, old_a, old_b, new_a, new_b) = m.groups()
    named = {int(d1), int(d2), int(d3)}
    on_disk = {int(re.search(r"_dim(\d+)\.json$", p.name).group(1))
               for p in DATA.glob("frontier_oscillator_bath_dim*.json")}
    assert named == on_disk, f"data/ holds frontier_oscillator_bath dims {sorted(on_disk)}"
    assert version == common.QUTIP_BUNDLING_VERSION
    for dim in named:
        path = DATA / f"frontier_oscillator_bath_dim{dim}.json"
        document = _p3a_json(path)
        meta = document["meta"]
        for key in ("qutip_bundling", "davies", "execution"):
            assert key not in meta, f"{path.name} records {key}"
        assert _p3a_day(_p3a_stamp(document)) == day, path.name
        for result, patterns in smoke_test.RESULT_DATA_GLOBS.items():
            assert not any(fnmatch.fnmatch(path.name, p) for p in patterns), (
                f"{path.name} is attributed to Result {result}")
    for dim, old, new in ((int(da), old_a, new_a), (int(db), old_b, new_b)):
        document = _p3a_json(DATA / f"frontier_oscillator_bath_dim{dim}.json")
        assert document["n_l"] == int(_printed(old))
        H, X, _ = common.build_oscillator_bath(dim // 2)
        shipped = len(common.build_davies_operators(H, X))
        assert shipped == int(_printed(new))
        assert shipped != document["n_l"]
    assert 'glob(f"frontier_{nm}_dim*.json")' in (BENCHMARKS / "plot_frontier.py").read_text(
        encoding="utf-8")
    assert '"frontier_": export_frontier' in (BENCHMARKS / "export_csv.py").read_text(
        encoding="utf-8")
    import datetime
    released = re.search(rf"^## {re.escape(version)} — (\d{{4}}-\d{{2}}-\d{{2}})$",
                         (BENCHMARKS.parent / "CHANGELOG.md").read_text(encoding="utf-8"),
                         re.M)
    assert released, f"CHANGELOG.md has no dated {version} entry"
    release_day = datetime.date.fromisoformat(released.group(1))
    for dim in named:
        stamp = _p3a_stamp(_p3a_json(DATA / f"frontier_oscillator_bath_dim{dim}.json"))
        assert stamp.date() < release_day, f"dim {dim} was written after {version}"
    manifest = (DATA / "README.md").read_text(encoding="utf-8")
    assert "`frontier_<system>_dim<D>.json`" not in manifest, (
        "data/README.md still lists the stale frontier files as canonical")
    assert "| 3. Method comparison | `method_comparison_<system>_dim<D>.json` |" in manifest
    copy = re.search(r"written on (\w+ \d+), before ([\d.]+)\. .*?at dims (\d+) and (\d+) "
                     r"they hold ([\d,]+) and ([\d,]+) operators where the shipped code "
                     r"builds ([\d,]+) and ([\d,]+)\.", _flat(manifest))
    assert copy, "data/README.md's stale-file paragraph has changed shape"
    assert copy.groups() == (day, version, da, db, old_a, old_b, new_a, new_b)


def test_section53_section6_sentence_matches_the_progress_files(doc):
    """Section 6's spin-chain panels were re-run under 0.6.4 at dims 128-512 by
    one job; its oscillator panels carry no metadata (pre-0.6.4)."""
    m = re.search(r"Section 6's spin-chain jackknife check was re-run under ([\d.]+) "
                  r"at dims (\d+)–(\d+) \(job (\d+)\)\. Its oscillator panels and its "
                  r"seed and substep figures are preserved from the original "
                  r"pre-[\d.]+ runs", _flat(doc))
    assert m, "section 5.3's section-6 sentence has changed shape"
    version, lo, hi, job = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    import plot_jackknife_rate_strip as strip
    spin = [json.loads((BENCHMARKS / f).read_text(encoding="utf-8"))
            for f, _ in strip.SYSTEMS["spin_chain"]["panels"]]
    assert (min(d["dim"] for d in spin), max(d["dim"] for d in spin)) == (lo, hi)
    for d in spin:
        assert d["meta"]["qutip_bundling"] == version
        assert str(d["meta"]["execution"]["slurm"]["job_id"]) == job
    for f, _ in strip.SYSTEMS["oscillator_bath"]["panels"]:
        assert "meta" not in json.loads((BENCHMARKS / f).read_text(encoding="utf-8")), f


def test_section53_provenance_dates_match_the_file_timestamps(doc):
    """Every date cell of the provenance table, recomputed from the files: the
    stamp range for runners that write at the end, and stamp plus recorded
    solve time for the three that stamp before their solves."""
    m = re.search(r"Dates are the UTC timestamps the files carry\. Most runners stamp a "
                  r"file when it is finished, so a job can have started the day before "
                  r"its row's first date\. Result 5's frontier sweeps, the §5\.2 grid and "
                  r"§6's check stamp theirs before the solves they time, so their end "
                  r"dates add the solve times the files record\.", _flat(doc))
    assert m, "section 5.3's date convention sentence has changed shape"
    table = _p3a_table(doc)
    g = lambda pattern, where=DATA: sorted(where.glob(pattern))
    figures = [DATA / f"cost_scaling_{name}.json"
               for name in ("spin_chain", "mixed_chain", "oscillator_bath")]
    side = [p for p in g("cost_scaling_*.json") if p not in figures]
    spin6 = [p for p in g("convergence_progress_*.json", BENCHMARKS)
             if "meta" in json.loads(p.read_text(encoding="utf-8"))]
    expected = {
        "1": _p3a_span(*_p3a_end_stamped(g("accuracy_vs_M_*.json"))),
        "2": (_p3a_span(*_p3a_end_stamped(figures)) + "; side run "
              + _p3a_span(*_p3a_end_stamped(side))),
        "3": _p3a_span(*_p3a_end_stamped(g("method_comparison_*.json"))),
        "4": _p3a_span(*_p3a_end_stamped(g("isocost_vs_dim_*.json"))),
        "5": (_p3a_span(*_p3a_end_stamped(g("extreme_dimension_*.json"))) + ", "
              + _p3a_span(*_p3a_start_stamped(g("frontier_spins_*.json")))),
        "Certified references": _p3a_span(*_p3a_end_stamped(
            g("high_dim_reference_spin_chain_dim*.json"))),
        "§5.2": _p3a_span(*_p3a_start_stamped(g("solver_timing_*.json"))),
        "6": _p3a_span(*_p3a_start_stamped(spin6)),
    }
    for label, want in expected.items():
        assert label in table, f"provenance table has no '{label}' row"
        assert table[label][1] == want, (
            f"row '{label}' prints '{table[label][1]}', the files give '{want}'")


def test_section53_result2_row_separates_the_unplotted_side_run(doc):
    """Result 2's figures read cost_scaling_<system>.json only (4 threads);
    the 32-thread dim-1024 re-time is listed apart as not plotted."""
    jobs = _p3a_table(doc)["2"][0]
    m = re.fullmatch(r"([\d, ]+); (\d{8}) is a (\d+)-thread side run, not plotted", jobs)
    assert m, f"Result 2's jobs cell has changed shape: {jobs!r}"
    figure_jobs = set(re.findall(r"\d{8}", m.group(1)))
    names = ("spin_chain", "mixed_chain", "oscillator_bath")
    figures = [DATA / f"cost_scaling_{name}.json" for name in names]
    assert figure_jobs == _p3a_jobs(figures)
    assert 'load_data(f"cost_scaling_{name}.json")' in (
        BENCHMARKS / "plot_cost_scaling.py").read_text(encoding="utf-8")
    side = [p for p in sorted(DATA.glob("cost_scaling_*.json")) if p not in figures]
    assert _p3a_jobs(side) == {m.group(2)}
    for p in side:
        threads = _p3a_json(p)["meta"]["execution"]["threads"]["OMP_NUM_THREADS"]
        assert threads == m.group(3), p.name
    for p in figures:
        assert _p3a_json(p)["meta"]["execution"]["threads"]["OMP_NUM_THREADS"] != m.group(3)


def test_section53_certified_reference_row_matches_its_files(doc):
    """The certified-reference row: its job, its dims, and the Result 3 files
    that reuse its archives."""
    rows = _p3a_table(doc)
    assert "Certified references" in rows
    jobs, _ = rows["Certified references"]
    m = re.search(r"\|\s*\*\*Certified references\*\*, System A dims (\d+)–(\d+) "
                  r"\(`high_dim_reference_spin_chain_dim\*\.json`; Result 3 reuses "
                  r"(\d+)–(\d+)\)\s*\|", doc)
    assert m, "the certified-reference row label has changed shape"
    lo, hi, rlo, rhi = map(int, m.groups())
    paths = sorted(DATA.glob("high_dim_reference_spin_chain_dim*.json"))
    dims = [_p3a_json(p)["meta"]["params"]["dimension"] for p in paths]
    assert (min(dims), max(dims)) == (lo, hi)
    assert _p3a_jobs(paths) == set(re.findall(r"\d{8}", jobs))
    reused = sorted(
        _p3a_json(p)["point"]["dim"]
        for p in DATA.glob("method_comparison_spin_chain_dim*.json")
        if _p3a_json(p)["point"]["reference"].get("reused_from_archive"))
    assert (min(reused), max(reused)) == (rlo, rhi)


# --- Part 3, unit U2: section 5.3's comparability, thread and record claims --
#
# Section 5.3 said every Result 3 figure was one allocation (two join two
# jobs), that the 2.3x thread speed-up had "every other setting identical"
# (the node differed), and that every file records its thread settings (three
# do not). These tests pin the corrected sentences to meta.execution.

def _p3b_load(name: str) -> dict:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p3b_job(document: dict) -> str:
    return document["meta"]["execution"]["slurm"]["job_id"]


def _p3b_host(document: dict) -> str:
    return document["meta"]["execution"]["hostname"]


def _p3b_threads(document: dict) -> int:
    return int(document["meta"]["execution"]["threads"]["OMP_NUM_THREADS"])


def _p3b_point(document: dict, dim: int) -> dict:
    return next(p for p in document["points"] if p["dim"] == dim)


def _p3b_ratio_near(printed: str, measured: float) -> bool:
    """A printed '2.3' (a ratio, 'x' stripped) against the measured ratio."""
    return abs(float(printed) - measured) <= _rounding_tolerance(printed) * (1 + 1e-9)


def test_p3b_result4_is_one_allocation_on_one_grid(doc):
    """Result 4's panels share a job, a thread count and a grid; the chains
    hold 4 substeps and the oscillator steps up as section 5.3 says."""
    text = _flat(doc)
    m = re.search(r"all three of its panels come from job (\d{8}), at (\d+) threads, "
                  r"on the (\d+)-point grid\. So its seconds compare across panels",
                  text)
    assert m, "section 5.3's Result 4 allocation sentence has changed shape"
    job, threads, n = m.groups()
    s = re.search(r"Both chains run at (\d+) substeps at every dimension\. The "
                  r"oscillator runs at (\d+) up to dim (\d+), then (\d+) at dim "
                  r"(\d+) and (\d+) at dim (\d+), as Result 4 notes\.", text)
    assert s, "section 5.3's Result 4 substep sentence has changed shape"
    chain_sub, low_sub, low_top, mid_sub, mid_dim, top_sub, top_dim = map(int, s.groups())
    expected_osc = {mid_dim: mid_sub, top_dim: top_sub}
    for system in ("spin_chain", "mixed_chain", "oscillator_bath"):
        document = _p3b_load(f"isocost_vs_dim_{system}.json")
        assert _p3b_job(document) == job, system
        assert _p3b_threads(document) == int(threads), system
        assert document["meta"]["tlist"]["n"] == int(n), system
        for p in document["points"]:
            if system != "oscillator_bath":
                want = chain_sub
            else:
                want = low_sub if p["dim"] <= low_top else expected_osc[p["dim"]]
            assert p["substeps"] == want, f"{system} dim {p['dim']}"


def test_p3b_result3_cross_job_ratio_names_its_jobs(doc):
    """Result 3's System A text compares across jobs three times: the 3.5x at
    dim 1024 (mcsolve against section 5.2's exact solve), and at dim 2048 the
    exact solve's and mcsolve's growth. Section 5.3 names every job and node."""
    m = re.search(
        r"The ([\d.]+)x at dim (\d+) sets `mcsolve` \((\d+) trajectories, job "
        r"(\d{8}) on (\w+)\) against §5\.2's (\d+)-substep exact solve \(job "
        r"(\d{8}) on (\w+)\)\. At dim (\d+) the exact solve's rise from ([\d,]+) s "
        r"to ([\d,]+) s \(([\d.]+)x\) sets that same job \d{8} against job (\d{8}) "
        r"on (\w+), and `mcsolve`'s per-trajectory rise from (\d+) s to (\d+) s "
        r"\(([\d.]+)x\) sets job \d{8} against \d{8}, both on \w+\. All ran at "
        r"(\d+) threads on the (\d+)-point grid\.", _flat(doc))
    assert m, "section 5.3's cross-job Result 3 sentence has changed shape"
    (ratio, dim, ntraj, mc_job, mc_host, sub, ex_job, ex_host, big, ex_lo, ex_hi,
     ex_x, big_job, big_host, mc_lo, mc_hi, mc_x, threads, n) = m.groups()
    big_doc = _p3b_load(f"method_comparison_spin_chain_dim{big}.json")
    assert (_p3b_job(big_doc), _p3b_host(big_doc)) == (big_job, big_host)
    assert _p3b_threads(big_doc) == int(threads)
    assert big_doc["meta"]["tlist"]["n"] == int(n)
    ref_hi = big_doc["point"]["reference"]["wall_s"]
    big_mc = big_doc["point"]["methods"]["mcsolve"]
    assert _near(ex_hi, ref_hi)
    mc_doc = _p3b_load(f"method_comparison_spin_chain_dim{dim}.json")
    timing = _p3b_load("solver_timing_spin_chain.json")
    mc = mc_doc["point"]["methods"]["mcsolve"]
    assert mc["ntraj"] == int(ntraj)
    assert (_p3b_job(mc_doc), _p3b_host(mc_doc)) == (mc_job, mc_host)
    assert (_p3b_job(timing), _p3b_host(timing)) == (ex_job, ex_host)
    assert mc_job != ex_job and mc_host != ex_host
    assert _p3b_threads(mc_doc) == _p3b_threads(timing) == int(threads)
    assert mc_doc["meta"]["tlist"]["n"] == timing["meta"]["tlist"]["n"] == int(n)
    exact = _p3b_point(timing, int(dim))
    assert exact["native_substeps"] == int(sub)
    assert _p3b_ratio_near(ratio, mc["wall_s"] / exact["timings"]["native"]["median_s"])
    # dim 2048: exact 2,685 s (job ex_job) -> 17,426 s; mcsolve per trajectory
    assert _near(ex_lo, exact["timings"]["native"]["median_s"])
    assert _p3b_ratio_near(ex_x, ref_hi / exact["timings"]["native"]["median_s"])
    per_lo, per_hi = mc["wall_s"] / mc["ntraj"], big_mc["wall_s"] / big_mc["ntraj"]
    assert _near(mc_lo, per_lo) and _near(mc_hi, per_hi)
    assert _p3b_ratio_near(mc_x, per_hi / per_lo)


def test_p3b_result3_figures_join_the_jobs_section53_names(doc):
    """System A's and C's Result 3 figures each join two jobs on one node;
    the mixed chain's is one job; plot_method_comparison's guard sees it."""
    pmc = pytest.importorskip("plot_method_comparison")
    text = _flat(doc)
    m = re.search(
        r"System A's takes dims (\d+)–(\d+) from job (\d{8}) and dims (\d+) and "
        r"(\d+) from job (\d{8})\. System C's takes dims (\d+)–(\d+) from (\d{8}) "
        r"and dim (\d+) from (\d{8})\. Each pair ran on the same node, (\w+), at "
        r"(\d+) threads", text)
    assert m, "section 5.3's Result 3 figure-provenance sentence has changed shape"
    (a_lo, a_hi, a_job1, a_d1, a_d2, a_job2, c_lo, c_hi, c_job1, c_dim, c_job2,
     host, threads) = m.groups()
    groups = {
        "spin_chain": {**{d: a_job1 for d in pmc.discover_dims("spin_chain")
                          if int(a_lo) <= d <= int(a_hi)},
                       int(a_d1): a_job2, int(a_d2): a_job2},
        "oscillator_bath": {**{d: c_job1 for d in pmc.discover_dims("oscillator_bath")
                               if int(c_lo) <= d <= int(c_hi)},
                            int(c_dim): c_job2},
    }
    for system, jobs in groups.items():
        assert min(jobs) == int(a_lo if system == "spin_chain" else c_lo)
        documents = {d: _p3b_load(f"method_comparison_{system}_dim{d}.json")
                     for d in jobs}
        for d, document in documents.items():
            assert _p3b_job(document) == jobs[d], f"{system} dim {d}"
            assert _p3b_host(document) == host, f"{system} dim {d}"
            assert _p3b_threads(document) == int(threads), f"{system} dim {d}"
        keys = {pmc.execution_key(document) for document in documents.values()}
        assert len(keys) == 2, f"{system}: the guard should see two allocations"

    s = re.search(r"System C's dim-(\d+) file also ran at (\d+) substeps, where "
                  r"dims (\d+)–(\d+) ran at (\d+)\.", text)
    assert s, "the oscillator substep clause has changed shape"
    top, top_sub, lo, hi, low_sub = map(int, s.groups())
    for d in pmc.discover_dims("oscillator_bath"):
        document = _p3b_load(f"method_comparison_oscillator_bath_dim{d}.json")
        if d == top:
            assert document["meta"]["substeps"] == top_sub
        elif lo <= d <= hi:
            assert document["meta"]["substeps"] == low_sub, f"dim {d}"

    b = re.search(r"The mixed chain's figure \(dims (\d+)–(\d+)\) is one job, (\d{8})\.",
                  text)
    assert b, "the mixed chain's one-job sentence has changed shape"
    lo, hi, job = int(b.group(1)), int(b.group(2)), b.group(3)
    dims = [d for d in pmc.discover_dims("mixed_chain") if lo <= d <= hi]
    assert dims and min(dims) == lo and max(dims) == hi
    for d in dims:
        assert _p3b_job(_p3b_load(f"method_comparison_mixed_chain_dim{d}.json")) == job


def test_p3b_thread_speedup_names_its_node_and_sample_confounds(doc):
    """The 2.3x compares Result 2's 4-thread run with job 19603810's 32-thread
    one: same grid and substeps, different node, one sample against three."""
    text = _flat(doc)
    m = re.search(
        r"Job `(\d{8})` re-ran System A's exact solver and SLB at (\d+) and (\d+) "
        r"spins with (\d+) threads, where the Result 2 data uses (\d+)\. At (\d+) "
        r"spins both came out about ([\d.]+)x faster: ([\d.]+) s against ([\d.]+) s "
        r"on the reference, and ([\d.]+) s against ([\d.]+) s on SLB "
        r"\(\$M=(\d+)\$, one realization\)\. The (\d+)-point grid and the substeps "
        r"match \((\d+) on the reference, (\d+) on SLB\)\. The node does not: "
        r"(\d{8}) ran on (\w+) and Result 2's job (\d{8}) on (\w+)\. Job (\d{8}) "
        r"also timed each solve once, where Result 2 takes the median of three\.",
        text)
    assert m, "section 5.3's thread-count paragraph has changed shape"
    (job, s1, s2, t_new, t_old, size, ratio, ref_new, ref_old, slb_new, slb_old,
     m_rep, n, ref_sub, slb_sub, job_again, host_new, job_old, host_old,
     job_third) = m.groups()
    assert job == job_again == job_third
    new = _p3b_load("cost_scaling_spin_chain_dim1024.json")
    old = _p3b_load("cost_scaling_spin_chain.json")
    assert new["meta"]["params"]["sizes"] == [int(s1), int(s2)]
    for document, j, host, threads in ((new, job, host_new, t_new),
                                       (old, job_old, host_old, t_old)):
        assert _p3b_job(document) == j
        assert _p3b_host(document) == host
        assert _p3b_threads(document) == int(threads)
        assert document["meta"]["tlist"]["n"] == int(n)
        assert document["meta"]["substeps"] == int(slb_sub)
        assert document["meta"]["params"]["NATIVE_REF_SUBSTEPS"] == int(ref_sub)
        assert document["meta"]["params"]["M_REP"] == int(m_rep)
    assert host_new != host_old
    dim = 2 ** int(size)
    p_new, p_old = _p3b_point(new, dim), _p3b_point(old, dim)
    assert _near(ref_new, p_new["t_native_ref"])
    assert _near(ref_old, p_old["t_native_ref"])
    assert _near(slb_new, p_new["t_slb_fixed"])
    assert _near(slb_old, p_old["t_slb_fixed"])
    assert _p3b_ratio_near(ratio, p_old["t_native_ref"] / p_new["t_native_ref"])
    assert _p3b_ratio_near(ratio, p_old["t_slb_fixed"] / p_new["t_slb_fixed"])
    assert len(p_new["t_native_ref_repeats"]) == len(p_new["t_slb_fixed_repeats"]) == 1
    assert len(p_old["t_native_ref_repeats"]) == len(p_old["t_slb_fixed_repeats"]) == 3


def test_p3b_thread_gain_varies_between_runs_and_systems(doc):
    """A second 32-thread run of the same 9-spin solves, and the oscillator's
    reference on one node at 4 and 32 threads."""
    m = re.search(
        r"§5\.2's job (\d{8}) timed the same two (\d+)-spin solves again, also at "
        r"(\d+) threads on (\w+), once each: ([\d.]+) s and ([\d.]+) s\. That is "
        r"([\d.]+)x and ([\d.]+)x faster than Result 2, not ([\d.]+)x\. On the "
        r"oscillator the gain vanishes\. On (\w+) and the same (\d+)-point grid, its "
        r"dimension-(\d+) reference at (\d+) substeps took ([\d.]+) s at (\d+) "
        r"threads \(job (\d{8}), median of three\) and ([\d.]+) s at (\d+) \(job "
        r"(\d{8}), one sample\)\.", _flat(doc))
    assert m, "section 5.3's thread-variation paragraph has changed shape"
    (job, size, threads, host, ref_s, slb_s, ref_x, slb_x, claimed, osc_host, n,
     osc_dim, osc_sub, osc_old_s, osc_old_t, osc_old_job, osc_new_s, osc_new_t,
     osc_new_job) = m.groups()
    timing = _p3b_load("solver_timing_spin_chain.json")
    assert _p3b_job(timing) == job and _p3b_host(timing) == host
    assert _p3b_threads(timing) == int(threads)
    assert timing["meta"]["params"]["repeats"] == 1
    assert timing["meta"]["tlist"]["n"] == int(n)
    dim = 2 ** int(size)
    t = _p3b_point(timing, dim)
    assert (t["native_substeps"], t["slb_substeps"]) == (8, 4)
    assert _near(ref_s, t["timings"]["native"]["median_s"])
    assert _near(slb_s, t["timings"]["slb"]["median_s"])
    old = _p3b_point(_p3b_load("cost_scaling_spin_chain.json"), dim)
    assert _p3b_ratio_near(ref_x, old["t_native_ref"] / t["timings"]["native"]["median_s"])
    assert _p3b_ratio_near(slb_x, old["t_slb_fixed"] / t["timings"]["slb"]["median_s"])
    # "not 2.3x": the thread-count paragraph's 4-to-32-thread speed-up
    side = _p3b_point(_p3b_load("cost_scaling_spin_chain_dim1024.json"), dim)
    assert _p3b_ratio_near(claimed, old["t_native_ref"] / side["t_native_ref"])

    osc_old = _p3b_load("cost_scaling_oscillator_bath.json")
    osc_new = _p3b_load("solver_timing_oscillator_bath.json")
    assert _p3b_job(osc_old) == osc_old_job and _p3b_job(osc_new) == osc_new_job
    assert _p3b_host(osc_old) == _p3b_host(osc_new) == osc_host
    assert _p3b_threads(osc_old) == int(osc_old_t)
    assert _p3b_threads(osc_new) == int(osc_new_t)
    assert osc_old["meta"]["tlist"]["n"] == osc_new["meta"]["tlist"]["n"] == int(n)
    assert osc_old["meta"]["params"]["NATIVE_REF_SUBSTEPS"] == int(osc_sub)
    assert osc_new["meta"]["params"]["repeats"] == 1
    p_old = _p3b_point(osc_old, int(osc_dim))
    p_new = _p3b_point(osc_new, int(osc_dim))
    assert len(p_old["t_native_ref_repeats"]) == 3
    assert p_new["native_substeps"] == int(osc_sub)
    assert _near(osc_old_s, p_old["t_native_ref"])
    assert _near(osc_new_s, p_new["timings"]["native"]["median_s"])
    assert float(osc_new_s) >= float(osc_old_s), "the 32-thread run is not faster"


def test_p3b_result3_32_thread_points_ran_no_slb(doc):
    """The Result 3 points section 5.3 calls 32-thread and SLB-free are exactly
    those, each from a job of its own; every other point of those two sweeps
    ran SLB at the stated threads."""
    m = re.search(r"System A at dims (\d+) and (\d+) and System B at (\d+) ran no SLB, "
                  r"so they quote no SLB ratio\. They also ran on (\d+) threads, where "
                  r"the rest of their sweeps ran on (\d+) \(A\) or (\d+) \(B\)",
                  _flat(doc))
    assert m, "section 5.3's 32-thread-points sentence has changed shape"
    a1, a2, b1, t_new, t_a, t_b = map(int, m.groups())
    pmc = pytest.importorskip("plot_method_comparison")
    for system, extra, rest in (("spin_chain", {a1, a2}, t_a),
                                ("mixed_chain", {b1}, t_b)):
        rest_jobs, extra_jobs = set(), set()
        for d in pmc.discover_dims(system):
            document = _p3b_load(f"method_comparison_{system}_dim{d}.json")
            has_slb = "slb" in document["point"]["methods"]
            if d in extra:
                assert not has_slb and _p3b_threads(document) == t_new, f"{system} {d}"
                extra_jobs.add(_p3b_job(document))
            else:
                assert has_slb and _p3b_threads(document) == rest, f"{system} {d}"
                rest_jobs.add(_p3b_job(document))
        # the guard would see them: none shares a job with the drawn points
        assert not (extra_jobs & rest_jobs), system


def test_p3b_every_current_data_file_records_its_execution_context(doc):
    """Only the three named pre-0.6.4 frontier files lack meta.execution, and
    only the named file has an empty threads block."""
    text = _flat(doc)
    m = re.search(r"except the three pre-0\.6\.4 oscillator frontier files named at the "
                  r"top of this section\. No Result uses them\. An empty `threads` block "
                  r"means no thread count was set\. One file has it, `([\w.]+)`", text)
    assert m, "section 5.3's execution-record sentence has changed shape"
    # The three files as the top of section 5.3 names them, with their date.
    top = re.search(r"`frontier_oscillator_bath_dim(\d+)\.json`, `_dim(\d+)\.json` and "
                    r"`_dim(\d+)\.json`, written on Jul (\d+) before 0\.6\.4", text)
    assert top, "section 5.3's stale-file sentence has changed shape"
    named = {f"frontier_oscillator_bath_dim{d}.json" for d in top.group(1, 2, 3)}
    missing, empty = set(), set()
    for path in sorted(DATA.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        execution = meta.get("execution")
        if execution is None:
            missing.add(path.name)
            # 0.6.4 shipped on 29 July; these ran before it
            assert meta["timestamp"].startswith(f"2026-07-{int(top.group(4)):02d}"), path.name
            continue
        assert execution.get("hostname"), path.name
        assert (execution.get("slurm") or {}).get("job_id"), path.name
        assert "threads" in execution, path.name
        if not execution["threads"]:
            empty.add(path.name)
    assert missing == named
    assert empty == {m.group(1)}
    # "no Result uses them": the document shows no frontier figure and names
    # the oscillator frontier files only in this sentence
    assert "benchmark_frontier_" not in doc
    assert text.count("frontier_oscillator_bath") == 1


# --- section 5.3: Result 2's re-timing ------------------------------------
#
# The re-timing passage called the 88,443 s reference and Result 3's 2,413 s
# "the same quantity, 37x apart"; 2,413 s is Result 3's 4-substep solve, and
# the same 8-substep solve is its 4,819 s reference, 18x apart. Its inflation
# table said N_L 3-513 inflated 1.1-2.4x while System A's dims 256 and 512 were
# 3.1x and 3.7x, and "1.03x or better at every headline point" held for the
# exact side only. The superseded files live in git history, one commit before
# the re-timing, so the before/after numbers are recomputed from there.

_p3c_retime_commit = "014f92c"
_p3c_letter = {"A": "spin_chain", "B": "mixed_chain", "C": "oscillator_bath"}


def _p3c_superseded(system: str) -> dict:
    """cost_scaling_<system>.json as committed just before the re-timing."""
    import subprocess
    spec = f"{_p3c_retime_commit}^:benchmarks/data/cost_scaling_{system}.json"
    try:
        out = subprocess.run(["git", "show", spec], cwd=BENCHMARKS.parent,
                             capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git history with the pre-re-timing files is not available")
    return json.loads(out.stdout.decode("utf-8"))


def _p3c_current(system: str) -> dict:
    path = DATA / f"cost_scaling_{system}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p3c_pairs(system: str):
    """(dim, old point, new point) wherever both files timed the exact solve."""
    old = {p["dim"]: p for p in _p3c_superseded(system)["points"]}
    return [(p["dim"], old[p["dim"]], p) for p in _p3c_current(system)["points"]
            if p.get("t_native_ref") is not None
            and old.get(p["dim"], {}).get("t_native_ref") is not None]


def _p3c_spread(samples) -> float:
    return max(samples) / min(samples)


def test_section53_retiming_jobs_and_inflation_table_match_git_history(doc):
    text = _flat(doc)
    head = re.search(
        r"Jobs (\d{8}), (\d{8}) and (\d{8}) re-timed all three systems on "
        r"exclusive nodes, three samples per native-RK4 and SLB solve; `mesolve`, "
        r"the reference up to dim 32, was timed once in both runs\. They replace "
        r"the timings of the first runs \(jobs (\d{8}), (\d{8}) and (\d{8})\)\.", text)
    assert head, "section 5.3's re-timing job sentence has changed shape"
    systems = ("mixed_chain", "spin_chain", "oscillator_bath")
    new_jobs = {str(_p3c_current(s)["meta"]["execution"]["slurm"]["job_id"])
                for s in systems}
    assert set(head.groups()[:3]) == new_jobs
    old_jobs = {str(_p3c_superseded(s)["meta"]["execution"]["slurm"]["job_id"])
                for s in systems}
    assert set(head.groups()[3:]) == old_jobs
    import merge_retimed
    for s in systems:
        old, new = _p3c_superseded(s), _p3c_current(s)
        # "On the same grid and substeps"; "no thread variables"; three samples.
        for key in ("tlist", "substeps"):
            assert old["meta"][key] == new["meta"][key], (s, key)
        assert (old["meta"]["params"]["NATIVE_REF_SUBSTEPS"]
                == new["meta"]["params"]["NATIVE_REF_SUBSTEPS"])
        assert not old["meta"]["execution"].get("threads"), s
        assert all(len(p["t_native_ref_repeats"]) == 3 for _, _, p in _p3c_pairs(s))
        # "All three files passed" merge_retimed's check.
        assert merge_retimed.check(new, old) == [], s

    table = re.search(r"^\| system \| dims \| inflation of the native RK4 solve \|\n"
                      r"\|[-|]+\|\n((?:\|[^\n]*\|\n)+)", doc, re.M)
    assert table, "section 5.3's inflation table has moved or changed its header"
    rows = re.findall(r"^\|\s*([ABC])\s*\|\s*\**(\d+)(?:\s*–\s*(\d+))?\**\s*\|"
                      r"\s*\**([\d.]+)(?:\s*–\s*([\d.]+))?×\**\s*\|$",
                      table.group(1), re.M)
    assert len(rows) == 5, f"section 5.3's inflation table has changed shape: {rows}"
    for letter, lo, hi, f_lo, f_hi in rows:
        system = _p3c_letter[letter]
        hi = hi or lo
        factors = {d: o["t_native_ref"] / n["t_native_ref"]
                   for d, o, n in _p3c_pairs(system) if int(lo) <= d <= int(hi)}
        assert int(lo) in factors and int(hi) in factors, (letter, lo, hi)
        assert _near(f_lo, min(factors.values())), (letter, f_lo, factors)
        assert _near(f_hi or f_lo, max(factors.values())), (letter, f_hi, factors)
    covered = {(l, d) for l, lo, hi, _, _ in rows for d, _, _ in
               _p3c_pairs(_p3c_letter[l]) if int(lo) <= d <= int(hi or lo)}
    every = {(l, d) for l, s in _p3c_letter.items() for d, _, _ in _p3c_pairs(s)}
    assert covered == every, f"points missing from the table: {every - covered}"


def test_section53_headline_ratios_before_and_after_retiming(doc):
    text = _flat(doc)
    b = re.search(
        r"On System B the certified exact solve \((\d+) substeps\) over one SLB "
        r"solve \(\$M=(\d+)\$, (\d+) substeps\) falls from \$([\d{},]+)\\times\$ to "
        r"\$([\d{},]+)\\times\$ at dim 128, and from \$(\d+)\\times\$ to "
        r"\$(\d+)\\times\$ at dim 64\. SLB's own solves had been inflated less there "
        r"\(\$([\d.]+)\\times\$ and \$([\d.]+)\\times\$\)\.", text)
    assert b, "section 5.3's System B before/after sentence has changed shape"
    meta = _p3c_current("mixed_chain")["meta"]
    assert int(b.group(1)) == meta["params"]["NATIVE_REF_SUBSTEPS"]
    assert int(b.group(2)) == meta["params"]["M_REP"]
    assert int(b.group(3)) == meta["substeps"]
    pairs = {d: (o, n) for d, o, n in _p3c_pairs("mixed_chain")}
    for dim, before, after, slb in ((128, 4, 5, 8), (64, 6, 7, 9)):
        o, n = pairs[dim]
        assert _near(b.group(before), o["t_native_ref"] / o["t_slb_fixed"]), dim
        assert _near(b.group(after), n["t_native_ref"] / n["t_slb_fixed"]), dim
        assert _near(b.group(slb), o["t_slb_fixed"] / n["t_slb_fixed"]), dim
        assert o["t_slb_fixed"] / n["t_slb_fixed"] < o["t_native_ref"] / n["t_native_ref"]

    c = re.search(
        r"On System C the dim-128 ratio \((\d+) substeps over one SLB solve at "
        r"\$M=(\d+)\$, (\d+) substeps\) rises, \$(\d+)\\times\$ to \$(\d+)\\times\$, "
        r"because its SLB solve had been inflated slightly more than its exact "
        r"solve \(\$([\d.]+)\\times\$ against \$([\d.]+)\\times\$\)\.", text)
    assert c, "section 5.3's System C before/after sentence has changed shape"
    meta = _p3c_current("oscillator_bath")["meta"]
    assert int(c.group(1)) == meta["params"]["NATIVE_REF_SUBSTEPS"]
    assert int(c.group(2)) == meta["params"]["M_REP"]
    assert int(c.group(3)) == meta["substeps"]
    o, n = {d: (o, n) for d, o, n in _p3c_pairs("oscillator_bath")}[128]
    assert _near(c.group(4), o["t_native_ref"] / o["t_slb_fixed"])
    assert _near(c.group(5), n["t_native_ref"] / n["t_slb_fixed"])
    slb, ref = o["t_slb_fixed"] / n["t_slb_fixed"], o["t_native_ref"] / n["t_native_ref"]
    assert slb > ref
    assert _near(c.group(6), slb) and _near(c.group(7), ref)


def test_section53_result3_times_the_same_solve_on_a_shared_node(doc):
    """88,443 s and 2,413 s were printed as 'the same quantity, 37x apart'.
    The same solve in Result 3 is the 8-substep reference, 4,819 s: 18x."""
    m = re.search(
        r"Result 3 times the same System B solve \(dim (\d+), (\d+) substeps, the "
        r"same (\d+)-point grid\) at ([\d,]+) s\. The old Result 2 file had "
        r"([\d,]+) s, \$(\d+)\\times\$ more, and that mismatch is how the "
        r"inflation was found\. Result 3's time is still \$([\d.]+)\\times\$ the "
        r"re-measured ([\d,]+) s\. It ran in another job \((\d{8})\) on another "
        r"node, and its script, `(slurm_[\w]+\.sh)`, does not ask for an "
        r"exclusive node\.", _flat(doc))
    assert m, "section 5.3's Result 3 comparison has changed shape"
    dim, subs, grid, r3, old, gap, still, new, job, script = m.groups()
    r3_doc = json.loads((DATA / f"method_comparison_mixed_chain_dim{dim}.json")
                        .read_text(encoding="utf-8"))
    r2_point = next(p for p in _p3c_current("mixed_chain")["points"]
                    if p["dim"] == int(dim))
    old_point = next(p for p in _p3c_superseded("mixed_chain")["points"]
                     if p["dim"] == int(dim))
    ref = r3_doc["point"]["reference"]
    # The same solve: same method, grid, operator count and trajectory.
    assert ref["method"] == r2_point["reference_method"] == f"native_rk4_substeps{subs}"
    assert r3_doc["meta"]["tlist"] == _p3c_current("mixed_chain")["meta"]["tlist"]
    assert r3_doc["meta"]["tlist"]["n"] == int(grid)
    assert r3_doc["point"]["n_l"] == r2_point["n_l"]
    assert np.allclose(ref["curves"]["energy"], r2_point["reference"], atol=1e-10)
    assert _near(r3, ref["wall_s"]) and _near(old, old_point["t_native_ref"])
    assert _near(gap, old_point["t_native_ref"] / ref["wall_s"])
    assert _near(new, r2_point["t_native_ref"])
    assert _near(still, ref["wall_s"] / r2_point["t_native_ref"])
    execution = r3_doc["meta"]["execution"]
    assert str(execution["slurm"]["job_id"]) == job
    assert execution["hostname"] != _p3c_current("mixed_chain")["meta"]["execution"]["hostname"]
    sbatch = (BENCHMARKS / script).read_text(encoding="utf-8")
    assert f"--job-name={execution['slurm']['job_name']}" in sbatch
    assert "--exclusive" not in sbatch


def test_section53_timing_spread_paragraph_matches_the_repeats(doc):
    """'1.03x or better at every headline point' held for the exact solves
    only; System B's SLB samples spread 1.21x and 1.15x."""
    m = re.search(
        r"Each published native-RK4 and SLB time is the median of three samples; "
        r"`mesolve`'s is one\. Across all points, "
        r"the slowest of the three is ([\d.]+) to ([\d.]+) times the fastest\. "
        r"At the three headline points \(System B at dims 64 and 128, System C at "
        r"dim 128\) the exact solves agree to \$([\d.]+)\\times\$ or better\. The "
        r"SLB solves agree to \$([\d.]+)\\times\$ on System C, but only to "
        r"\$([\d.]+)\\times\$ and \$([\d.]+)\\times\$ on System B, where one SLB "
        r"solve takes under (\d+) s\. Dividing the median exact solve by each SLB "
        r"sample in turn gives \$(\d+)\\times\$ to \$(\d+)\\times\$ at dim 64 and "
        r"\$([\d{},]+)\\times\$ to \$([\d{},]+)\\times\$ at dim 128, around the "
        r"published \$(\d+)\\times\$ and \$([\d{},]+)\\times\$\.", _flat(doc))
    assert m, "section 5.3's timing-spread paragraph has changed shape"
    (lo, hi, ref_worst, c_slb, b64, b128, under,
     r64a, r64b, r128a, r128b, pub64, pub128) = m.groups()

    spreads = []
    for s in _p3c_letter.values():
        for p in _p3c_current(s)["points"]:
            if p.get("t_native_ref") is None or p.get("t_slb_fixed") is None:
                continue
            for key, median in (("t_native_ref_repeats", "t_native_ref"),
                                ("t_slb_fixed_repeats", "t_slb_fixed")):
                assert len(p[key]) == 3 and float(np.median(p[key])) == p[median]
                spreads.append(_p3c_spread(p[key]))
    assert _near(lo, min(spreads)) and _near(hi, max(spreads))

    b = {p["dim"]: p for p in _p3c_current("mixed_chain")["points"]}
    c = {p["dim"]: p for p in _p3c_current("oscillator_bath")["points"]}
    headline = (b[64], b[128], c[128])
    worst = max(_p3c_spread(p["t_native_ref_repeats"]) for p in headline)
    assert worst <= _printed(ref_worst) + _rounding_tolerance(ref_worst)
    assert _near(c_slb, _p3c_spread(c[128]["t_slb_fixed_repeats"]))
    assert _near(b64, _p3c_spread(b[64]["t_slb_fixed_repeats"]))
    assert _near(b128, _p3c_spread(b[128]["t_slb_fixed_repeats"]))
    assert max(b[64]["t_slb_fixed_repeats"] + b[128]["t_slb_fixed_repeats"]) < int(under)
    for p, a, z, pub in ((b[64], r64a, r64b, pub64), (b[128], r128a, r128b, pub128)):
        each = [p["t_native_ref"] / t for t in p["t_slb_fixed_repeats"]]
        assert _near(a, min(each)) and _near(z, max(each)), (a, z, each)
        assert _near(pub, p["t_native_ref"] / p["t_slb_fixed"])


def test_section53_thread_probe_is_the_one_the_retiming_script_quotes(doc):
    """The pinned/unpinned times live only in job 19599549's log; the text says
    they are quoted in slurm_r2_retime.sh, and the probe's settings are in
    slurm_thread_pinning_probe.sh. Pin both, and the hedged cause."""
    m = re.search(
        r"The cause was never pinned down\. The re-timing script, `(slurm_\w+\.sh)`, "
        r"names other jobs on the same node as the most likely one, so each re-run "
        r"took a whole node\. Thread settings were ruled out\. The first runs set "
        r"none \(their files record no thread variables\), but job (\d{8}) timed "
        r"System B at dim (\d+) both ways, back to back in one allocation: "
        r"([\d.]+) s with (\d+) threads pinned, ([\d.]+) s unpinned\.", _flat(doc))
    assert m, "section 5.3's thread-probe sentence has changed shape"
    script, job, dim, pinned, threads, unpinned = m.groups()
    retime = (BENCHMARKS / script).read_text(encoding="utf-8")
    assert "#SBATCH --exclusive" in retime
    assert "most likely" in retime and "node contention" in retime
    assert f"Job {job}" in retime
    assert f"{pinned} s against {unpinned} s" in re.sub(r"\s+", " ", retime)
    probe = (BENCHMARKS / "slurm_thread_pinning_probe.sh").read_text(encoding="utf-8")
    assert f"export OMP_NUM_THREADS={threads}" in probe
    assert "for MODE in unpinned pinned" in probe
    # --sizes 6 on the mixed chain is 6 spins: dim 2**6.
    assert "--system mixed_chain --sizes 6" in probe and int(dim) == 2 ** 6
    for s in _p3c_letter.values():
        assert _p3c_current(s)["meta"]["execution"]["threads"], s


# --- section 5.3: the code behind the data, and the old operator counts ----
#
# Section 5.3 said the Result 1 and Result 4 runners "are byte-identical" to
# the cluster copies, in the present tense, long after both had changed. It
# credited mcsolve with a 3.1x speed-up from two timings taken on different
# machines and grids, and it quoted System A's old N_L = 113 without saying
# which older construction produced it, while section 6 quotes counts from two
# other ones. These pin the corrected sentences to the files and the code.

def _p3d_text(doc: str) -> str:
    """Section 5.3's bullets are indented, which _flat keeps; fold that too."""
    return _flat_ws(_flat(doc))


def _p3d_load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p3d_date(text: str):
    """'25 August 2026' -> date."""
    import datetime
    return datetime.datetime.strptime(text, "%d %B %Y").date()


def _p3d_written(document: dict):
    """The UTC date a data file was written, from its own metadata."""
    import datetime
    return datetime.datetime.fromisoformat(document["meta"]["timestamp"]).date()


def test_section53_result4_runner_change_predates_its_files(doc):
    """Result 4's runner changed once after the 25 August check (it began
    recording s_repeats), and every committed Result 4 file was written by
    that version: one job, after the change, s_repeats on every mcsolve row,
    and the same settings the runner uses today."""
    m = re.search(
        r"So on (\d+ \w+ (\d{4})) the cluster's working copies were diffed against "
        r"`main`: `run_accuracy_vs_M\.py` and `run_isocost_vs_dim\.py` were "
        r"\*\*byte-identical\*\* to the published ones that day\. Both have changed "
        r"since\. None of the changes can move a committed number: "
        r"- \*\*Result 4\.\*\* `run_isocost_vs_dim\.py` changed once after the check, "
        r"in commit (\w+) \((\d+ \w+)\)\. The change added dim 128 on the mixed chain "
        r"and the oscillator, made the sweep cover every observable and stop on the "
        r"worst one, and made it record `s_repeats`, the spread across `mcsolve` "
        r"trajectories\. Job (\d+) wrote every Result 4 file between (\d+ \w+) and "
        r"(\d+ \w+) \(the files' timestamps\), so the cluster copy already had the "
        r"change: every `mcsolve` row in them carries `s_repeats`\.", _p3d_text(doc))
    assert m, "section 5.3's runner check has changed shape"
    checked, year, commit, changed, job, first, last = m.groups()
    checked = _p3d_date(checked)
    # The one commit to the runner after the check, and its date.
    import subprocess
    try:
        out = subprocess.run(["git", "log", "--format=%h %ad", "--date=short",
                              "--", "benchmarks/run_isocost_vs_dim.py"],
                             cwd=BENCHMARKS.parent, capture_output=True, check=True,
                             timeout=60).stdout.decode("utf-8").split("\n")
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git history is not available")
    import datetime
    after = [line.split() for line in out if line.strip()
             and datetime.date.fromisoformat(line.split()[1]) > checked]
    assert [h for h, _ in after] == [commit], f"commits since the check: {after}"
    assert datetime.date.fromisoformat(after[0][1]) == _p3d_date(f"{changed} {year}")
    written = []

    import run_isocost_vs_dim as R
    rows = 0
    for name, (_, points) in R.SYSTEMS.items():
        document = _p3d_load(DATA / f"isocost_vs_dim_{name}.json")
        meta = document["meta"]
        assert meta["execution"]["slurm"]["job_id"] == job, name
        written.append(_p3d_written(document))
        params = meta["params"]
        assert params["sizes"] == [size for size, _ in points], name
        assert params["substeps_by_size"] == {str(s): ss for s, ss in points}, name
        assert params["M_GRID"] == R.M_GRID and params["MC_FIT_GRID"] == R.MC_FIT_GRID
        assert params["MC_REPEATS"] == R.MC_REPEATS and params["rng_sweep"] == R.RNG_SWEEP
        for point in document["points"]:
            for row in point["mc_fit"]:
                assert len(row["s_repeats"]) == len(row["rmse_repeats"]), (
                    f"{name} dim {point['dim']} ntraj {row['ntraj']}: no s_repeats")
                rows += 1
    assert rows > 0
    assert (min(written), max(written)) == (_p3d_date(f"{first} {year}"),
                                            _p3d_date(f"{last} {year}"))


def test_section53_result1_files_match_the_runner_as_it_stands(doc):
    """Every Result 1 file records the seed, M ladder, substeps, time grid and
    realization count it ran with; each matches run_accuracy_vs_M.py today.
    The old-layout files are exactly the 7 August job's, the files carrying
    sweep_complete are exactly the jobs the sentence names, and every file
    predates the cheaper reference check."""
    m = re.search(
        r"- \*\*Result 1\.\*\* All (\d+) files record the seed, \$M\$ ladder, "
        r"substeps, time grid and realization count they ran with\. Each matches "
        r"what `run_accuracy_vs_M\.py` uses for that size today\. The (\d+) files "
        r"from (\d+ \w+) \(job (\d+)\) predate the check: an older version wrote "
        r"them, one that recorded only energy and coherence\. .*?"
        r"a `--realizations` option \(default still (\d+); any other count gets its "
        r"own `_r<R>` file, like job (\d+)'s (\d+)-realization run\), a cap of (\d+) "
        r"realizations on System A's (\d+)-spin size, and a save after every \$M\$\. "
        r"Since (\d+ \w+) its reference check no longer repeats the reference solve "
        r"\(([\d.]+) solves instead of ([\d.]+)\); every committed file is older\. "
        r".*?Four files were run by the changed version and carry the "
        r"`sweep_complete` field only it writes: jobs (\d+), (\d+), (\d+) and (\d+)\.",
        _p3d_text(doc))
    assert m, "section 5.3's Result 1 runner bullet has changed shape"
    (q_files, q_old_n, q_old_day, q_old_job, q_default, q_side_job, q_side_r,
     q_cap, q_cap_spins, q_since, q_new, q_old, *q_jobs) = m.groups()
    checked = re.search(r"So on (\d+ \w+ (\d{4})) the cluster's working copies",
                        _p3d_text(doc))
    assert checked, "section 5.3 no longer dates the runner check"
    year = checked.group(2)

    import run_accuracy_vs_M as R1
    assert int(q_default) == R1.DEFAULT_REALIZATIONS
    assert R1.REALIZATION_CAPS == {("spin_chain", int(q_cap_spins)): int(q_cap)}
    assert (float(q_new), float(q_old)) == (1.5, 2.5)
    grid = {"t0": float(common.TLIST_FINE[0]), "t1": float(common.TLIST_FINE[-1]),
            "n": len(common.TLIST_FINE)}

    paths = sorted(DATA.glob("accuracy_vs_M_*_dim*.json"))
    assert len(paths) == int(q_files), f"{len(paths)} Result 1 files, not {q_files}"
    later, old_layout, newest = set(), [], None
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        meta, params = document["meta"], document["meta"]["params"]
        name, size = params["system"], params["size"]
        ladder, substeps = {s: (lad, sub) for s, lad, sub in R1.SYSTEMS[name][1]}[size]
        assert params["rng"] == R1.RNG, path.name
        assert params["M_LADDER"] == ladder, path.name
        assert document["substeps"] == substeps, path.name
        assert {k: meta["tlist"][k] for k in grid} == grid, path.name
        suffix = re.search(r"_r(\d+)$", path.stem)
        realizations = int(suffix.group(1)) if suffix else R1.DEFAULT_REALIZATIONS
        assert params["N_REALIZATIONS"] == realizations, path.name
        job = meta["execution"]["slurm"]["job_id"]
        if suffix:
            assert (job, realizations) == (q_side_job, int(q_side_r)), path.name
        if "sweep_complete" in document:
            later.add(job)
        if "observables" not in document:          # energy/coherence-only layout
            old_layout.append((job, _p3d_written(document)))
        written = _p3d_written(document)
        newest = written if newest is None else max(newest, written)
    assert later == set(q_jobs)
    assert len(old_layout) == int(q_old_n)
    assert {job for job, _ in old_layout} == {q_old_job}
    assert {day for _, day in old_layout} == {_p3d_date(f"{q_old_day} {year}")}
    assert _p3d_date(f"{q_old_day} {year}") < _p3d_date(checked.group(1))
    assert newest < _p3d_date(f"{q_since} {year}"), "a file postdates the check change"
    more = re.search(r"Jobs (\d+) \((\d+ \w+)\) and (\d+) \((\d+ \w+)\) also ran "
                     r"before it, under versions that differ from the checked one "
                     r"only by the sizes added in (\w+) and (\w+)\. So for those "
                     r"(\d+) the recorded settings", _p3d_text(doc))
    assert more, "section 5.3's pre-check job sentence has changed shape"
    job_a, day_a, job_b, day_b, _c1, _c2, total = more.groups()
    stamps = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        stamps.setdefault(document["meta"]["execution"]["slurm"]["job_id"], set()).add(
            _p3d_written(document))
    for job, day in ((job_a, day_a), (job_b, day_b)):
        assert stamps[job] == {_p3d_date(f"{day} {year}")}, (job, stamps[job])
        assert _p3d_date(f"{day} {year}") < _p3d_date(checked.group(1))
    assert int(total) == int(q_old_n) + 2


def test_section53_mcsolve_timings_at_the_two_counts_are_not_comparable(doc):
    """The old and new mcsolve timings on the chain at dim 64 differ in machine,
    time grid and code, not only in N_L; the sentence must say so, and every
    fact it gives about the two files must hold."""
    m = re.search(
        r"on the chain at dim (\d+) the count fell \$([\d.]+)\\times\$, from (\d+) "
        r"\((0\.6\.\d)'s count\) to (\d+)\. No run has timed `mcsolve` at both counts "
        r"on one machine\. The timings at (\d+) \(`data/legacy/([\w.]+)`, (\d+) to "
        r"(\d+) trajectories\) ran on a Windows machine on the (\d+)-point grid\. "
        r"Result 3's at (\d+) \((\d+) trajectories, job (\d+)\) ran on `(\w+)` on the "
        r"(\d+)-point grid\. Machine, grid and code version all differ, so the two "
        r"cannot be compared\. .*?the old data allowed `M` up to (\d+), the corrected "
        r"construction caps it at (\d+)\.", _p3d_text(doc))
    assert m, "section 5.3's mcsolve paragraph has changed shape"
    (dim, ratio, old, version, new, old2, legacy_name, lo, hi, old_grid, new2, ntraj,
     job, host, new_grid, old3, new3) = m.groups()
    assert old == old2 == old3 and new == new2 == new3

    legacy = _p3d_load(DATA / "legacy" / legacy_name)
    assert legacy["dim"] == int(dim) and legacy["n_l"] == int(old)
    assert legacy["meta"]["qutip_bundling"] == version
    assert legacy["meta"]["platform"].startswith("Windows")
    assert "execution" not in legacy["meta"], "the legacy run has no Slurm record"
    assert legacy["meta"]["tlist"]["n"] == int(old_grid)
    counts = sorted(row["ntraj"] for row in legacy["mc"])
    assert (counts[0], counts[-1]) == (int(lo), int(hi))

    current = _p3d_load(DATA / f"method_comparison_spin_chain_dim{dim}.json")
    point, meta = current["point"], current["meta"]
    assert point["dim"] == int(dim) and point["n_l"] == int(new)
    assert point["methods"]["mcsolve"]["ntraj"] == int(ntraj)
    assert meta["execution"]["slurm"]["job_id"] == job
    assert meta["execution"]["hostname"] == host
    assert meta["qutip_bundling"] != version
    assert meta["tlist"]["n"] == int(new_grid) != int(old_grid)
    assert _near(ratio, int(old) / int(new))


def test_section53_old_operator_counts_name_their_construction(doc, monkeypatch):
    """Section 5.3's table of System A's operator counts under four
    constructions: 0.6.4 recomputed through davies_operator_count, the
    no-cutoff count through the same code with its roundoff floor set to zero,
    and the 0.6.3 and 0.6.2 counts read from the files that recorded them."""
    from qutip_bundling import operators

    rows = {label: tuple(int(v) for v in values) for label, *values in re.findall(
        r"^\| (0\.6\.4|0\.6\.3|0\.6\.2|no cutoff)[,:][^|]*\| (\d+) \| (\d+) \| (\d+) \|",
        doc, re.M)}
    assert set(rows) == {"0.6.4", "0.6.3", "0.6.2", "no cutoff"}, (
        "section 5.3's operator-count table has changed shape")

    def count(H, X):
        return operators.davies_operator_count(
            H, X, common.gamma, degeneracy_tol=common.DAVIES_DEGENERACY_TOL)

    chains = [common.build_spin_chain(spins)[:2] for spins in (4, 5, 6)]
    assert rows["0.6.4"] == tuple(count(H, X) for H, X in chains)
    monkeypatch.setattr(operators, "_coupling_roundoff_floor", lambda X_eig: 0.0)
    assert rows["no cutoff"] == tuple(count(H, X) for H, X in chains)

    for dim, n_l in zip((16, 32, 64), rows["0.6.3"]):
        legacy = _p3d_load(DATA / "legacy" / f"accuracy_vs_M_spin_chain_dim{dim}.json")
        assert legacy["meta"]["qutip_bundling"] == "0.6.3"
        assert legacy["meta"]["davies"]["construction"] == "grouped_frequency_sectors"
        assert "coupling_block_floor" not in legacy["meta"]["davies"]
        assert (legacy["dim"], legacy["n_l"]) == (dim, n_l)

    for suffix, dim, n_l in zip(("", "_dim32", "_dim64"), (16, 32, 64), rows["0.6.2"]):
        old = _p3d_load(BENCHMARKS / f"convergence_progress_spin_chain{suffix}.json")
        assert "meta" not in old, "the pre-0.6.3 files carry no provenance block"
        assert (old["dim"], old["n_l"]) == (dim, n_l)


def test_section53_mesolve_retiming_is_its_largest_single_sample_move(doc):
    """mesolve was timed once in both runs, so the inflation table leaves it
    out; the sentence quotes its largest move, old over new t_full, over every
    system and dimension both files timed."""
    m = re.search(r"`mesolve`, timed once each time, moved more at small sizes: up "
                  r"to ([\d.]+)x on System ([ABC]) at dim (\d+) \(([\d.]+) s to "
                  r"([\d.]+) s\)\.", _flat(doc))
    assert m, "section 5.3's mesolve re-timing sentence has changed shape"
    q_ratio, q_letter, q_dim, q_old, q_new = m.groups()
    moves = []
    for letter, system in _p3c_letter.items():
        old = {p["dim"]: p for p in _p3c_superseded(system)["points"]}
        for point in _p3c_current(system)["points"]:
            before = old.get(point["dim"], {}).get("t_full")
            after = point.get("t_full")
            if before and after and math.isfinite(before) and math.isfinite(after):
                moves.append((before / after, letter, point["dim"], before, after))
    ratio, letter, dim, before, after = max(moves)
    assert (letter, dim) == (q_letter, int(q_dim)), (letter, dim)
    assert _near(q_ratio, ratio) and _near(q_old, before) and _near(q_new, after)


# --- Result 2's panel description (part 4, unit U1) ------------------------
# The panel text described a dashed vertical line no figure draws, called
# System A's mesolve stop a 60 s budget crossing (it is the size cap), said
# averaging shrinks a variance as 1/sqrt(N), called the one hatched bar out of
# reach (its sweep hit the stop floor), and read a 33-59% bias share as "the
# larger half". Every number below is recomputed through plot_cost_scaling.

_P4A_SYSTEMS = {"A": "spin_chain", "B": "mixed_chain", "C": "oscillator_bath"}

_P4A_COUNT = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8}


def _p4a_region(doc: str) -> str:
    """Result 2 from its heading to its cost-curve subsection, flattened."""
    text = _flat(doc)
    start = text.index("### Result 2 — cost scaling versus the exact solver")
    return text[start:text.index("#### The Cost Curves", start)]


def _p4a_load(system: str) -> dict:
    path = DATA / f"cost_scaling_{system}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p4a_bars(system: str):
    """{dim: (M*, N_L, bias share in %, sweep entry at M*)} for the dims that
    reach the target, plus derive_iso's unreached list, as the figure draws
    them."""
    pcs = pytest.importorskip("plot_cost_scaling")
    document = _p4a_load(system)
    n_acc = document["meta"]["params"]["N_ACC"]
    m_star, _, bias_sq, noise_sq, unreached = pcs.derive_iso(
        document["points"], pcs.TARGET_BY_SYSTEM[system], pcs.ESTIMATE_TYPE,
        pcs.ERROR_TYPE, n_acc)
    bars = {}
    for p, m, b, n in zip(document["points"], m_star, bias_sq, noise_sq):
        if m == m:
            entry = next(e for e in p["m_sweep"] if e.get("M") == int(m))
            bars[p["dim"]] = (int(m), p["n_l"], 100.0 * b / (b + n), entry)
    return bars, unreached


def test_result2_top_panel_names_the_mesolve_cap_not_a_vertical_line(doc):
    """No committed Result 2 figure draws a vertical line: the dashed red line
    is mesolve's extrapolation. mesolve stops at the max_full_dim cap on all
    three systems; only B and C had also crossed the 60 s budget there."""
    text = _p4a_region(doc)
    source = (BENCHMARKS / "plot_cost_scaling.py").read_text(encoding="utf-8")
    assert "axvline" not in source, "the figure now draws a vertical line; say so"
    assert "vertical line" not in text, "no committed figure draws a vertical line"

    m = re.search(r"It stops at dim (\d+) on all three systems, because the "
                  r"cost-scaling runner never starts `mesolve` above that size "
                  r"\(`max_full_dim = (\d+)`\)", text)
    assert m, "Result 2's mesolve-cap sentence has changed shape"
    cap = int(m.group(1))
    assert int(m.group(2)) == cap == common.MAX_FULL_DIM
    slope = re.search(r"extends the curve's two-point slope through dims (\d+) and "
                      r"(\d+); it is an extrapolation", text)
    assert slope, "the extrapolation sentence has changed shape"
    budget = re.search(r"On Systems B and C one `mesolve` solve at dim (\d+) already "
                       r"took (\d+) s and (\d+) s, over the (\d+) s time budget", text)
    assert budget, "the B/C budget sentence has changed shape"
    a = re.search(r"On System A it took ([\d.]+) s there, so only the size cap "
                  r"stops it", text)
    assert a, "the System A cap sentence has changed shape"
    assert int(budget.group(1)) == cap
    past = re.search(r"The pale dashed red line past dim (\d+) extends", text)
    native = re.search(r"Past dim (\d+) the exact solver on the plot is native RK4", text)
    assert past and native and int(past.group(1)) == int(native.group(1)) == cap

    printed = {"mixed_chain": budget.group(2), "oscillator_bath": budget.group(3),
               "spin_chain": a.group(1)}
    for system, value in printed.items():
        document, dims, _, full, _, _ = _cost_curves(system)
        assert document["meta"]["max_full_dim"] == cap, f"{system}: cap"
        limit = document["meta"]["full_time_budget_s"]
        assert float(budget.group(4)) == limit
        ran = dims[np.isfinite(full)]
        assert int(ran[-1]) == cap, f"{system}: mesolve's last dim is {ran[-1]}"
        t_cap = float(full[dims == cap][0])
        assert _near(value, t_cap), f"{system}: printed {value} s, data {t_cap:.2f} s"
        _, n, lo, hi = _fit(dims, full)
        assert (n, lo, hi) == (2, int(slope.group(1)), int(slope.group(2))), system
        if system == "spin_chain":
            assert t_cap <= limit and document["wall_dim"] > cap, (
                "System A's mesolve now crosses the budget; the cap is not the reason")
        else:
            assert t_cap > limit and document["wall_dim"] == cap, (
                f"{system}: mesolve no longer crosses the budget at dim {cap}")


def test_result2_light_bar_is_one_runs_variance(doc):
    """The light bar is Std^2, one run's variance. Averaging N runs divides it
    by N (SEM^2 = Std^2 / N); only its square root falls as 1/sqrt(N)."""
    pcs = pytest.importorskip("plot_cost_scaling")
    text = _p4a_region(doc)
    assert pcs.ESTIMATE_TYPE == "single", "the bars are no longer one run's split"
    assert ("The light part is one run's variance ($\\text{Std}^2$). Averaging "
            "$N_{\\text{real}}$ runs divides it by $N_{\\text{real}}$") in text
    assert "shrinks the light part" not in text
    m = re.search(r"Both parts are estimated from the (\d+) runs made at each \$M\$",
                  text)
    assert m, "the bottom-panel sample-count sentence has changed shape"
    for system in _P4A_SYSTEMS.values():
        document = _p4a_load(system)
        n_acc = document["meta"]["params"]["N_ACC"]
        assert n_acc == int(m.group(1)), f"{system}: N_ACC is {n_acc}"
        entry = next(e for p in document["points"] for e in p.get("m_sweep") or []
                     if e.get("sem_sq") is not None)
        split = pcs.get_metrics(entry, n_acc)
        assert math.isclose(split["single"]["noise_sq"],
                            n_acc * split["ensemble"]["noise_sq"])


def test_result2_hatched_bar_is_a_sweep_stop_not_a_miss(doc):
    """The only hatched bar, System C at dim 8, is where the sweep stopped
    because the 16-run average fell below SWEEP_STOP_RMSE, before M reached
    N_L; the one-run RMSE there is still above the target."""
    pcs = pytest.importorskip("plot_cost_scaling")
    text = _p4a_region(doc)
    m = re.search(
        r"There is one, System C at dim (\d+), and it does not show the target out "
        r"of reach: the sweep stopped at \$M=(\d+)\$ of \$N_L=(\d+)\$ because the "
        r"(\d+)-run average's RMSE \(([\d.]+)\) was already below the sweep's stop "
        r"floor of ([\d.]+)\. One run's RMSE at \$M=(\d+)\$ is ([\d.]+), above the "
        r"([\d.]+) target, and \$M=(\d+)\$ and (\d+) were never tried\.", text)
    assert m, "Result 2's hatched-bar sentence has changed shape"
    (dim, m_last, n_l, n_runs, ens, floor, m_again, single, target,
     untried_a, untried_b) = m.groups()

    for letter, system in _P4A_SYSTEMS.items():
        _, unreached = _p4a_bars(system)
        dims = [int(u[0]) for u in unreached]
        assert dims == ([int(dim)] if letter == "C" else []), (
            f"System {letter}: hatched bars at {dims}")

    document = _p4a_load("oscillator_bath")
    params = document["meta"]["params"]
    point = next(p for p in document["points"] if p["dim"] == int(dim))
    sweep = point["m_sweep"]
    last = sweep[-1]
    assert int(m_last) == int(m_again) == last["M"]
    assert int(n_l) == point["n_l"] and int(n_runs) == params["N_ACC"]
    assert float(floor) == params["SWEEP_STOP_RMSE"]
    assert last["rmse"] <= params["SWEEP_STOP_RMSE"], "the sweep did not hit its floor"
    assert last["M"] < point["n_l"], "the sweep ran out of operators, not floor"
    assert _near(ens, last["rmse"]), f"ensemble RMSE {last['rmse']:.5f}"
    one_run = pcs.get_metrics(last, params["N_ACC"])["single"]["rmse"]
    assert _near(single, one_run), f"one-run RMSE {one_run:.5f}"
    assert float(target) == pcs.TARGET_BY_SYSTEM["oscillator_bath"] < one_run
    swept = {e["M"] for e in sweep}
    never = sorted({min(g, point["n_l"]) for g in params["M_SWEEP_GRID"]} - swept)
    assert never == [int(untried_a), int(untried_b)], f"untried M: {never}"


def test_result2_bias_shares_match_the_bottom_panel(doc):
    """System A's shares are under half at four of eight sizes, so they are
    not 'the larger half'; B's are noise-dominated; C's ladder reaches dim 128.
    Shares are bias^2 / (bias^2 + Std^2) at M*, as the bottom panel draws them."""
    pcs = pytest.importorskip("plot_cost_scaling")
    text = _p4a_region(doc)
    assert "larger half" not in text and "nothing to compress" not in text
    for letter, system in _P4A_SYSTEMS.items():
        t = re.search(rf"On \*\*System {letter}\*\* \(target ([\d.]+)\)", text)
        assert t, f"System {letter}'s bias-share sentence has changed shape"
        assert float(t.group(1)) == pcs.TARGET_BY_SYSTEM[system]

    # System A: every share, the count under half, and M* against N_L
    bars, _ = _p4a_bars("spin_chain")
    m = re.search(r"The bias share is ((?:\d+%, )+\d+%) and (\d+)% across dims "
                  r"(\d+) to (\d+): under half at (\w+) of the (\w+) sizes", text)
    assert m, "System A's share list has changed shape"
    printed = re.findall(r"(\d+)%", m.group(1)) + [m.group(2)]
    dims = sorted(bars)
    assert (dims[0], dims[-1]) == (int(m.group(3)), int(m.group(4)))
    assert len(printed) == len(dims) == _P4A_COUNT[m.group(6)]
    for value, d in zip(printed, dims):
        assert _near(value, bars[d][2]), f"A dim {d}: {value}% vs {bars[d][2]:.2f}%"
    under = sum(bars[d][2] < 50 for d in dims)
    assert under == _P4A_COUNT[m.group(5)], f"{under} shares under half"
    r = re.search(r"from dim (\d+) up it is (\d+)% to (\d+)% of \$N_L\$ "
                  r"\(((?:\d+ of \d+, )+\d+ of \d+)\)", text)
    assert r, "System A's M*/N_L clause has changed shape"
    upper = [d for d in dims if d >= int(r.group(1))]
    pairs = [tuple(map(int, x)) for x in re.findall(r"(\d+) of (\d+)", r.group(4))]
    assert pairs == [bars[d][:2] for d in upper]
    fractions = [100.0 * ms / nl for ms, nl in pairs]
    assert _near(r.group(2), min(fractions)) and _near(r.group(3), max(fractions))

    # System B: the clamped zeros and the range at the other sizes
    bars, _ = _p4a_bars("mixed_chain")
    m = re.search(r"the bias share is (\d+)% at dims (\d+) and (\d+) \(there the "
                  r"estimated \$\\text\{bias\}\^2\$ comes out negative, too small to "
                  r"see under the noise, and is drawn as zero\) and (\d+)% to "
                  r"(\d+)% at dims (\d+) to (\d+)\.", text)
    assert m, "System B's share clause has changed shape"
    zero_dims = [int(m.group(2)), int(m.group(3))]
    for d in zero_dims:
        entry = bars[d][3]
        assert int(m.group(1)) == 0 and bars[d][2] == 0.0
        assert entry["mse"] - entry["sem_sq"] < 0, f"B dim {d}: bias not clamped"
    rest = [d for d in sorted(bars) if d not in zero_dims]
    assert (rest[0], rest[-1]) == (int(m.group(6)), int(m.group(7)))
    shares = [bars[d][2] for d in rest]
    assert _near(m.group(4), min(shares)) and _near(m.group(5), max(shares))
    assert max(bars[d][2] for d in bars) < 50, "B is no longer noise-dominated"

    # System C: the M* ladder and the share at each rung
    bars, _ = _p4a_bars("oscillator_bath")
    m = re.search(r"\(\$((?:\d+ \\to )+\d+)\$ over dims (\d+) to (\d+); bias share "
                  r"(\d+)% at dim (\d+), (\d+)% at (\d+), (\d+)% at (\d+), (\d+)% at "
                  r"(\d+)\)", text)
    assert m, "System C's ladder clause has changed shape"
    dims = sorted(bars)
    assert (dims[0], dims[-1]) == (int(m.group(2)), int(m.group(3)))
    ladder = [int(v) for v in m.group(1).split(r" \to ")]
    assert ladder == [bars[d][0] for d in dims]
    g = m.groups()[3:]
    quoted = {int(g[i + 1]): g[i] for i in range(0, len(g), 2)}
    assert sorted(quoted) == dims
    for d, value in quoted.items():
        assert _near(value, bars[d][2]), f"C dim {d}: {value}% vs {bars[d][2]:.2f}%"


# --- Result 2's cost curves: the mesolve cap and the fixed-M exponents -----
#
# The paragraph quoted a 219.63 s dim-64 mesolve from Result 3's 8-thread job
# as if it were this sweep's, and blamed "too slow" for a stop the
# max_full_dim cap makes. Its System A fit claimed dims 4-512 where the floor
# keeps 32-512, and its System C bullet described a flattening the data do not
# show: the local slope rises at every doubling on both systems.

_P4B_WORDS = {"two": 2, "three": 3}


def _p4b_local_slopes(dims, times):
    """Slope of log(time) against log(dim) between each pair of neighbours."""
    return np.diff(np.log(times)) / np.diff(np.log(dims))


def _p4b_fixed_m(system: str):
    """(dims, one SLB run at fixed M) as the figure plots it, single-run view."""
    import plot_cost_scaling as pcs
    document = _cost_curves(system)[0]
    assert pcs.ESTIMATE_TYPE == "single", "the figure no longer costs one run"
    dims = common.as_array([p["dim"] for p in document["points"]])
    slb = common.as_array([p.get("t_slb_fixed") for p in document["points"]])
    return dims, slb


def test_result2_mesolve_cap_and_the_dim64_time_are_sourced(doc):
    """This sweep stops mesolve at max_full_dim, not because it is slow; the
    219.63 s comes from Result 3's job, at twice this sweep's threads."""
    text = _flat(doc)
    m = re.search(
        r"This sweep never starts `mesolve` above dim (\d+) \(`max_full_dim = (\d+)` "
        r"in all three data files\)\. The cap exists because the sweep's ([\d.]+) s "
        r"budget can act only after a solve returns: it cannot stop one long solve "
        r"that has already started\. On Systems B and C the dim-(\d+) solve took "
        r"([\d.]+) s and ([\d.]+) s, past the budget\. On System A it took ([\d.]+) s, "
        r"so there the cap, not the budget, ends the curve\. `mesolve` does run at "
        r"dim (\d+) on System A: Result 3 timed it at ([\d.]+) s, but in another job "
        r"\((\d{8})\) with (\d+) threads against this sweep's (\d+), so that time is "
        r"not a point on this plot\.", text)
    assert m, "Result 2's mesolve-cap sentences have changed shape"
    (cap, cap_meta, budget, cap_dim, t_b, t_c, t_a,
     dim, secs, job, thr, thr_sweep) = m.groups()
    assert int(cap) == int(cap_meta) == int(cap_dim) == common.MAX_FULL_DIM
    assert float(budget) == common.FULL_TIME_BUDGET
    runs = {}
    for system in ("spin_chain", "mixed_chain", "oscillator_bath"):
        document, dims, _, full, _, _ = _cost_curves(system)
        meta = document["meta"]
        assert meta["max_full_dim"] == int(cap), system
        assert meta["full_time_budget_s"] == float(budget), system
        assert meta["execution"]["threads"]["OMP_NUM_THREADS"] == thr_sweep, system
        assert not np.isfinite(full[dims > int(cap)]).any(), f"{system}: mesolve past the cap"
        runs[system] = float(full[dims == int(cap)][0])
    assert _near(t_a, runs["spin_chain"]) and runs["spin_chain"] < float(budget)
    assert _near(t_b, runs["mixed_chain"]) and runs["mixed_chain"] > float(budget)
    assert _near(t_c, runs["oscillator_bath"]) and runs["oscillator_bath"] > float(budget)

    assert int(dim) == 2 * int(cap), "the Result 3 time must be the size just past the cap"
    path = DATA / f"method_comparison_spin_chain_dim{dim}.json"
    assert path.exists(), f"BENCHMARKS.md quotes Result 3's dim-{dim} mesolve time but {path.name} is not committed"
    result3 = json.loads(path.read_text(encoding="utf-8"))
    assert _near(secs, result3["point"]["methods"]["mesolve"]["wall_s"])
    execution = result3["meta"]["execution"]
    assert execution["slurm"]["job_id"] == job
    assert execution["threads"]["OMP_NUM_THREADS"] == thr and thr != thr_sweep


def test_result2_fixed_m_exponents_state_their_range_and_curvature(doc):
    """Each fixed-M exponent is fit_slope's, over the dimensions it keeps, and
    the quoted local slopes are the per-doubling ones inside that range."""
    import plot_cost_scaling as pcs
    text = _flat(doc)
    spin = re.search(
        r"\*\*System A \(Spin chain\):\*\* Fixed \$M\$ cost fits \$N\^\{([\d.]+)\}\$ "
        r"over dims (\d+) to (\d+) \((\d+) points; dims (\d+) to (\d+) take under "
        r"([\d.]+) s and are left out\)\. The curve is still bending upward: the local "
        r"slope rises at every doubling in that range, from \$N\^\{([\d.]+)\}\$ \(dims "
        r"(\d+) to (\d+)\) to \$N\^\{([\d.]+)\}\$ \(dims (\d+) to (\d+)\)\.", text)
    assert spin, "Result 2's System A fixed-M bullet has changed shape"
    mixed = re.search(
        r"Complete to dimension 128 \(job \d{8}, run on an exclusive node\)\. Fixed "
        r"\$M\$ cost fits \$N\^\{([\d.]+)\}\$ over dims (\d+) to (\d+) \((\d+) points; "
        r"dims (\d+) to (\d+) take under ([\d.]+) s and are left out\), and both "
        r"doublings in that range are near \$N\^\{([\d.]+)\}\$ on their own\.", text)
    assert mixed, "Result 2's System B fixed-M sentence has changed shape"
    osc = re.search(
        r"\*\*System C \(Oscillator\):\*\* Fixed \$M\$ cost fits \$N\^\{([\d.]+)\}\$ "
        r"over dims (\d+) to (\d+) \((\d+) points\)\. The local slope rises at every "
        r"doubling: \$N\^\{([\d.]+)\}\$, \$N\^\{([\d.]+)\}\$, \$N\^\{([\d.]+)\}\$ and "
        r"\$N\^\{([\d.]+)\}\$, from dims (\d+) to (\d+) through dims (\d+) to (\d+)\.",
        text)
    assert osc, "Result 2's System C fixed-M bullet has changed shape"
    assert "BLAS regime" not in text, "the refuted BLAS flattening is back"
    averages = re.findall(r"So \$N\^\{([\d.]+)\}\$ is an average over a curve", text)
    assert averages == [spin.group(1), osc.group(1)], averages

    # Systems A and B: the fit, its range and count, and the dropped sizes.
    for match, system in ((spin, "spin_chain"), (mixed, "mixed_chain")):
        g = match.groups()
        dims, slb = _p4b_fixed_m(system)
        s, n, lo, hi = _fit(dims, slb)
        assert _near(g[0], s), f"{system}: printed {g[0]}, fit {s:.3f}"
        assert (int(g[1]), int(g[2]), int(g[3])) == (lo, hi, n), system
        assert float(g[6]) == pcs.FIT_FLOOR_SECONDS
        dropped = dims[np.isfinite(slb) & (dims < lo)]
        assert (int(g[4]), int(g[5])) == (int(dropped[0]), int(dropped[-1])), system
        assert (slb[np.isin(dims, dropped)] < pcs.FIT_FLOOR_SECONDS).all(), system

    # System A: still steepening; first and last doubling inside the fit.
    g = spin.groups()
    dims, slb = _p4b_fixed_m("spin_chain")
    _, _, lo, hi = _fit(dims, slb)
    window = (dims >= lo) & (dims <= hi)
    local = _p4b_local_slopes(dims[window], slb[window])
    assert (np.diff(local) > 0).all(), f"System A local slopes no longer rise: {local}"
    assert _near(g[7], local[0]) and (int(g[8]), int(g[9])) == tuple(dims[window][:2])
    assert _near(g[10], local[-1]) and (int(g[11]), int(g[12])) == tuple(dims[window][-2:])

    # System B: both doublings in the fitted range print as the fit itself.
    g = mixed.groups()
    dims, slb = _p4b_fixed_m("mixed_chain")
    _, _, lo, hi = _fit(dims, slb)
    window = (dims >= lo) & (dims <= hi)
    local = _p4b_local_slopes(dims[window], slb[window])
    assert len(local) == 2 and all(_near(g[7], x) for x in local), local
    assert g[7] == g[0]

    # System C: every doubling, all rising.
    g = osc.groups()
    dims, slb = _p4b_fixed_m("oscillator_bath")
    s, n, lo, hi = _fit(dims, slb)
    assert _near(g[0], s) and (int(g[1]), int(g[2]), int(g[3])) == (lo, hi, n)
    window = (dims >= lo) & (dims <= hi)
    local = _p4b_local_slopes(dims[window], slb[window])
    assert (np.diff(local) > 0).all(), f"System C local slopes no longer rise: {local}"
    assert len(local) == 4 and all(_near(p, x) for p, x in zip(g[4:8], local))
    assert (int(g[8]), int(g[9])) == tuple(dims[window][:2])
    assert (int(g[10]), int(g[11])) == tuple(dims[window][-2:])


def test_result2_fitted_exponent_note_describes_fit_slope(doc):
    """The note carried exponents from superseded sweeps, tied to no curve.
    It now states the rule fit_slope applies: the rising tail, the floor that
    stops at two points, and the two-point label."""
    import plot_cost_scaling as pcs
    text = _flat(doc)
    m = re.search(
        r"\*Note on fitted exponents:\* Every exponent in the figure legends, and "
        r"every fitted cost exponent in Result 2 that does not name its own range, "
        r"comes from `fit_slope` in `plot_cost_scaling\.py`\. It "
        r"keeps the longest run of rising times that ends at the largest dimension\. "
        r"Then it drops leading points under ([\d.]+) s, as long as more than (\w+) "
        r"remain: .*? A fit left with (\w+) points is labelled a local slope, not an "
        r"exponent\.", text)
    assert m, "Result 2's note on fitted exponents has changed shape"
    floor, keep, label = m.groups()
    assert float(floor) == pcs.FIT_FLOOR_SECONDS
    # The two named-range exponents are plain least-squares fits over those dims.
    eq = re.search(r"The equal-range exponents below \(System A's \$N\^\{([\d.]+)\}\$ "
                   r"for both curves over dims (\d+) to (\d+), System B's exact "
                   r"\$N\^\{([\d.]+)\}\$ over dims (\d+) to (\d+)\) are plain "
                   r"least-squares fits over the dims named", text)
    assert eq, "the equal-range sentence of the note has changed shape"
    for system, (q, lo, hi) in (("spin_chain", eq.group(1, 2, 3)),
                                ("mixed_chain", eq.group(4, 5, 6))):
        _, dims_eq, native_eq, _, _, _ = _cost_curves(system)
        sel = (dims_eq >= int(lo)) & (dims_eq <= int(hi)) & np.isfinite(native_eq)
        slope = np.polyfit(np.log(dims_eq[sel]), np.log(native_eq[sel]), 1)[0]
        assert _near(q, slope), f"{system}: {q} against {slope:.3f}"
    assert _P4B_WORDS[label] == pcs.FIT_MIN_POINTS - 1
    # The rule as stated, on series built to exercise each clause.
    dims = np.array([4.0, 8.0, 16.0, 32.0])
    assert pcs.fit_slope(dims, np.array([1.0, 0.5, 1.0, 2.0]))[1] == 3   # rising tail
    assert pcs.fit_slope(dims, np.array([0.01, 0.02, 0.04, 0.08]))[1] == _P4B_WORDS[keep]
    assert "$N^{3.35}$" not in text and "$N^{2.36}$" not in text


# --- part 4, unit U3: Result 2's iso-accuracy passage ----------------------
# One run's error, as the figure scores it: plot_cost_scaling.get_metrics'
# "single" entry (sqrt(bias^2 + Std^2)) with N_ACC from the file's metadata.

def _p4c_doc(system: str) -> dict:
    path = DATA / f"cost_scaling_{system}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p4c_single(system: str, bundles: int) -> dict:
    """{dim: (rmse, bias, std)} of one run at a fixed M, every dim that swept it."""
    import plot_cost_scaling as pcs
    document = _p4c_doc(system)
    n_acc = document["meta"]["params"]["N_ACC"]
    out = {}
    for p in document["points"]:
        for e in p.get("m_sweep") or []:
            if e.get("M") == bundles and (m := pcs.get_metrics(e, n_acc)):
                s = m["single"]
                out[p["dim"]] = (float(s["rmse"]), float(s["bias"]), float(s["std"]))
    return out


def _p4c_ladder(system: str, target: float):
    """(dims, M* list with NaN for a miss, unreached list) through derive_iso."""
    import plot_cost_scaling as pcs
    document = _p4c_doc(system)
    points = [p for p in document["points"] if p.get("m_sweep")]
    m_star, *_, unreached = pcs.derive_iso(
        points, target, "single", "rmse", document["meta"]["params"]["N_ACC"])
    return [p["dim"] for p in points], list(m_star), unreached


def test_p4c_fixed_m_error_grows_on_the_chains_and_falls_on_the_oscillator(doc):
    """'At a fixed M the RMSE grows with dimension' was stated for all three
    systems; on System C it falls. Every endpoint is recomputed, and each
    series is checked to move the way the sentence says, at every size."""
    m = re.search(
        r"On the two chains it grows: at \$M=(\d+)\$, one run's energy RMSE rises "
        r"from ([\d.]+) at dim (\d+) to ([\d.]+) at dim (\d+) on System A, and "
        r"from ([\d.]+) at dim (\d+) to ([\d.]+) at dim (\d+) on System B\. On "
        r"System C it falls: at \$M=(\d+)\$ it drops from ([\d.]+) at dim (\d+) "
        r"to ([\d.]+) at dim (\d+)\. Each of these is one run's error, estimated "
        r"from (\d+) realizations\.", _flat(doc))
    assert m, "Result 2's fixed-M error sentence has changed shape"
    g = m.groups()
    chains = int(g[0])
    for system, (lo_v, lo_d, hi_v, hi_d), bundles, rising in (
            ("spin_chain", g[1:5], chains, True),
            ("mixed_chain", g[5:9], chains, True),
            ("oscillator_bath", g[10:14], int(g[9]), False)):
        series = _p4c_single(system, bundles)
        dims = sorted(series)
        assert (int(lo_d), int(hi_d)) == (dims[0], dims[-1]), (system, dims)
        assert _near(lo_v, series[dims[0]][0]), (system, series[dims[0]][0])
        assert _near(hi_v, series[dims[-1]][0]), (system, series[dims[-1]][0])
        values = [series[d][0] for d in dims]
        steps = np.diff(values)
        assert (steps > 0).all() if rising else (steps < 0).all(), (system, values)
        assert _p4c_doc(system)["meta"]["params"]["N_ACC"] == int(g[14])


def test_p4c_iso_definition_names_what_it_scores(doc):
    """The definition left out that the error is ONE run's, energy only, in
    absolute units, on a grid capped at N_L -- without the cap, System A's
    31 and 57 cannot come off a 1, 2, 4, 8 grid."""
    import plot_cost_scaling as pcs
    import run_cost_scaling as rcs
    m = re.search(
        r"takes the smallest bundle size \$M\^\\ast\$ at which \*\*one\*\* SLB run "
        r"meets a fixed error target, and plots the cost of that one run\. The "
        r"error is the time-averaged RMSE of the energy alone, in absolute energy "
        r"units, against the exact reference\. It is one run's error, "
        r"\$\\sqrt\{\\text\{bias\}\^2 \+ \\text\{Std\}\^2\}\$, estimated from (\d+) "
        r"realizations; the average of those (\d+) would be more accurate\. The swept "
        r"grid is \$M = 1, 2, 4, \\dots, (\d+)\$, capped at \$N_L\$ with \$N_L\$ "
        r"itself included, which is where System A's (\d+) and (\d+) come from\. "
        r"The grid doubles, so \$M\^\\ast\$ is known only to within a factor of 2\. "
        r"Result 4 scores differently: the ensemble average's error, on six "
        r"observables, at (\d+)% of each one's span\.", _flat(doc))
    assert m, "Result 2's iso-accuracy definition has changed shape"
    q_nacc, q_nacc2, q_top, q_a, q_b, q_pct = m.groups()
    assert q_nacc2 == q_nacc
    assert (pcs.ESTIMATE_TYPE, pcs.ERROR_TYPE) == ("single", "rmse")
    import inspect
    body = inspect.getsource(rcs.slb_estimate)
    assert "e_ops=[H]" in body and "samples[:, 0, :]" in body, "the sweep scores more than H"
    grid = rcs.M_SWEEP_GRID
    assert grid == [2 ** k for k in range(len(grid))] and grid[-1] == int(q_top)
    for system in ("spin_chain", "mixed_chain", "oscillator_bath"):
        document = _p4c_doc(system)
        assert document["meta"]["params"]["N_ACC"] == int(q_nacc)
        assert document["meta"]["params"]["M_SWEEP_GRID"] == grid
        for p in document["points"]:
            swept = [e["M"] for e in p.get("m_sweep") or []]
            assert all(s in grid or s == p["n_l"] for s in swept), (system, swept)
            assert all(s <= p["n_l"] for s in swept), (system, swept)
    dims, m_star, _ = _p4c_ladder("spin_chain", pcs.TARGET_BY_SYSTEM["spin_chain"])
    n_ls = {p["dim"]: p["n_l"] for p in _p4c_doc("spin_chain")["points"]}
    capped = [(d, int(v)) for d, v in zip(dims, m_star) if v == v and int(v) not in grid]
    assert [v for _, v in capped] == [int(q_a), int(q_b)]
    assert all(n_ls[d] == v for d, v in capped), "an off-grid M* that is not N_L"
    isocost = pytest.importorskip("plot_isocost_vs_dim")
    assert isocost.ESTIMATE_TYPE == "ensemble"
    assert _near(q_pct, 100 * isocost.TARGET_REL)


def test_p4c_target_table_reasons_hold_for_one_run(doc):
    """The A row's upper end (0.029) missed 0.033 at dim 256, and the C row's
    'M*=1 already clears 0.02 at every size' is true only of the 16-run
    average, which the figure no longer scores."""
    import plot_cost_scaling as pcs
    text = _flat(doc)
    for letter, system in (("A", "spin_chain"), ("B", "mixed_chain"),
                           ("C", "oscillator_bath")):
        row = re.search(rf"\|\s*\*\*{letter}\*\*[^|]*\|\s*\*\*([\d.]+)\*\*\s*\|", text)
        assert row and float(row.group(1)) == pcs.TARGET_BY_SYSTEM[system], letter

    a = re.search(r"At 0\.02 one run misses at every dimension past dim (\d+), "
                  r"because \$M\$ cannot exceed \$N_L\$ and even \$M=N_L\$ leaves "
                  r"one run at ([\d.]+) to ([\d.]+) \(dims (\d+) to (\d+)\)\.", text)
    assert a, "the A row of the target table has changed shape"
    dims, m_star, unreached = _p4c_ladder("spin_chain", 0.02)
    assert dims[0] == int(a.group(1)) and m_star[0] == m_star[0]
    assert all(v != v for v in m_star[1:]) and len(unreached) == len(dims) - 1
    document = _p4c_doc("spin_chain")
    at_cap = {}
    for p in document["points"]:
        top = next(e for e in p["m_sweep"] if e["M"] == p["n_l"])
        at_cap[p["dim"]] = float(pcs.get_metrics(
            top, document["meta"]["params"]["N_ACC"])["single"]["rmse"])
    window = [v for d, v in at_cap.items() if int(a.group(4)) <= d <= int(a.group(5))]
    assert (int(a.group(4)), int(a.group(5))) == (dims[1], dims[-1])
    assert _near(a.group(2), min(window)) and _near(a.group(3), max(window))

    c = re.search(r"At 0\.02 one run's \$M\^\\ast\$ is \$((?:\d+ \\to )+\d+)\$ \(dims "
                  r"(\d+) to (\d+)\)\. It reaches \$M=1\$ at dim (\d+) and can fall "
                  r"no further, so dims (\d+) and (\d+) look the same\. At 0\.005, "
                  r"\$M\^\\ast\$ stays above 1 wherever the target is met "
                  r"\(\$((?:\d+ \\to )+\d+)\$ over dims (\d+) to (\d+); dim (\d+) is "
                  r"missed, the hatched bar above\)\.", text)
    assert c, "the C row of the target table has changed shape"
    dims, m_star, _ = _p4c_ladder("oscillator_bath", 0.02)
    reached = [(d, int(v)) for d, v in zip(dims, m_star) if v == v]
    assert [int(v) for v in c.group(1).split(r" \to ")] == [v for _, v in reached]
    assert (int(c.group(2)), int(c.group(3))) == (reached[0][0], reached[-1][0])
    at_floor = [d for d, v in reached if v == 1]
    assert at_floor == [int(c.group(5)), int(c.group(6))] and at_floor[0] == int(c.group(4))
    tight_dims, tight, unreached = _p4c_ladder("oscillator_bath", 0.005)
    assert all(v > 1 for v in tight if v == v)
    met = [(d, int(v)) for d, v in zip(tight_dims, tight) if v == v]
    assert [int(v) for v in c.group(7).split(r" \to ")] == [v for _, v in met]
    assert (int(c.group(8)), int(c.group(9))) == (met[0][0], met[-1][0])
    assert [u[0] for u in unreached] == [int(c.group(10))], unreached
    assert all(u[2] > 1 for u in unreached), "a miss at M=1 would not be 'above 1'"


def test_p4c_system_a_mstar_and_the_matched_range_exponents(doc):
    """'M* tracks N_L almost exactly' held at 2 of 8 sizes, and 'N^2.7 against
    N^2.6' compared a 5-point fit (dims 32-512) with a 6-point one (16-512).
    Over the same dims both are N^2.7; the curves run parallel."""
    import plot_cost_scaling as pcs
    m = re.search(
        r"\*\*System A \(Control 1\):\*\* \$M\^\\ast\$ climbs with \$N_L\$ but "
        r"equals it only at dims (\d+) and (\d+): \$M\^\\ast/N_L = ([\d/, ]+)\$ "
        r"across dims (\d+) to (\d+)\. From dim (\d+) on, one run needs (\d+)% to "
        r"(\d+)% of the operators; since the grid doubles, the true minimum can sit "
        r"up to one grid step lower\. Over dims (\d+) to (\d+) \((\d+) points "
        r"each\), one SLB run at \$M\^\\ast\$ and the (\d+)-substep exact solve "
        r"both grow as \$N\^\{([\d.]+)\}\$ \(the legend's \$N\^\{([\d.]+)\}\$ for "
        r"the exact solve is a (\d+)-point fit that starts at dim (\d+)\)\. The two "
        r"curves run parallel rather than converging: over those sizes the exact "
        r"solve costs ([\d.]+) to ([\d.]+) times one SLB run at (\d+) substeps, and "
        r"about half that at matched substeps \(([\d.]+) to ([\d.]+) times, using "
        r"the ([\d.]+)x measured on System B above\)\.", _flat(doc))
    assert m, "Result 2's System A iso-accuracy bullet has changed shape"
    (q_eq1, q_eq2, q_list, q_lo, q_hi, q_from, q_pmin, q_pmax, q_wlo, q_whi,
     q_npts, q_ref, q_exp, q_leg, q_legn, q_legdim, q_rmin, q_rmax, q_slb,
     q_mmin, q_mmax, q_margin) = m.groups()

    document, dims, native, _, iso, m_star = _cost_curves("spin_chain")
    n_ls = [p["n_l"] for p in document["points"]]
    pairs = [tuple(int(x) for x in s.split("/")) for s in q_list.split(", ")]
    assert pairs == [(int(a), int(b)) for a, b in zip(m_star, n_ls)]
    assert (int(q_lo), int(q_hi)) == (int(dims[0]), int(dims[-1]))
    equal = [int(d) for d, a, b in zip(dims, m_star, n_ls) if a == b]
    assert equal == [int(q_eq1), int(q_eq2)]
    share = [a / b for d, a, b in zip(dims, m_star, n_ls) if d >= int(q_from)]
    assert int(q_from) == int(dims[1])
    assert _near(q_pmin, 100 * min(share)) and _near(q_pmax, 100 * max(share))

    meta = document["meta"]
    assert int(q_ref) == meta["params"]["NATIVE_REF_SUBSTEPS"]
    assert int(q_slb) == meta["substeps"]
    s_iso, n_iso, lo, hi = _fit(dims, iso)
    assert (lo, hi, n_iso) == (int(q_wlo), int(q_whi), int(q_npts))
    window = (dims >= lo) & (dims <= hi)
    s_same, n_same = pcs.fit_slope(dims[window], native[window])
    assert n_same == int(q_npts)
    assert _near(q_exp, s_iso) and _near(q_exp, s_same), (s_iso, s_same)
    s_leg, n_leg, leg_lo, _ = _fit(dims, native)
    assert _near(q_leg, s_leg) and (n_leg, leg_lo) == (int(q_legn), int(q_legdim))

    ratio = native[window] / iso[window]
    assert _near(q_rmin, ratio.min()) and _near(q_rmax, ratio.max())
    margins = [_substep_margin(d) for d in (64, 128)]
    assert all(_near(q_margin, g) for g in margins)
    for g in margins:
        assert _near(q_mmin, ratio.min() / g) and _near(q_mmax, ratio.max() / g)


def test_p4c_system_b_ladder_fit_and_role_names(doc):
    """The bullets used 'Generic' for System B; section 1's roles table calls
    it Control 2. The ladder's N^0.77 is a fit over its six sizes."""
    text = _flat(doc)
    roles = re.search(r"\| Role here \| \*\*(.+?)\*\* —[^|]*\| \*\*(.+?)\*\* —[^|]*\| "
                      r"\*\*(.+?)\*\* —", text)
    assert roles, "section 1's roles row has changed shape"
    for letter, role in zip("ABC", roles.groups()):
        assert f"- **System {letter} ({role}):** $M^\\ast$" in text, (letter, role)
    m = re.search(r"across dims 4 to 128 \(fitted over those (\d+) sizes, "
                  r"\$M\^\\ast \\sim N\^\{([\d.]+)\}\$\)", text)
    assert m, "System B's ladder fit has changed shape"
    dims, m_star, _ = _p4c_ladder("mixed_chain", 0.02)
    assert int(m.group(1)) == len(dims) and all(v == v for v in m_star)
    slope = np.polyfit(np.log(dims), np.log(m_star), 1)[0]
    assert _near(m.group(2), slope), slope


def test_p4c_system_c_ladder_falls_and_dim8_is_a_stopped_sweep(doc):
    """'Nearly flat because bias barely grows' read a falling M* as flat and
    credited the wrong term: at M=4 the spread falls about 13x, the bias 2x."""
    m = re.search(
        r"\*\*System C \(Demonstration\):\*\* \$M\^\\ast\$ falls with size, "
        r"\$((?:\d+ \\to )+\d+)\$ across dims (\d+) to (\d+)\. It falls because one "
        r"run's error at a fixed \$M\$ shrinks as the oscillator grows \(above\), "
        r"mostly through a smaller run-to-run spread: at \$M=(\d+)\$ the Std falls "
        r"from ([\d.]+) at dim (\d+) to ([\d.]+) at dim (\d+), while the bias moves "
        r"only from ([\d.]+) to ([\d.]+)\. Dim (\d+) is drawn as missed because its "
        r"sweep stopped at \$M=(\d+)\$: the (\d+)-run average was already below the "
        r"sweep's ([\d.]+) stopping floor, but one run there is at ([\d.]+), above "
        r"the target\. Larger \$M\$ was never tried there\.", _flat(doc))
    assert m, "Result 2's System C iso-accuracy bullet has changed shape"
    (q_ladder, q_lo, q_hi, q_m, q_s0, q_d0, q_s1, q_d1, q_b0, q_b1, q_miss,
     q_stop_m, q_nacc, q_floor, q_one) = m.groups()
    import plot_cost_scaling as pcs
    dims, m_star, unreached = _p4c_ladder(
        "oscillator_bath", pcs.TARGET_BY_SYSTEM["oscillator_bath"])
    reached = [(d, int(v)) for d, v in zip(dims, m_star) if v == v]
    assert [v for _, v in reached] == [int(v) for v in q_ladder.split(r" \to ")]
    assert (reached[0][0], reached[-1][0]) == (int(q_lo), int(q_hi))
    assert [v for _, v in reached] == sorted((v for _, v in reached), reverse=True)

    series = _p4c_single("oscillator_bath", int(q_m))
    first, last = series[int(q_d0)], series[int(q_d1)]
    assert (int(q_d0), int(q_d1)) == (min(series), max(series))
    assert _near(q_s0, first[2]) and _near(q_s1, last[2])
    assert _near(q_b0, first[1]) and _near(q_b1, last[1])
    assert first[2] / last[2] > first[1] / last[1], "the bias fell faster than the spread"

    assert [(u[0], u[2]) for u in unreached] == [(int(q_miss), int(q_stop_m))]
    document = _p4c_doc("oscillator_bath")
    params = document["meta"]["params"]
    assert int(q_nacc) == params["N_ACC"] and float(q_floor) == params["SWEEP_STOP_RMSE"]
    point = next(p for p in document["points"] if p["dim"] == int(q_miss))
    top = point["m_sweep"][-1]
    assert top["M"] == int(q_stop_m) and top["M"] < point["n_l"]
    assert top["rmse"] <= params["SWEEP_STOP_RMSE"], "the sweep did not stop on its floor"
    one = _p4c_single("oscillator_bath", int(q_stop_m))[int(q_miss)][0]
    assert _near(q_one, one) and one > pcs.TARGET_BY_SYSTEM["oscillator_bath"]


def test_p4c_cost_scaling_comment_drops_the_ensemble_era_reasons():
    """plot_cost_scaling.py's target comment repeated the two stale reasons:
    'M*=1 clears 0.02 at every size' and 'M = N_L ... 0.024 to 0.029'."""
    source = (BENCHMARKS / "plot_cost_scaling.py").read_text(encoding="utf-8")
    flat = re.sub(r"\s*\n\s*#\s*", " ", source)
    assert "M*=1 clears 0.02" not in flat
    assert "0.024 to 0.029" not in flat
    import plot_cost_scaling as pcs
    loose = re.search(r"one run's M\* is ((?:\d+ -> )+\d+) \(dims (\d+)-(\d+)\)", flat)
    assert loose, "the System C 0.02 ladder has left the comment"
    dims, m_star, _ = _p4c_ladder("oscillator_bath", 0.02)
    reached = [(d, int(v)) for d, v in zip(dims, m_star) if v == v]
    assert [int(v) for v in loose.group(1).split(" -> ")] == [v for _, v in reached]
    assert (int(loose.group(2)), int(loose.group(3))) == (reached[0][0], reached[-1][0])
    tight = re.search(r"\((\d+(?:, \d+)+) over dims (\d+)-(\d+); dim (\d+) is missed", flat)
    assert tight, "the System C 0.005 ladder has left the comment"
    dims, m_star, unreached = _p4c_ladder("oscillator_bath", 0.005)
    met = [(d, int(v)) for d, v in zip(dims, m_star) if v == v]
    assert [int(v) for v in tight.group(1).split(", ")] == [v for _, v in met]
    assert [u[0] for u in unreached] == [int(tight.group(4))]
    cap = re.search(r"\((\d\.\d+) to (\d\.\d+) across dims (\d+)-(\d+)\)", flat)
    assert cap, "the System A at-N_L range has left the comment"
    document = _p4c_doc("spin_chain")
    at_cap = {p["dim"]: float(pcs.get_metrics(
        next(e for e in p["m_sweep"] if e["M"] == p["n_l"]),
        document["meta"]["params"]["N_ACC"])["single"]["rmse"])
        for p in document["points"]}
    window = [v for d, v in at_cap.items() if int(cap.group(3)) <= d <= int(cap.group(4))]
    assert _near(cap.group(1), min(window)) and _near(cap.group(2), max(window))
    frac = re.search(r"M\*/N_L over dims (\d+)-(\d+) is ((?:\d+/\d+, )+\d+/\d+)", flat)
    assert frac, "the System A M*/N_L list has left the comment"
    dims, m_star, _ = _p4c_ladder("spin_chain", 0.05)
    n_l = {p["dim"]: p["n_l"] for p in document["points"]}
    assert frac.group(3).split(", ") == [f"{int(v)}/{n_l[d]}" for d, v in zip(dims, m_star)]


# --- part 4: Result 2's "Numerical Certification & Limits" -------------------
#
# The list said the two exact routes agree to 1e-10 everywhere (the oscillator
# agrees only to 1.1e-8, as its own footer says), quoted a 4e17 divergence no
# committed file records, called a 32-substep rerun of a 64-substep reference
# "a 64-substep check", and wrote bundle assembly as N^4 on every system when
# only System B's N_L grows as N^2. These pin the corrected sentences.

_p4d_systems = ("spin_chain", "mixed_chain", "oscillator_bath")
_P4D_SCI = r"\$([\d.]+)\\times 10\^\{(-?\d+)\}\$"
# every key run_cost_scaling.py writes into a point; none of them is a size
_P4D_POINT_KEYS = {"size", "dim", "n_l", "t_davies", "t_full", "t_slb_fixed",
                   "t_slb_fixed_repeats", "slb_unstable_at_substeps",
                   "reference", "reference_method", "t_native_ref",
                   "native_ref_selfcheck", "t_native_ref_repeats", "m_sweep"}


def _p4d_text(doc: str) -> str:
    """The list items are indented, which _flat keeps; fold that too."""
    return _flat_ws(_flat(doc))


def _p4d_load(system: str) -> dict:
    path = DATA / f"cost_scaling_{system}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def _p4d_n_l(system: str):
    """(dims, N_L) of every point in the Result 2 file, as float arrays."""
    points = _p4d_load(system)["points"]
    return (np.array([p["dim"] for p in points], dtype=float),
            np.array([p["n_l"] for p in points], dtype=float))


def test_result2_exact_routes_agreement_is_quoted_per_system(doc):
    """Native RK4 and mesolve agree to 3.4e-10 on the chains but only 1.1e-8
    on the oscillator, each over the dims, substeps and saved times its file
    records; and the native route takes over where the sweep stops calling
    mesolve."""
    text = _p4d_text(doc)
    m = re.search(
        r"\(\$O\(N_L N\^2\)\$ memory instead of `mesolve`'s \$O\(N_L N\^4\)\$, "
        r"§5\.2\)\. It supplies the certified exact reference past dim (\d+), "
        r"where the sweep stops calling `mesolve` \(`max_full_dim = (\d+)`\)\. "
        r"Where both ran, the two routes' energy curves agree to within "
        + _P4D_SCI + r" on the two chains \(dims (\d+) to (\d+), (\d+) substeps\) "
        r"and " + _P4D_SCI + r" on the oscillator \(dims (\d+) to (\d+), (\d+) "
        r"substeps\)\. Each value is the largest gap over the (\d+) saved times "
        r"between one deterministic solve per route\. The figure footers round "
        r"these to (\S+) on both chains and (\S+) on the oscillator\.", text)
    assert m, "Result 2's exact-route sentence has changed shape"
    (past, cap, c_man, c_exp, c_lo, c_hi, c_sub,
     o_man, o_exp, o_lo, o_hi, o_sub, n_times, c_footer, o_footer) = m.groups()
    # the memory model it cites is section 5.2's
    assert r"$N_L \times N^4 \times 16$" in doc

    for system in _p4d_systems:
        document = _p4d_load(system)
        assert document["meta"]["max_full_dim"] == int(cap) == int(past)
        assert document["meta"]["tlist"]["n"] == int(n_times)
        ran = [p["dim"] for p in document["points"] if p.get("t_full") is not None]
        assert max(ran) == int(cap), f"{system}: mesolve last ran at {max(ran)}"

    chains = [_p4d_load(s)["native_vs_mesolve"] for s in _p4d_systems[:2]]
    worst = max(max(v["max_devs"]) for v in chains)
    _assert_latex_rounds_to(worst, c_man, c_exp, "chains' native-vs-mesolve gap")
    for v in chains:
        assert (min(v["dims"]), max(v["dims"])) == (int(c_lo), int(c_hi))
        assert v["substeps"] == int(c_sub)
        # plot_cost_scaling.figure prints each file's worst gap with :.0e
        assert f"{max(v['max_devs']):.0e}" == c_footer

    osc = _p4d_load("oscillator_bath")["native_vs_mesolve"]
    _assert_latex_rounds_to(max(osc["max_devs"]), o_man, o_exp,
                            "oscillator's native-vs-mesolve gap")
    assert (min(osc["dims"]), max(osc["dims"])) == (int(o_lo), int(o_hi))
    assert osc["substeps"] == int(o_sub)
    assert f"{max(osc['max_devs']):.0e}" == o_footer
    # the oscillator's gap is the larger one, so no single number covers both
    assert max(osc["max_devs"]) > 10 * worst


def test_result2_oscillator_limit_quotes_only_what_its_file_records(doc):
    """Dim 256 records that the 32-substep SLB solve failed, with no size; the
    dim-128 reference runs at 64 substeps and was checked against a 32-substep
    rerun (downward), not by a 64-substep check."""
    text = _p4d_text(doc)
    m = re.search(
        r"Why System C stops at dim (\d+) .*?Holding a uniform (\d+) substeps "
        r"across all dimensions for slope comparability, the bundled solver goes "
        r"unstable at dim (\d+)\. The file records only that it failed there "
        r"\(`slb_unstable_at_substeps = (\d+)`\), not how large the state grew\. "
        r"The ceiling is set by explicit fixed-step RK4 stability, not by operator "
        r"count\. At dim (\d+) the exact reference runs at (\d+) substeps; a rerun "
        r"at (\d+) substeps moves its energy curve by at most " + _P4D_SCI +
        r", far inside the \$10\^\{(-?\d+)\}\$ tolerance\.", text)
    assert m, "Result 2's oscillator-limit item has changed shape"
    (stop, uniform, bad, unstable_at, ref_dim, ref_sub, rerun_sub,
     man, exp, tol_exp) = m.groups()
    document = _p4d_load("oscillator_bath")
    points = {p["dim"]: p for p in document["points"]}

    assert document["meta"]["substeps"] == int(uniform) == int(unstable_at)
    assert document["stiff_dim"] == int(bad) == max(points)
    failed = points[int(bad)]
    assert failed["slb_unstable_at_substeps"] == int(unstable_at)
    assert failed["t_slb_fixed"] is None
    # nothing in the point says how large the state grew
    assert set(failed) <= _P4D_POINT_KEYS, sorted(set(failed) - _P4D_POINT_KEYS)
    timed = [d for d, p in points.items() if p.get("t_slb_fixed") is not None]
    assert max(timed) == int(stop)

    ref = points[int(ref_dim)]
    assert ref["reference_method"] == f"native_rk4_substeps{ref_sub}"
    check = ref["native_ref_selfcheck"]
    assert check["direction"] == "down" and check["passed"] is True
    assert check["substeps_pair"] == [int(rerun_sub), int(ref_sub)]
    _assert_latex_rounds_to(check["max_abs_dev"], man, exp,
                            "dim-128 reference self-check")
    assert check["tol"] == 10.0 ** int(tol_exp)


def test_result2_bundle_assembly_growth_follows_n_l(doc):
    """O(M N_L N^2) is N^4 only where N_L grows as N^2 (System B). The N_L
    ranges and the exponents are fitted by plot_cost_scaling.fit_slope over
    every point of each Result 2 file."""
    text = _p4d_text(doc)
    m = re.search(
        r"bundle assembly costs \$O\(M N_L N\^2\)\$, an implementation overhead "
        r"rather than the \$O\(N\^3\)\$ propagation core\. How fast it grows "
        r"depends on how fast \$N_L\$ grows\. On System B, \$N_L\$ is about "
        r"\$N\^2/2\$ \(([\d,]+) at dim (\d+); fitted \$N\^\{([\d.]+)\}\$ "
        r"over dims (\d+) to (\d+), (\d+) sizes\), so assembly grows as "
        r"\$N\^\{([\d.]+)\}\$\. On System A, \$N_L\$ grows only from (\d+) to "
        r"(\d+) across dims (\d+) to (\d+) \(fitted \$N\^\{([\d.]+)\}\$, (\d+) "
        r"sizes\), and on System C from ([\d,]+) to ([\d,]+) across dims (\d+) "
        r"to (\d+) \(\$N\^\{([\d.]+)\}\$, (\d+) sizes\), so assembly grows as "
        r"about \$N\^\{([\d.]+)\}\$ and \$N\^\{([\d.]+)\}\$ there\. These are "
        r"least-squares summaries: \$N_L\$ bends downward on A and C, so at "
        r"their largest sizes it grows more slowly still\.", text)
    assert m, "Result 2's bundle-assembly item has changed shape"
    (b_top, b_dim, b_exp, b_lo, b_hi, b_n, b_asm,
     a_first, a_last, a_lo, a_hi, a_exp, a_n,
     c_first, c_last, c_lo, c_hi, c_exp, c_n, a_asm, c_asm) = m.groups()

    dims, n_l = _p4d_n_l("mixed_chain")
    assert (int(dims[-1]), n_l[-1]) == (int(b_dim), _printed(b_top))
    assert np.all(np.abs(n_l / dims ** 2 - 0.5) < 0.1), n_l / dims ** 2
    slope, n, lo, hi = _fit(dims, n_l)
    assert (n, lo, hi) == (int(b_n), int(b_lo), int(b_hi)) == (len(dims), dims[0], dims[-1])
    assert _near(b_exp, slope)
    assert _near(b_asm, _fit(dims, n_l * dims ** 2)[0])

    for first, last, lo_q, hi_q, exp_q, n_q, asm_q, system in (
            (a_first, a_last, a_lo, a_hi, a_exp, a_n, a_asm, "spin_chain"),
            (c_first, c_last, c_lo, c_hi, c_exp, c_n, c_asm, "oscillator_bath")):
        dims, n_l = _p4d_n_l(system)
        slope, n, lo, hi = _fit(dims, n_l)
        assert (int(dims[0]), int(dims[-1])) == (int(lo_q), int(hi_q)) == (lo, hi)
        assert (n_l[0], n_l[-1]) == (_printed(first), _printed(last))
        assert n == int(n_q) == len(dims)
        assert _near(exp_q, slope), (system, slope)
        assert _near(asm_q, _fit(dims, n_l * dims ** 2)[0]), system
        # "bends downward": every local slope is below the one before it, and
        # the last is below the fitted one
        local = np.diff(np.log(n_l)) / np.diff(np.log(dims))
        assert np.all(np.diff(local) < 0) and local[-1] < slope, (system, local)


# --- part 6 U1: Result 3's solver list, curve thinning, error axis, M=1 ------
# The list said native RK4 was the reference only "past mesolve limits"; the
# accuracy paragraph said one point per bundle size from M=2 and called the
# error a plain "deviation"; the M=1 paragraph said the repeats agreed to under a
# millisecond and that the error is monotone in M at every dimension. Each claim
# is now recomputed from the files and from plot_method_comparison itself.

_P6A_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_P6A_SYSTEMS = {"A": "spin_chain", "B": "mixed_chain", "C": "oscillator_bath"}


def _p6a_region(doc: str) -> str:
    """Result 3 from its heading to the filled-or-hollow subsection, flattened."""
    text = _flat(doc)
    start = text.index("### Result 3 — accuracy versus cost: SLB against mcsolve")
    return text[start:text.index("#### Filled or hollow: which knob to turn", start)]


def _p6a_files() -> dict:
    """{(system, dim): document} for every committed Result 3 file."""
    out = {}
    for path in sorted(DATA.glob("method_comparison_*_dim*.json")):
        system, dim = re.match(r"method_comparison_(\w+?)_dim(\d+)$", path.stem).groups()
        out[(system, int(dim))] = json.loads(path.read_text(encoding="utf-8"))
    assert out, "no Result 3 files found"
    return out


def _p6a_curves(min_m: int) -> dict:
    """{(system, dim, observable): SLB rows from method_errors, sorted by M}.

    zz_per_bond is zz divided by a constant, so it is not an independent curve
    and is left out of every tally."""
    import plot_method_comparison as pmc
    saved = pmc._MIN_M
    pmc._MIN_M = min_m
    try:
        out = {}
        for (system, dim), document in _p6a_files().items():
            point = document["point"]
            if not point["methods"].get("slb"):
                continue
            for obs in point["observables"]:
                if obs == "zz_per_bond":
                    continue
                rows = [r for r in pmc.method_errors(point, obs) if r[0] == "slb"]
                out[(system, dim, obs)] = sorted(rows, key=pmc._m_of)
        return out
    finally:
        pmc._MIN_M = saved


def _p6a_figures(doc: str) -> list:
    """(system, observable) of every Result 3 figure the document embeds."""
    found = re.findall(
        r"\]\(benchmark_comparison_(spin_chain|mixed_chain|oscillator_bath)_(\w+)\.png\)",
        doc)
    for system, obs in found:
        png = BENCHMARKS / f"benchmark_comparison_{system}_{obs}.png"
        assert png.exists(), f"{png.name} is embedded but missing"
    return found


def test_p6a_result3_native_runs_as_reference_and_as_timed_baseline(doc):
    """Native RK4 runs at 2x SLB's substeps as the certified reference and at
    SLB's own substeps as the timed baseline; the stored references, and the one
    dimension quoted from §5.2's 8-substep grid, are the ones the text names."""
    import inspect
    import run_method_comparison as rmc

    text = _p6a_region(doc)
    m = re.search(
        r"It plays two roles at every dimension, not only where `mesolve` stops\. At twice "
        r"SLB's substeps it is the certified reference that every error is scored against "
        r"\((\d+) substeps on Systems A and B, (\d+) on System C, (\d+) at C's dimension "
        r"(\d+)\)\. At SLB's own substeps \((\d+) on A and B, (\d+) on C, (\d+) at C's "
        r"dimension (\d+)\) it is the timed baseline behind the `SLB speed vs native` "
        r"column, so that ratio compares equal step counts\. Two exceptions, both on "
        r"System A: at dimensions (\d+), (\d+) and (\d+) the reference is a stored certified "
        r"run at the same (\d+) substeps, reused rather than rerun in the job; and at (\d+) "
        r"there is no timed run at SLB's substeps, so the text quotes §5\.2's exact solve at "
        r"(\d+) substeps instead\.",
        text)
    assert m, "Result 3's native RK4 list item has changed shape"
    (ref_ab, ref_c, ref_hi, dim_hi, sub_ab, sub_c, sub_hi, dim_hi2,
     r1, r2, r3, stored_sub, no_timed, grid_sub) = map(int, m.groups())
    assert dim_hi == dim_hi2
    assert "| SLB speed vs native |" in doc

    # The timed native run uses SLB's substeps; the reference uses ref_substeps,
    # which defaults to twice that.
    source = inspect.getsource(rmc.run)
    assert re.search(r"run_native\([^)]*args\.slb_substeps\)", source), \
        "the timed native run no longer uses SLB's substeps"
    assert re.search(r"certified_reference\(\s*H, rho0, c_ops, ref_substeps\)", source)
    assert re.search(r"ref_substeps = args\.ref_substeps or 2 \* args\.slb_substeps", source)

    files = _p6a_files()
    for (system, dim), document in files.items():
        meta, point = document["meta"], document["point"]
        substeps, ref_sub = meta["substeps"], meta["params"]["ref_substeps"]
        assert point["reference"]["method"] == f"native_rk4_substeps{ref_sub}", (system, dim)
        assert point["reference"]["selfcheck"]["passed"], (system, dim)
        assert ref_sub == 2 * substeps, f"{system} dim {dim}: reference is not at 2x SLB's substeps"
        if system == "oscillator_bath":
            want = (ref_hi, sub_hi) if dim == dim_hi else (ref_c, sub_c)
        else:
            want = (ref_ab, sub_ab)
        assert (ref_sub, substeps) == want, f"{system} dim {dim}: {(ref_sub, substeps)} != {want}"
        if point["methods"].get("slb"):
            assert "wall_s" in point["methods"].get("native", {}), \
                f"{system} dim {dim}: SLB ran with no timed native baseline"

    reused = sorted(k for k, d in files.items()
                    if d["point"]["reference"].get("reused_from_archive"))
    assert reused == [("spin_chain", r1), ("spin_chain", r2), ("spin_chain", r3)], reused
    for _, dim in reused:
        archive = json.loads((DATA / f"high_dim_reference_spin_chain_dim{dim}.json")
                             .read_text(encoding="utf-8"))
        assert archive["meta"]["substeps"] == stored_sub == ref_ab, dim
        assert archive["point"]["selfcheck"]["passed"], dim
    untimed = sorted(k for k, d in files.items()
                     if "wall_s" not in d["point"]["methods"].get("native", {}))
    assert untimed == [("spin_chain", no_timed)], untimed
    grid = json.loads((DATA / "solver_timing_spin_chain.json").read_text(encoding="utf-8"))
    row = [p for p in grid["points"] if p["dim"] == no_timed]
    assert len(row) == 1 and "median_s" in row[0]["timings"]["native"]
    assert row[0]["native_substeps"] == grid_sub == 2 * row[0]["slb_substeps"]


def test_p6a_result3_curve_thinning_matches_the_plotter(doc):
    """At most MAX_CURVE_POINTS per curve, ending at the largest M or at the
    first M from which every point is hollow; the start-M tally over the
    figures the document embeds."""
    import inspect
    import plot_method_comparison as pmc

    text = _p6a_region(doc)
    m = re.search(
        r"A curve does not show every bundle size from \$M=(\d+)\$ up; it shows at most "
        r"(\w+)\. It ends at the largest \$M\$ run, or sooner, at the first \$M\$ where that "
        r"point and every larger one are hollow \(mostly sampling noise; see below\), and "
        r"keeps up to (\w+) sizes before that end\. So of the (\d+) SLB curves in the (\w+) "
        r"figures below, (\d+) start at \$M=(\d+)\$, (\d+) at \$M=(\d+)\$ and (\d+) at "
        r"\$M=(\d+)\$ or later, and (\w+) are a single point\. Pass `--full-curves` to draw "
        r"every \$M\$\.", text)
    assert m, "Result 3's curve-thinning sentence has changed shape"
    m_min, most, before, total, n_fig, na, ma, nb, mb, nc, mc, n_single = m.groups()
    assert _P6A_WORDS[most] == pmc.MAX_CURVE_POINTS
    assert _P6A_WORDS[before] == pmc.MAX_CURVE_POINTS - 1
    assert int(m_min) == pmc.MIN_M_PLOTTED
    assert "--full-curves" in inspect.getsource(pmc.main)

    figures = set(_p6a_figures(doc))
    assert len(figures) == _P6A_WORDS[n_fig], sorted(figures)

    saved = pmc._MAX_POINTS
    pmc._MAX_POINTS = pmc.MAX_CURVE_POINTS
    try:
        starts, sizes = [], []
        for (system, dim, obs), rows in _p6a_curves(pmc.MIN_M_PLOTTED).items():
            # The rule as the prose states it, computed independently.
            cut = len(rows)
            for i in range(len(rows)):
                if not any(pmc._bias_limited(r[2], r[5]) for r in rows[i:]):
                    cut = i + 1
                    break
            want = rows[:cut][-pmc.MAX_CURVE_POINTS:]
            window = pmc._curve_window(rows)
            assert window == want, (system, dim, obs)
            if (system, obs) in figures:
                starts.append(pmc._m_of(window[0]))
                sizes.append(len(window))
    finally:
        pmc._MAX_POINTS = saved
    assert len(starts) == int(total)
    assert starts.count(int(ma)) == int(na)
    assert starts.count(int(mb)) == int(nb)
    assert sum(s >= int(mc) for s in starts) == int(nc)
    assert int(na) + int(nb) + int(nc) == int(total)
    assert sizes.count(1) == _P6A_WORDS[n_single]


def test_p6a_result3_error_axis_is_absolute_bias_sem_average(doc):
    """The plotted error is mean_t sqrt(bias^2 + sem^2) in the observable's own
    units, for SLB (16 realizations) and mcsolve (500 trajectories) alike."""
    import plot_method_comparison as pmc

    text = _p6a_region(doc)
    m = re.search(
        r"\*\*What the error axis measures\.\*\* At each time, the error is "
        r"\$\\sqrt\{\\text\{bias\}\^2 \+ \\text\{s\.e\.m\.\}\^2\}\$ against the certified "
        r"reference, and the plotted value is its average over the (\d+) time points\. Both "
        r"methods use this same formula\. SLB's s\.e\.m\. comes from its (\d+) realizations, "
        r"`mcsolve`'s from its (\d+) trajectories\. The error is in the observable's own units, "
        r"not a fraction of its size\.", text)
    assert m, "Result 3's error-axis paragraph has changed shape"
    n_t, n_runs, ntraj = map(int, m.groups())
    budget = re.search(r"`mcsolve` is a single fixed-budget point at "
                       r"\$N_\{\\text\{traj\}\} = (\d+)\$", text)
    assert budget, "Result 3's mcsolve-budget sentence has changed shape"
    assert int(budget.group(1)) == ntraj, "the two quotes of mcsolve's budget disagree"

    checked = 0
    for (system, dim), document in _p6a_files().items():
        point = document["point"]
        assert document["meta"]["tlist"]["n"] == n_t, (system, dim)
        mc = point["methods"].get("mcsolve")
        slb = point["methods"].get("slb", [])
        for row in slb:
            assert int(row["n_runs"]) == n_runs, (system, dim, row["M"])
        if mc and "skipped" not in mc:
            assert int(mc["ntraj"]) == ntraj, (system, dim)
        if not (slb and mc):
            continue
        for obs_index, obs in enumerate(point["observables"]):
            reference = pmc.mean_curve(point["reference"]["curves"][obs])
            assert len(reference) == n_t
            rows = pmc.method_errors(point, obs)
            # mcsolve, recomputed by hand: no normalisation anywhere.
            curve = pmc.mean_curve(mc["curves"][obs])
            sem = np.asarray(mc["traj_std"][obs], dtype=float) / np.sqrt(ntraj)
            want = float(np.mean(np.sqrt((curve - reference) ** 2 + sem ** 2)))
            got = [r for r in rows if r[0] == "mcsolve"][0][2]
            assert math.isclose(got, want, rel_tol=1e-12), (system, dim, obs, "mcsolve")
            # SLB at every bundle size the figures can draw.
            for row in slb:
                if int(row["M"]) < pmc.MIN_M_PLOTTED:
                    continue
                samples = np.asarray(row["samples"], dtype=float)[:, obs_index, :]
                bias = samples.mean(axis=0) - reference
                s = samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])
                want = float(np.mean(np.sqrt(bias ** 2 + s ** 2)))
                got = [r for r in rows if r[0] == "slb" and r[3] == f"M={row['M']}"][0][2]
                assert math.isclose(got, want, rel_tol=1e-12), (system, dim, obs, row["M"])
                checked += 1
    assert checked, "no file carries both SLB and mcsolve"


def test_p6a_result3_m1_paragraph_matches_the_data(doc):
    """Where M=1 timed slower than M=2, the dim-32 medians and repeat spreads,
    the M=1 -> 2 tally and its exceptions, the rises past M=2, and the largest
    rise in units of the s.e.m. of the point it reaches."""
    import inspect
    import plot_method_comparison as pmc

    text = _p6a_region(doc)
    assert "--include-m1" in inspect.getsource(pmc.main)
    files = _p6a_files()

    m = re.search(
        r"At (\w+) dimensions \(System A at (\d+), System B at (\d+), System C at (\d+)\) it "
        r"also timed \*slower\* than \$M=2\$ in the same job at the same substeps, despite "
        r"doing strictly less arithmetic; at the other (\d+) it was faster\.", text)
    assert m, "Result 3's M=1 timing sentence has changed shape"
    n_slow, da, db, dc, n_fast = m.groups()
    slower, faster = [], 0
    for (system, dim), document in files.items():
        slb = {int(r["M"]): r for r in document["point"]["methods"].get("slb", [])}
        if not slb:
            continue
        assert 1 in slb and 2 in slb, (system, dim)
        if slb[1]["wall_s"] > slb[2]["wall_s"]:
            slower.append((system, dim))
        else:
            faster += 1
    assert sorted(slower) == sorted([(_P6A_SYSTEMS["A"], int(da)), (_P6A_SYSTEMS["B"], int(db)),
                                     (_P6A_SYSTEMS["C"], int(dc))]), slower
    assert len(slower) == _P6A_WORDS[n_slow]
    assert faster == int(n_fast)

    m = re.search(
        r"On System A at dimension (\d+) it took ([\d.]+) s against ([\d.]+) s, each the "
        r"median of (\w+) repeats; the \$M=1\$ repeats spread by under (\d+) ms, the \$M=2\$ "
        r"repeats by (\d+) ms\.", text)
    assert m, "Result 3's M=1 repeats sentence has changed shape"
    dim, t1, t2, n_rep, under1, spread2 = m.groups()
    slb = {int(r["M"]): r for r in files[("spin_chain", int(dim))]["point"]["methods"]["slb"]}
    reps1, reps2 = slb[1]["wall_s_repeats"], slb[2]["wall_s_repeats"]
    assert len(reps1) == len(reps2) == _P6A_WORDS[n_rep]
    assert _near(t1, float(np.median(reps1))) and _near(t1, slb[1]["wall_s"]), t1
    assert _near(t2, float(np.median(reps2))) and _near(t2, slb[2]["wall_s"]), t2
    # The tightest whole-millisecond bound: "under 2 ms" would also be true.
    assert int(under1) - 1 <= 1000 * (max(reps1) - min(reps1)) < int(under1)
    assert _near(spread2, 1000 * (max(reps2) - min(reps2)))

    m = re.search(
        r"Across all (\d+) SLB curves in the files \(every distinct observable, not only "
        r"the (\w+) drawn per system\), going from \$M=1\$ to \$M=2\$ lowers "
        r"the error on (\d+)\. The (\w+) exceptions are (\w+) at System A dimension (\d+) and "
        r"System C dimension (\d+)\. Past \$M=2\$ the error does not fall at every step "
        r"either: (\d+) of the (\d+) curves rise at least once\. Every rise, the (\w+) from "
        r"\$M=1\$ included, is smaller than the s\.e\.m\. of the point it rises to \(([\d.]+) "
        r"of it at most\)\.", text)
    assert m, "Result 3's monotonicity sentences have changed shape"
    (total, n_drawn, n_lower, n_exc, obs, ea, ec, n_rise, total2, n_exc2,
     q_max) = m.groups()
    drawn = {}
    for system, fig_obs in _p6a_figures(doc):
        drawn.setdefault(system, set()).add(fig_obs)
    assert all(len(v) == _P6A_WORDS[n_drawn] for v in drawn.values()), drawn

    curves = _p6a_curves(1)
    assert len(curves) == int(total) == int(total2)
    exceptions = []
    for key, rows in curves.items():
        assert [pmc._m_of(r) for r in rows[:2]] == [1, 2], key
        if not rows[1][2] < rows[0][2]:
            exceptions.append(key)
    assert len(curves) - len(exceptions) == int(n_lower)
    assert len(exceptions) == _P6A_WORDS[n_exc] == _P6A_WORDS[n_exc2]
    assert sorted(exceptions) == sorted([("spin_chain", int(ea), obs),
                                         ("oscillator_bath", int(ec), obs)]), exceptions

    rising_past_2, ratios = set(), []
    for key, rows in curves.items():
        for i, (a, b) in enumerate(zip(rows, rows[1:])):
            if b[2] >= a[2]:
                ratios.append((b[2] - a[2]) / b[5])
                if i >= 1:
                    rising_past_2.add(key)
    assert len(rising_past_2) == int(n_rise)
    assert max(ratios) < 1, "a rise now exceeds the s.e.m. of the point it reaches"
    _assert_rounds_to(max(ratios), q_max, "largest rise in s.e.m. units")


def test_p6a_result3_slb_sweep_starts_at_m1(doc):
    """The SLB list item: the sweep runs from M=1 (every file has it), the top
    bundle size reached, the figures' first M and the realization count."""
    import plot_method_comparison as pmc

    m = re.search(
        r"4\. \*\*SLB:\*\* Stochastically bundled dissipators, \$M\$ swept from (\d+) up to "
        r"(\d+) where the sweep reached it \(the figures start at \$M=(\d+)\$\), (\d+) "
        r"realizations per point\.", _p6a_region(doc))
    assert m, "Result 3's SLB list item has changed shape"
    low, high, first, n_runs = map(int, m.groups())
    sweeps = [[int(r["M"]) for r in d["point"]["methods"]["slb"]]
              for d in _p6a_files().values() if d["point"]["methods"].get("slb")]
    assert sweeps
    assert all(min(s) == low for s in sweeps)
    assert max(max(s) for s in sweeps) == high
    assert first == pmc.MIN_M_PLOTTED
    assert all(int(r["n_runs"]) == n_runs for d in _p6a_files().values()
               for r in d["point"]["methods"].get("slb", []))


# --- Result 3, "Filled or hollow": which knob to turn -----------------------
# Every count, ratio and fitted crossover in that subsection is recomputed
# through plot_method_comparison (method_errors, _bias_limited), the functions
# that fill or hollow each marker on the figures.

def _p6b_point(system: str, dim: int) -> dict:
    path = DATA / f"method_comparison_{system}_dim{dim}.json"
    assert path.exists(), f"{path.name} is quoted in Result 3 but not committed"
    return json.loads(path.read_text(encoding="utf-8"))


def _p6b_slb(point: dict, observable: str) -> dict:
    """{M: (error/s.e.m., error, wall_s, s.e.m.)} for SLB, via method_errors."""
    import plot_method_comparison as pmc
    return {pmc._m_of(r): (r[2] / r[5], r[2], r[1], r[5])
            for r in pmc.method_errors(point, observable) if r[0] == "slb"}


def _p6b_crossover(ratios: dict, lo: int, hi: float = math.inf) -> float:
    """Where a log-log line through error/s.e.m. against M, fitted over
    lo <= M <= hi, meets sqrt(2)."""
    ms = sorted(m for m in ratios if lo <= m <= hi)
    fit = linregress(np.log(ms), np.log([ratios[m][0] for m in ms]))
    return float(np.exp((np.log(math.sqrt(2)) - fit.intercept) / fit.slope))


def _p6b_region(doc: str) -> str:
    start = doc.index("#### Filled or hollow: which knob to turn")
    end = doc.index("**Where `mcsolve` sits against its own noise", start)
    return doc[start:end]


def test_p6b_filled_markers_are_counted_per_observable(doc):
    """'Nearly every SLB point is filled' holds for the energy only. The
    counts over every method_comparison file (M >= 2, zz_per_bond left out as
    zz rescaled), System B's share, and which observables go hollow at B dim
    64 are all recomputed through _bias_limited."""
    import plot_method_comparison as pmc

    text = _flat(_p6b_region(doc))
    m = re.search(
        r"\*\*On the energy, nearly every SLB point is filled\*\*: (\d+) of (\d+), "
        r"over all three systems at every \$M \\ge (\d+)\$\. .*?Across all "
        r"distinct observables, (\d+) of (\d+) points are filled, about two in three\. "
        r"On System B only about half are \((\d+) of (\d+)\)\. At dimension 64 there, "
        r"`coherence` is hollow from \$M=(\d+)\$ up, `sx` from \$M=(\d+)\$ to "
        r"(\d+), `zz` from \$M=(\d+)\$ and `sz` from \$M=(\d+)\$\.", text)
    assert m, "Result 3's filled/hollow count paragraph has changed shape"
    (e_f, e_n, min_m, a_f, a_n, b_f, b_n,
     coh_lo, sx_lo, sx_hi, zz_lo, sz_lo) = map(int, m.groups())
    assert min_m == pmc.MIN_M_PLOTTED

    counts = {}
    for system in pmc.SYSTEMS:
        for dim in pmc.discover_dims(system):
            point = _p6b_point(system, dim)["point"]
            for obs in point["observables"]:
                if obs == "zz_per_bond":
                    continue
                for _, error, _, sem in _p6b_slb(point, obs).values():
                    key = (system, obs)
                    filled, total = counts.get(key, (0, 0))
                    counts[key] = (filled + pmc._bias_limited(error, sem),
                                   total + 1)

    def tally(pred):
        return tuple(sum(v[i] for k, v in counts.items() if pred(k)) for i in (0, 1))

    assert (e_f, e_n) == tally(lambda k: k[1] == "energy")
    assert (a_f, a_n) == tally(lambda k: True)
    assert (b_f, b_n) == tally(lambda k: k[0] == "mixed_chain")
    assert e_f / e_n > 0.9, "'nearly every' energy point must be filled"
    assert 0.6 < a_f / a_n < 0.7, "'about two in three'"
    assert 0.45 < b_f / b_n < 0.55, "'about half' on System B"

    point = _p6b_point("mixed_chain", 64)["point"]
    hollow = {obs: {mm for mm, row in _p6b_slb(point, obs).items()
                    if not pmc._bias_limited(row[1], row[3])}
              for obs in ("coherence", "sx", "zz", "sz")}
    ms = sorted(_p6b_slb(point, "energy"))
    assert {coh_lo, sx_lo, sx_hi, zz_lo, sz_lo} <= set(ms), "a quoted bound is not a swept M"
    assert (min(hollow["coherence"]), max(hollow["coherence"])) == (coh_lo, max(ms))
    assert (min(hollow["sx"]), max(hollow["sx"])) == (sx_lo, sx_hi)
    assert min(hollow["zz"]) == zz_lo and min(hollow["sz"]) == sz_lo
    assert hollow["coherence"] == {mm for mm in ms if mm >= coh_lo}
    assert hollow["sx"] == {mm for mm in ms if sx_lo <= mm <= sx_hi}
    assert hollow["zz"] == {mm for mm in ms if mm >= zz_lo}
    assert hollow["sz"] == {mm for mm in ms if mm >= sz_lo}


def test_p6b_sixteen_realizations_buy_little_on_a_bias_limited_energy(doc):
    """B dim 64 energy: error/s.e.m. from M=2 to M=256, and how much 16
    realizations lower the error against one realization (common.tavg_rmse
    with n_eff=1 keeps the bias and uses one realization's spread), against
    the sqrt(16) = 4x that pure noise would give."""
    text = _flat(_p6b_region(doc))
    m = re.search(
        r"The energy ratio at that size runs from ([\d.]+) at \$M=(\d+)\$ down to "
        r"([\d.]+) at \$M=(\d+)\$ — bias-limited throughout\. So averaging (\d+) "
        r"realizations instead of 1 lowers the energy error only ([\d.]+)x at "
        r"\$M=(\d+)\$ and ([\d.]+)x at \$M=(\d+)\$; even at \$M=(\d+)\$ it is "
        r"([\d.]+)x, short of the (\d+)x that pure noise would give\.", text)
    assert m, "Result 3's 16-versus-1 realization sentence has changed shape"
    (hi, m_hi, lo, m_lo, n_runs, g1, m1, g2, m2, m3, g3, pure) = m.groups()

    import plot_method_comparison as pmc

    point = _p6b_point("mixed_chain", 64)["point"]
    ratios = _p6b_slb(point, "energy")
    assert m_hi == str(min(ratios)) and m_lo == str(max(ratios))
    assert _near(hi, ratios[int(m_hi)][0]) and _near(lo, ratios[int(m_lo)][0])
    assert all(pmc._bias_limited(row[1], row[3]) for row in ratios.values()), (
        "'bias-limited throughout'")

    index = point["observables"].index("energy")
    reference = pmc.mean_curve(point["reference"]["curves"]["energy"])
    rows = {int(r["M"]): r for r in point["methods"]["slb"]}
    for printed, bundles in ((g1, m1), (g2, m2), (g3, m3)):
        row = rows[int(bundles)]
        samples = np.asarray(row["samples"], dtype=float)[:, index, :]
        assert samples.shape[0] == int(n_runs) == int(row["n_runs"])
        gain = (common.tavg_rmse(samples, reference, n_eff=1)
                / common.tavg_rmse(samples, reference))
        assert _near(printed, gain), f"M={bundles}: gain {gain:.4f} vs {printed}"
    assert int(pure) == round(math.sqrt(int(n_runs)))


def test_p6b_hollow_point_still_gains_from_a_larger_bundle(doc):
    """B dim 128 coherence: hollow at M=16, yet M=256 cut the error by more
    than sqrt(the wall-clock ratio), which is the most the same wall-clock
    spent on more realizations could buy -- because each realization's spread
    also shrinks with M. Both walls are one job at one substep count."""
    import plot_method_comparison as pmc

    text = _flat(_p6b_region(doc))
    m = re.search(
        r"System B's `coherence` at dimension (\d+) is hollow at \$M=(\d+)\$ "
        r"\(error/s\.e\.m\. ([\d.]+)\)\. Going to \$M=(\d+)\$ cut its error "
        r"([\d.]+)x for ([\d.]+)x the \$M=(\d+)\$ wall-clock \(both with (\d+) "
        r"realizations run in series, in one job, at the same (\d+) substeps\)\. "
        r"Spending that "
        r"([\d.]+)x on more realizations at \$M=(\d+)\$ instead would have cut the "
        r"error at most ([\d.]+)x \(\$\\sqrt\{([\d.]+)\}\$\)", text)
    assert m, "Result 3's hollow-point example has changed shape"
    (dim, m_a, ratio, m_b, cut, cost, m_base, n_runs, substeps, cost_again,
     m_base_again, noise_cut, under_root) = m.groups()
    assert m_base == m_base_again == m_a and cost_again == under_root == cost

    document = _p6b_point("mixed_chain", int(dim))
    point = document["point"]
    slb = _p6b_slb(point, "coherence")
    a, b = slb[int(m_a)], slb[int(m_b)]
    assert not pmc._bias_limited(a[1], a[3]), "the M=16 point must be hollow"
    assert _near(ratio, a[0])
    assert _near(cut, a[1] / b[1])
    assert _near(cost, b[2] / a[2])
    assert _near(noise_cut, math.sqrt(b[2] / a[2]))
    assert a[1] / b[1] > math.sqrt(b[2] / a[2]), "a larger M must beat more samples here"
    assert int(substeps) == document["meta"]["substeps"]
    rows = {int(r["M"]): r for r in point["methods"]["slb"]}
    assert int(rows[int(m_a)]["n_runs"]) == int(rows[int(m_b)]["n_runs"]) == int(n_runs)
    index = point["observables"].index("coherence")
    spread = {mm: float(np.mean(np.asarray(rows[mm]["samples"], dtype=float)
                                [:, index, :].std(axis=0, ddof=1)))
              for mm in (int(m_a), int(m_b))}
    assert spread[int(m_b)] < spread[int(m_a)], "one realization's spread falls with M"


def test_p6b_crossover_table_and_its_fit(doc):
    """The crossover table: energy error/s.e.m. at each dimension's largest M,
    the fitted crossover (log-log line over M >= 4, solved for sqrt(2)), N_L,
    the status, and the job; the prose's pre-extension prediction (fit over
    M=2..32 at dim 64), the dim-16 fit swing, and the fitted values repeated
    in the limits paragraph."""
    import plot_method_comparison as pmc

    region = _p6b_region(doc)
    start = region.index("| dim | energy error/s.e.m. at the largest `M` | "
                         "fitted crossover `M` | `N_L` | status |")
    block = region[start:region.index("\n\n", start)]
    rows = re.findall(
        r"^\| \**(\d+)\** \| \**([\d.]+)\** \(at `?M=(\d+)(=N_L)?`?\) \| "
        r"(~ (\d+)|none: stops falling) \| ([\d,]+) \| (.+?) \|$", block, re.M)
    assert len(rows) == 4, "Result 3's crossover table has changed shape"

    text = _flat(region)
    job = re.search(r"sweep to \$M=256\$ \(job (\d{8})\) pushed one energy curve", text)
    assert job, "the crossover paragraph no longer names its job"
    window = re.search(r"The fitted crossover is where a straight line through "
                       r"log\(error/s\.e\.m\.\) against log \$M\$, fitted over "
                       r"\$M \\ge (\d+)\$, meets \$\\sqrt\{2\}\$", text)
    assert window, "the crossover fit is no longer defined next to its table"
    lo = int(window.group(1))
    fits = {}
    for dim, ratio, largest, at_n_l, fit_cell, fit, n_l, status in rows:
        document = _p6b_point("mixed_chain", int(dim))
        point = document["point"]
        assert str(pmc.execution_key(document)[1]) == job.group(1)
        slb = _p6b_slb(point, "energy")
        assert int(largest) == max(slb)
        assert _near(ratio, slb[int(largest)][0])
        assert int(n_l.replace(",", "")) == int(point["n_l"])
        crossed = not pmc._bias_limited(slb[int(largest)][1], slb[int(largest)][3])
        assert ("crossed" in status) == crossed
        if at_n_l:
            assert int(largest) == int(point["n_l"]) and "beyond" in status
            assert fit_cell.startswith("none")
        else:
            fits[int(dim)] = _p6b_crossover(slb, lo)
            assert _near(fit, fits[int(dim)]), f"dim {dim}: fit {fits[int(dim)]:.1f}"
    assert [r[0] for r in rows if "crossed" in r[7]] == ["32"], (
        "'Of the four sizes, it is the only one that crosses'")

    m = re.search(
        r"So the dimension-(\d+) curve ends on a hollow marker: at \$M=(\d+)\$ there, "
        r"bias no longer dominates the energy error, so more realizations now help too, "
        r"not only a larger bundle\. "
        r"Of the four sizes, it is the only one that crosses\. At dimension 64 the "
        r"ratio is still ([\d.]+) at \$M=(\d+)\$\. A fit over the (\w+) points "
        r"\$M=(\d+)\$ to (\d+), made before the sweep was extended, "
        r"predicted the crossover there at \$M\\approx(\d+)\$\.", text)
    assert m, "the dim-64 prediction sentence has changed shape"
    b64 = _p6b_slb(_p6b_point("mixed_chain", 64)["point"], "energy")
    hollow_dim, hollow_m, still, at, count, first, last, predicted = m.groups()
    # "the dimension-32 curve ends on a hollow marker": the one crossed row,
    # at its largest M, and that point really is hollow.
    assert [r[0] for r in rows if "crossed" in r[7]] == [hollow_dim]
    b_hollow = _p6b_slb(_p6b_point("mixed_chain", int(hollow_dim))["point"], "energy")
    assert int(hollow_m) == max(b_hollow)
    assert not pmc._bias_limited(b_hollow[int(hollow_m)][1], b_hollow[int(hollow_m)][3])
    assert _near(still, b64[int(at)][0]) and int(at) == max(b64)
    assert int(first) in b64 and int(last) in b64 and int(first) == min(b64), (
        "the pre-extension fit window must start at the first swept M and end on a swept M")
    window64 = [mm for mm in b64 if int(first) <= mm <= int(last)]
    assert count == {5: "five"}.get(len(window64)), window64
    assert _near(predicted, _p6b_crossover(b64, int(first), int(last)))

    m = re.search(
        r"predicted the crossover there at \$M\\approx(\d+)\$\. The data have not "
        r"crossed by 256, so that prediction was about a quarter low, or more\. "
        r"The fitted column errs both ways: (\d+) at dimension 64, where the data "
        r"have not crossed by 256, and (\d+) at dimension 32, where they have\.",
        text)
    assert m, "the crossover prediction sentence has changed shape"
    assert max(b64) == 256 and b64[256][0] > pmc.BIAS_LIMITED_RATIO
    assert 0.2 < 1 - int(m.group(1)) / 256 < 0.3, "'about a quarter low'"
    assert _near(m.group(2), fits[64]) and fits[64] < 256
    assert _near(m.group(3), fits[32]) and fits[32] > 256

    m = re.search(
        r"At dimension 16 the ratio stops falling \(([\d.]+), ([\d.]+) and ([\d.]+) "
        r"at \$M=(\d+)\$, (\d+) and (\d+)\), and the fitted value swings from (\d+) "
        r"to (\d+) when the \$M=(\d+)\$ point is added, so none is given; \$M\$ "
        r"cannot pass \$N_L=(\d+)\$ there anyway\.", text)
    assert m, "the dim-16 crossover sentence has changed shape"
    point16 = _p6b_point("mixed_chain", 16)["point"]
    b16 = _p6b_slb(point16, "energy")
    for printed, bundles in zip(m.groups()[0:3], m.groups()[3:6]):
        assert _near(printed, b16[int(bundles)][0])
    assert b16[int(m.group(6))][0] > b16[int(m.group(5))][0], "'stops falling'"
    assert int(m.group(6)) == max(b16) == int(m.group(10)) == int(point16["n_l"])
    assert int(m.group(9)) == min(b16) < lo
    assert _near(m.group(7), _p6b_crossover(b16, lo))
    assert _near(m.group(8), _p6b_crossover(b16, int(m.group(9))))

    m = re.search(r"\*\*shifts with size\*\* with no steady trend \(fitted (\d+), "
                  r"(\d+) and (\d+) at dimensions (\d+), (\d+) and (\d+)\)", text)
    assert m, "the crossover-limits sentence has changed shape"
    for printed, dim in zip(m.groups()[:3], m.groups()[3:]):
        assert _near(printed, fits[int(dim)])
    values = [fits[int(d)] for d in m.groups()[3:]]
    assert not (values == sorted(values) or values == sorted(values, reverse=True)), (
        "'no steady trend': the fitted crossover is now monotone in dimension")


# --- part 6, U2b: Result 3, mcsolve against its own noise -----------------
#
# Result 3 called a filled mcsolve marker "an error mcsolve genuinely has",
# although section 4 shows mcsolve has no bias at any trajectory count; called
# System B's 1.43 at dim 256 "right on the line" while System C's 1.43 was a
# plain "filled"; and quoted a coherence swing of 33.7x -> 16.5x from the
# retired bias-only scoring without saying so. Every number below is
# recomputed through plot_method_comparison, the module that draws the figures.

_p6c_superseded_commit = "00ba14f"   # replaced job 19559989's System B dim-64 file


def _p6c_region(doc: str) -> str:
    """Result 3 from the mcsolve-noise paragraph up to System C's heading."""
    start = doc.index("**Where `mcsolve` sits against its own noise, by the same rule.**")
    end = doc.index("#### System C — oscillator (dim 64", start)
    return doc[start:end]


def _p6c_point(system: str, dim: int) -> dict:
    path = DATA / f"method_comparison_{system}_dim{dim}.json"
    assert path.exists(), f"{path.name} is quoted in BENCHMARKS.md but not committed"
    return json.loads(path.read_text(encoding="utf-8"))


def _p6c_mcsolve(point: dict, observable: str):
    """(error, s.e.m.) of mcsolve through method_errors, the figures' scoring,
    or None when the file ran no mcsolve."""
    import plot_method_comparison as pmc
    row = next((r for r in pmc.method_errors(point, observable)
                if r[0] == "mcsolve"), None)
    if row is None:
        return None
    assert row[4] == 500, f"mcsolve ran {row[4]} trajectories, not 500"
    return row[2], row[5]


def _p6c_files():
    """(system, dim) of every canonical Result 3 data file."""
    canonical = re.compile(r"^method_comparison_(.+)_dim(\d+)$")
    return sorted((m.group(1), int(m.group(2)))
                  for p in DATA.glob("method_comparison_*_dim*.json")
                  if (m := canonical.match(p.stem)))


def _p6c_superseded_mixed64() -> dict:
    """Job 19559989's System B dim-64 file, as committed just before the
    re-run replaced it. The prose names it, so a missing history fails."""
    import subprocess
    spec = (f"{_p6c_superseded_commit}^:benchmarks/data/"
            "method_comparison_mixed_chain_dim64.json")
    try:
        out = subprocess.run(["git", "show", spec], cwd=BENCHMARKS.parent,
                             capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.fail(f"Result 3 quotes a superseded file from git history ({spec}): {exc}")
    return json.loads(out.stdout.decode("utf-8"))


def test_p6c_result3_mcsolve_noise_table_matches_the_data(doc):
    """The three-row table: mcsolve's energy error, s.e.m., their ratio and
    the sqrt(2) marker at dimension 64 on each system."""
    import plot_method_comparison as pmc
    region = _p6c_region(doc)
    assert re.search(
        r"Scored on the same \$\\sqrt\{\\text\{bias\}\^2\+\\text\{s\.e\.m\.\}\^2\}\$ "
        r"as SLB, on the energy at dimension 64 on each system, 500 trajectories "
        r"each:", _flat(region)), "the table's lead-in no longer names its observable"
    table = re.search(r"^\| system \| `mcsolve` error \| its s\.e\.m\. \| ratio \| marker \|\n"
                      r"\|[-|]+\|\n((?:\|[^\n]*\|\n)+)", region, re.M)
    assert table, "Result 3's mcsolve-noise table has moved or changed its header"
    rows = re.findall(r"^\| ([ABC]) \w+ \| ([^|]+?) \| ([^|]+?) \| \*\*([\d.]+)\*\* "
                      r"\| (hollow|filled) \|$", table.group(1), re.M)
    assert [r[0] for r in rows] == ["A", "C", "B"]
    letter_to_system = {"A": "spin_chain", "B": "mixed_chain", "C": "oscillator_bath"}
    for letter, q_err, q_sem, q_ratio, q_marker in rows:
        err, sem = _p6c_mcsolve(_p6c_point(letter_to_system[letter], 64)["point"], "energy")
        for printed, measured in ((q_err, err), (q_sem, sem)):
            value, half = _decode_with_precision(printed)
            assert abs(measured - value) <= half, (letter, printed, measured)
        assert _near(q_ratio, err / sem), (letter, q_ratio, err / sem)
        assert q_marker == ("filled" if pmc._bias_limited(err, sem) else "hollow"), letter


def test_p6c_result3_filled_mcsolve_markers_are_called_noise(doc):
    """A filled mcsolve marker is noise landing past sqrt(2) s.e.m., not a
    bias: the tally of filled points over every Result 3 file, the Gaussian
    32%, and how far C at dim 64 and B at dim 256 sit past the line."""
    import plot_method_comparison as pmc
    text = _flat(_p6c_region(doc))
    for stale in ("genuinely has", "right on the line", "noise floor, not a converged"):
        assert stale not in text, f"Result 3 still says {stale!r}"
    assert "`mcsolve` has no bias at any trajectory count (§4)" in text
    assert "**Every `mcsolve` error in this section is sampling noise**" in text
    m = re.search(
        r"If the noise moved the whole curve by one Gaussian draw, the marker would "
        r"be filled (\d+)% of the time: the chance that a draw lands more than one "
        r"standard deviation out\. Across the (\d+) Result 3 files with an `mcsolve` "
        r"run, (\d+) of the (\d+) `mcsolve` points are filled, at ratios from "
        r"([\d.]+) to ([\d.]+)\. The observables of one run share its (\d+) "
        r"trajectories, so the (\d+) are far fewer independent draws\. C's ratio in "
        r"the table \(([\d.]+)\) sits ([\d.]+)% past the \$\\sqrt\{2\}\$ line and "
        r"System B's at dimension 256 \(([\d.]+), below\) sits ([\d.]+)% past it", text)
    assert m, "Result 3's filled-means-noise paragraph has changed shape"
    (q_gauss, q_files, q_filled, q_points, q_lo, q_hi, q_ntraj, q_points_again,
     q_c, q_c_past, q_b, q_b_past) = m.groups()

    assert int(q_ntraj) == 500 and q_points_again == q_points
    # The paragraph's own trajectory counts, at its start and at its end.
    head = re.search(r"whether this run's (\d+)-trajectory mean happened to land", text)
    tail = re.search(r"ratio as a ratio against (\d+) trajectories, not against a "
                     r"converged `mcsolve`\.", text)
    assert head and tail, "the trajectory counts in the filled-marker paragraph moved"
    assert head.group(1) == tail.group(1) == q_ntraj
    assert _near(q_gauss, 100 * math.erfc(1 / math.sqrt(2)))
    files, ratios, filled = 0, [], 0
    for system, dim in _p6c_files():
        point = _p6c_point(system, dim)["point"]
        if _p6c_mcsolve(point, point["observables"][0]) is None:
            continue
        files += 1
        scored = {obs: _p6c_mcsolve(point, obs) for obs in point["observables"]}
        if "zz_per_bond" in scored:
            # zz divided by a constant: the same error/s.e.m. ratio as zz.
            (e1, s1), (e2, s2) = scored.pop("zz_per_bond"), scored["zz"]
            assert e1 / s1 == pytest.approx(e2 / s2, rel=1e-9), (system, dim)
        for err, sem in scored.values():
            ratios.append(err / sem)
            filled += bool(pmc._bias_limited(err, sem))
    assert int(q_files) == files
    assert int(q_points) == len(ratios)
    assert int(q_filled) == filled
    # "filled, at ratios from ... to ...": the range of the filled points only.
    filled_ratios = [r for r in ratios if r > pmc.BIAS_LIMITED_RATIO]
    assert len(filled_ratios) == filled
    assert _near(q_lo, min(filled_ratios)) and _near(q_hi, max(filled_ratios))

    for printed, past, system, dim in ((q_c, q_c_past, "oscillator_bath", 64),
                                       (q_b, q_b_past, "mixed_chain", 256)):
        err, sem = _p6c_mcsolve(_p6c_point(system, dim)["point"], "energy")
        assert pmc._bias_limited(err, sem), f"{system} dim {dim} is not filled"
        assert _near(printed, err / sem)
        assert _near(past, 100 * (err / sem / pmc.BIAS_LIMITED_RATIO - 1))


def test_p6c_result3_coherence_seed_swing_matches_both_runs(doc):
    """System B dim 64 ran twice. The coherence deficit on the old bias-only
    score (33.7x -> 16.5x) and on the current combined score (10.0x -> 8.3x),
    recomputed from the superseded file in git history and the committed one."""
    import inspect
    import plot_method_comparison as pmc
    import run_method_comparison
    m = re.search(
        r"System B at dimension 64 ran twice, (\d+) trajectories each: job (\d{8}) on "
        r"(\w+) \(its file is superseded and kept in git history\) and job (\d{8}) on "
        r"(\w+) \(the file drawn here\)\. `mcsolve` draws fresh random numbers each "
        r"run, while SLB's seed is fixed, so SLB's \$M=16\$ coherence error \((\d+) "
        r"realizations\) is identical in both files\. On the old bias-only score, "
        r"`mcsolve`'s coherence error changed by ([\d.]+)x between the runs, and "
        r"SLB's coherence deficit went from ([\d.]+)x worse to ([\d.]+)x worse\. On the current score "
        r"`mcsolve`'s coherence error changed by only ([\d.]+)x, and SLB's deficit "
        r"reads \*\*([\d.]+)x worse\*\* in the first run and \*\*([\d.]+)x worse\*\* "
        r"in the second, the value the table below quotes\. Both coherence points "
        r"are hollow \(ratios ([\d.]+) and ([\d.]+)\)\.", _flat(_p6c_region(doc)))
    assert m, "Result 3's coherence seed-swing paragraph has changed shape"
    (q_ntraj, q_old_job, q_old_host, q_new_job, q_new_host, q_runs, q_bias_swing,
     q_bias_old, q_bias_new, q_comb_swing, q_comb_old, q_comb_new,
     q_hollow_old, q_hollow_new) = m.groups()
    old, new = _p6c_superseded_mixed64(), _p6c_point("mixed_chain", 64)

    # mcsolve is called with no seed; SLB's seed is the same in both files.
    assert "seed" not in inspect.getsource(run_method_comparison.run_mcsolve)
    assert not any("seed" in key for key in common.MC_OPTIONS)
    assert old["meta"]["params"]["rng_slb"] == new["meta"]["params"]["rng_slb"]

    out = {}
    for tag, document, q_job, q_host in (("old", old, q_old_job, q_old_host),
                                         ("new", new, q_new_job, q_new_host)):
        execution = document["meta"]["execution"]
        assert q_job == str(execution["slurm"]["job_id"]), tag
        assert q_host == execution["hostname"], tag
        point = document["point"]
        assert point["methods"]["mcsolve"]["ntraj"] == int(q_ntraj)
        err, sem = _p6c_mcsolve(point, "coherence")
        reference = pmc.mean_curve(point["reference"]["curves"]["coherence"])
        bias = _deviation(pmc.mean_curve(point["methods"]["mcsolve"]["curves"]["coherence"]),
                          reference)
        slb = next(r for r in pmc.method_errors(point, "coherence")
                   if r[0] == "slb" and r[3] == "M=16")
        assert slb[4] == int(q_runs)
        samples = next(r for r in point["methods"]["slb"] if r["M"] == 16)["samples"]
        out[tag] = dict(err=err, sem=sem, bias=bias, slb=slb[2], samples=samples)
    assert out["old"]["samples"] == out["new"]["samples"], "SLB's M=16 run differs"
    assert q_old_host != q_new_host

    assert _near(q_bias_swing, out["new"]["bias"] / out["old"]["bias"])
    assert _near(q_bias_old, out["old"]["slb"] / out["old"]["bias"])
    assert _near(q_bias_new, out["new"]["slb"] / out["new"]["bias"])
    assert _near(q_comb_swing, out["new"]["err"] / out["old"]["err"])
    assert _near(q_comb_old, out["old"]["slb"] / out["old"]["err"])
    assert _near(q_comb_new, out["new"]["slb"] / out["new"]["err"])
    for printed, tag in ((q_hollow_old, "old"), (q_hollow_new, "new")):
        assert not pmc._bias_limited(out[tag]["err"], out[tag]["sem"]), tag
        assert _near(printed, out[tag]["err"] / out[tag]["sem"])

    # "the value the table below quotes": System B's coherence row.
    start = doc.index("#### System B — mixed-field chain (dim 64")
    section_b = doc[start:doc.index("\n#### ", start + 1)]
    row = re.search(r"^\| `coherence` \| [^|]+ \| [^|]+ \| \*\*([\d.]+)x worse\*\* \|",
                    section_b, re.M)
    assert row and row.group(1) == q_comb_new


# --- Result 3, System C: the table, the two costs, the observable split ------
#
# The System C part of Result 3 printed "470" for a table cell of 472x, said
# bias dominates one SLB run "on this system" when that holds for the energy
# alone, and called `x_sx` and `sz` the "independent" observables when all four
# of n, n2, sz and x_sx are terms of the energy. Its six-row table was not read
# by any test. These pin all three to the file they quote, through
# plot_method_comparison.method_errors, common.tavg_bias_sem_rmse and
# common.reconstruct_energy.

_P6D_WORDS = {"four": 4, "five": 5, "six": 6}


def _p6d_region(doc: str) -> str:
    """Result 3's System C subsection, raw (tables are matched line by line)."""
    start = doc.index("#### System C — oscillator (dim 64")
    end = doc.index("#### System B — mixed-field chain", start)
    return doc[start:end]


def _p6d_oscillator() -> dict:
    """Per observable at dim 64, M=16: Result 3's scored errors for SLB and
    mcsolve, and one SLB run's spread, the 16-run mean's bias and one run's
    mean distance from the reference."""
    import plot_method_comparison as pmc

    path = DATA / "method_comparison_oscillator_bath_dim64.json"
    if not path.exists():
        pytest.fail(f"{path.name} is quoted in BENCHMARKS.md but not committed")
    document = json.loads(path.read_text(encoding="utf-8"))
    point = document["point"]
    slb16 = next(r for r in point["methods"]["slb"] if r["M"] == 16)
    samples_all = np.asarray(slb16["samples"], dtype=float)
    out = {"observables": list(point["observables"]),
           "substeps": document["meta"]["substeps"],
           "native_wall": point["methods"]["native"]["wall_s"],
           "slb_wall": slb16["wall_s"], "n_runs": int(slb16["n_runs"]),
           "reference": {}, "obs": {}}
    for index, obs in enumerate(point["observables"]):
        rows = pmc.method_errors(point, obs)
        mc = next(r for r in rows if r[0] == "mcsolve")
        slb = next(r for r in rows if r[0] == "slb" and r[3] == "M=16")
        reference = pmc.mean_curve(point["reference"]["curves"][obs])
        samples = samples_all[:, index, :]
        bias = common.tavg_bias_sem_rmse(samples, reference)[0]
        out["reference"][obs] = reference
        out["obs"][obs] = {
            "mc": mc, "slb": slb,
            "ratio": mc[2] / slb[2],
            "bias": bias,
            "spread": float(np.mean(samples.std(axis=0, ddof=1))),
            "single": float(np.mean([_deviation(s, reference) for s in samples])),
        }
    return out


def test_p6d_result3_oscillator_table_matches_the_data(doc):
    """All six rows of System C's table: SLB's and mcsolve's scored errors,
    their ratio, and the 16-realization ensemble's speed against native."""
    d = _p6d_oscillator()
    region = _p6d_region(doc)
    assert "| observable | SLB (`M=16`) error | `mcsolve` error | SLB/mc ratio | SLB speed vs native |" in region
    rows = re.findall(r"^\| `(\w+)` \| ([^|]+) \| ([^|]+) \| \**([\d.]+)x better\** \| ([\d.]+)x \|$",
                      region, re.M)
    assert [r[0] for r in rows] == d["observables"], (
        f"the table lists {[r[0] for r in rows]}; the file has {d['observables']}")
    assert d["n_runs"] == 16
    for obs, slb_cell, mc_cell, q_ratio, q_speed in rows:
        o = d["obs"][obs]
        assert o["mc"][4] == 500 and o["slb"][4] == 16
        for measured, cell, what in ((o["slb"][2], slb_cell, "SLB"), (o["mc"][2], mc_cell, "mcsolve")):
            value, half = _decode_with_precision(cell)
            assert abs(measured - value) <= half, (
                f"{obs} {what} error: measured {measured:.4e}, printed {cell}")
        _assert_rounds_to(o["ratio"], q_ratio, f"{obs} SLB/mc ratio")
        _assert_rounds_to(d["native_wall"] / d["slb_wall"], q_speed, f"{obs} ensemble speed")


def test_p6d_result3_oscillator_two_costs_paragraph_matches_the_data(doc):
    """'Two costs for SLB': the ensemble and one-run speeds, and the claim
    that one run suffices only on the energy, where one run's spread is below
    the 16-run mean's bias; on every other observable it is not."""
    d = _p6d_oscillator()
    m = re.search(
        r"\*\*Two costs for SLB\.\*\* The `SLB speed vs native` column is the "
        r"(\d+)-realization ensemble \(([\d.]+)x\); the paragraph below quotes one run "
        r"\((\d+)x; §5\.1\)\. One run is enough only where its bias outweighs its noise, "
        r"and here that is the energy alone: one run's spread \(the standard deviation "
        r"across the (\d+) realizations, " + LATEX + r"\) is below the bias of their mean "
        r"\(" + LATEX + r"\)\. On the other (\w+) observables one run's spread is "
        r"([\d.]+) to ([\d.]+) times the bias, and the mean of (\d+) sits ([\d.]+) to "
        r"([\d.]+) times closer, so use the ensemble\.",
        _flat_ws(_p6d_region(doc)))
    assert m, "Result 3's System C 'Two costs for SLB' paragraph has changed shape"
    (q_ens, q_34, q_54, q_runs, sp_m, sp_e, b_m, b_e,
     q_others, lo_sb, hi_sb, q_mean2, lo_cl, hi_cl) = m.groups()
    n = d["n_runs"]
    assert int(q_ens) == int(q_runs) == int(q_mean2) == n == 16
    _assert_rounds_to(d["native_wall"] / d["slb_wall"], q_34, "ensemble vs native")
    _assert_rounds_to(d["native_wall"] / (d["slb_wall"] / n), q_54, "one run vs native")

    energy = d["obs"]["energy"]
    _assert_latex_rounds_to(energy["spread"], sp_m, sp_e, "one run's energy spread")
    _assert_latex_rounds_to(energy["bias"], b_m, b_e, "16-run energy bias")

    # 'the energy alone': the only observable whose one-run spread is below
    # the bias of the 16-run mean.
    below = [o for o, v in d["obs"].items() if v["spread"] < v["bias"]]
    assert below == ["energy"], f"one run's spread is below the bias on {below}"
    others = [v for o, v in d["obs"].items() if o != "energy"]
    assert _P6D_WORDS[q_others] == len(others)
    spread_over_bias = [v["spread"] / v["bias"] for v in others]
    closer = [v["single"] / v["bias"] for v in others]
    _assert_rounds_to(min(spread_over_bias), lo_sb, "smallest spread/bias")
    _assert_rounds_to(max(spread_over_bias), hi_sb, "largest spread/bias")
    _assert_rounds_to(min(closer), lo_cl, "smallest one-vs-sixteen")
    _assert_rounds_to(max(closer), hi_cl, "largest one-vs-sixteen")


def test_p6d_result3_oscillator_observable_split_matches_the_data(doc):
    """'The 914x headline is real but observable-dependent': the headline
    itself, the energy as exactly the four-term sum (common.reconstruct_energy),
    the span of each term, the 472-914x on energy/n/n2, the three smaller
    ratios, x_sx as the smallest of the six and the only tie within noise."""
    d = _p6d_oscillator()
    m = re.search(
        r"\*\*The (\d+)x headline is real but observable-dependent\.\*\* "
        r"The energy is built from (\w+) of the other observables \(§2\.4\): \$\$ "
        r"\\langle H\\rangle = \\omega_0\\left\(\\langle n\\rangle\+\\tfrac12\\right\) \+ "
        r"\\chi\\langle n\^2\\rangle \+ \\tfrac\{\\Delta\}\{2\}\\langle\\sigma_z\\rangle \+ "
        r"g_\{\\rm int\}\\langle x\\sigma_x\\rangle \$\$ The first two terms carry almost "
        r"all of the energy's motion\. Over the run the energy spans ([\d.]+); the `n` "
        r"term spans ([\d.]+) and the `n2` term ([\d.]+), while the `sz` term spans "
        r"([\d.]+) and the `x_sx` term ([\d.]+)\. So `energy`, `n` and `n2` are nearly one "
        r"curve, and in the table above SLB's advantage on them is (\d+)–(\d+)x\. The "
        r"energy barely registers the other two terms, so its (\d+)x says nothing about "
        r"them: the advantage drops to \*\*([\d.]+)x\*\* on `sz` and "
        r"\*\*([\d.]+)x\*\* on `x_sx`, and the second is a tie within noise, its gap "
        r"smaller than `mcsolve`'s s\.e\.m\. The coherence, the one observable that is not a "
        r"term of \$H\$, gives ([\d.]+)x\. The `x_sx` figure is included because it is "
        r"the harshest of the (\w+)\.", _flat_ws(_p6d_region(doc)))
    assert m, "Result 3's System C observable paragraph has changed shape"
    (q_headline, q_four, q_e, q_n, q_n2, q_sz, q_xsx, lo, hi, q_energy, r_sz, r_xsx,
     r_coh, q_six) = m.groups()

    ref = d["reference"]
    terms = ("n", "n2", "sz", "x_sx")
    assert _P6D_WORDS[q_four] == len(terms)
    rebuilt = common.reconstruct_energy("oscillator_bath", {t: ref[t] for t in terms})
    assert rebuilt is not None and "coherence" not in terms
    span = float(np.ptp(ref["energy"]))
    assert float(np.max(np.abs(rebuilt - ref["energy"]))) < 1e-9 * span, (
        "the energy is no longer the four-term sum of n, n2, sz and x_sx")
    p = common.OSCILLATOR_PARAMS
    weight = {"n": p["omega0"], "n2": p["anh"], "sz": 0.5 * p["spin_gap"], "x_sx": p["coupling"]}
    term_span = {t: float(np.ptp(weight[t] * ref[t])) for t in terms}
    _assert_rounds_to(span, q_e, "energy span")
    for t, printed in zip(terms, (q_n, q_n2, q_sz, q_xsx)):
        _assert_rounds_to(term_span[t], printed, f"span of the {t} term")
    assert term_span["n"] + term_span["n2"] > 0.99 * span, "the first two terms no longer carry the energy"

    ratio = {o: v["ratio"] for o, v in d["obs"].items()}
    trio = [ratio[o] for o in ("energy", "n", "n2")]
    _assert_rounds_to(min(trio), lo, "smallest of energy/n/n2")
    _assert_rounds_to(max(trio), hi, "largest of energy/n/n2")
    _assert_rounds_to(ratio["energy"], q_energy, "energy ratio")
    _assert_rounds_to(ratio["sz"], r_sz, "sz ratio")
    _assert_rounds_to(ratio["x_sx"], r_xsx, "x_sx ratio")
    _assert_rounds_to(ratio["coherence"], r_coh, "coherence ratio")
    assert _P6D_WORDS[q_six] == len(ratio)
    assert min(ratio, key=ratio.get) == "x_sx", "x_sx is no longer the harshest observable"
    _assert_rounds_to(ratio["energy"], q_headline, "headline energy ratio")
    # Result 3's gap rule: x_sx is the one tie, and mcsolve holds the larger s.e.m.
    verdict = {o: _p6g_verdict(v["slb"], v["mc"]) for o, v in d["obs"].items()}
    assert [o for o, v in verdict.items() if v != "win"] == ["x_sx"], verdict
    x_sx = d["obs"]["x_sx"]
    assert x_sx["mc"][5] > x_sx["slb"][5], "the larger s.e.m. on x_sx is no longer mcsolve's"
    figure = "benchmark_comparison_oscillator_bath_x_sx.png"
    assert (BENCHMARKS / figure).exists(), f"{figure} is embedded but not committed"
    assert f"]({figure})" in _p6d_region(doc)


# --- Document review part 6, unit U4: Result 3, System B -------------------
#
# The System B subsection quoted M=16 numbers the figures above it do not draw,
# blamed two noise-dominated losses on the estimator, set a 13.5x headline
# against a hollow mcsolve point without saying so, counted zz_per_bond (zz
# divided by a constant) as a sixth observable, and named no denominator for
# its speed ratios. Every number below is recomputed through
# plot_method_comparison.method_errors, the scoring the figures use.

_P6E_DISTINCT = ("energy", "zz", "sx", "sz", "coherence")
_P6E_COUNT = {"three": 3, "four": 4, "five": 5, "six": 6}


def _p6e_region(doc: str) -> str:
    """Result 3's System B subsection, flattened."""
    text = _flat(doc)
    start = text.index("#### System B — mixed-field chain (dim 64")
    return text[start:text.index("#### System A — TFIM chain", start)]


def _p6e_file(dim: int) -> dict:
    """One System B Result 3 file, which the subsection names."""
    path = DATA / f"method_comparison_mixed_chain_dim{dim}.json"
    assert path.exists(), f"{path.name} is named by Result 3 but not committed"
    return json.loads(path.read_text(encoding="utf-8"))


def _p6e_point_of(dim: int) -> dict:
    """The 'point' block of one System B Result 3 file."""
    return _p6e_file(dim)["point"]


def _p6e_rows(dim: int, observable: str) -> dict:
    """{'mcsolve' | 'M=16' | ...: method_errors row} for one observable."""
    import plot_method_comparison as pmc
    rows = pmc.method_errors(_p6e_point_of(dim), observable)
    return {("mcsolve" if r[0] == "mcsolve" else r[3]): r
            for r in rows if r[0] in ("mcsolve", "slb")}


def _p6e_ratio(row) -> float:
    """error / s.e.m. of one method_errors row."""
    return row[2] / row[5]


def _p6e_spread(row) -> float:
    """One sample's time-averaged standard deviation: s.e.m. x sqrt(n)."""
    return row[5] * math.sqrt(row[4])


def _p6e_sci(mantissa: str, exponent: str) -> float:
    """('4.06', '⁻²') -> 1e-2, the scale of a printed mantissa."""
    return 10.0 ** int(exponent.translate(SUPERSCRIPT))


def test_p6e_system_b_figures_do_not_draw_m16(doc, monkeypatch):
    """The note under the System B figures: which bundle sizes each curve
    draws at dims 64 and 128, through the plotter's own window rule."""
    import plot_method_comparison as pmc
    monkeypatch.setattr(pmc, "_MAX_POINTS", pmc.MAX_CURVE_POINTS)
    monkeypatch.setattr(pmc, "_MIN_M", pmc.MIN_M_PLOTTED)
    text = _p6e_region(doc)
    m = re.search(
        r"The tables below quote \$M=16\$, which these figures do not draw at "
        r"dimensions (\d+) and (\d+)\. Each curve keeps at most (\w+) bundle sizes"
        r".*?at dimension \1 the curves show \$M=(\d+)\$ to (\d+) on energy and sx "
        r"and \$M=(\d+)\$ to (\d+) on coherence, and at dimension \2 they show "
        r"\$M=(\d+)\$ to (\d+) on all three\. The \$M=(\d+)\$ numbers come from the same "
        r"data files \(`method_comparison_mixed_chain_dim(\d+)\.json` and "
        r"`_dim(\d+)\.json`\), scored the same way\.", text)
    assert m, "the note on which bundle sizes the System B figures draw has changed shape"
    d1, d2, most, e_lo, e_hi, c_lo, c_hi, b_lo, b_hi, tabled, f1, f2 = m.groups()
    assert int(tabled) == 16 and (f1, f2) == (d1, d2)
    for dim in (int(f1), int(f2)):
        assert any(r["M"] == int(tabled) for r in _p6e_point_of(dim)["methods"]["slb"])
    assert _P6E_COUNT[most] == pmc.MAX_CURVE_POINTS
    for image in ("energy", "sx", "coherence"):
        assert (BENCHMARKS / f"benchmark_comparison_mixed_chain_{image}.png").exists()
        assert f"(benchmark_comparison_mixed_chain_{image}.png)" in text

    def window(dim, observable):
        rows = pmc.method_errors(_p6e_point_of(dim), observable)
        return [pmc._m_of(r) for r in pmc._curve_window([r for r in rows if r[0] == "slb"])]

    for observable in ("energy", "sx"):
        drawn = window(int(d1), observable)
        assert (drawn[0], drawn[-1]) == (int(e_lo), int(e_hi)), (observable, drawn)
    drawn = window(int(d1), "coherence")
    assert (drawn[0], drawn[-1]) == (int(c_lo), int(c_hi)), drawn
    for observable in ("energy", "sx", "coherence"):
        drawn = window(int(d2), observable)
        assert (drawn[0], drawn[-1]) == (int(b_lo), int(b_hi)), (observable, drawn)
    for dim in (int(d1), int(d2)):
        for observable in ("energy", "sx", "coherence"):
            assert 16 not in window(dim, observable), (dim, observable)


def test_p6e_system_b_tables_match_the_data(doc):
    """Every cell of the three System B tables (dim 64 at M=16, dim 128's
    cost table, dim 128 at M=256), recomputed through method_errors."""
    start = doc.index("#### System B — mixed-field chain (dim 64")
    raw = doc[start:doc.index("#### System A — TFIM chain", start)]
    sci = r"([\d.]+)×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)"

    def check(mantissa, exponent, measured):
        assert _near(mantissa, measured / _p6e_sci(mantissa, exponent)), (
            mantissa, exponent, measured)

    def ratio_of(a, b):
        return max(a, b) / min(a, b)

    rows64 = re.findall(r"^\| `(\w+)` \| " + sci + r" \| " + sci
                        + r" \| \**([\d.]+)x (better|worse)\** \| ([\d.]+)x \|$", raw, re.M)
    assert [r[0] for r in rows64] == list(_p6e_point_of(64)["observables"])
    native = _p6e_point_of(64)["methods"]["native"]["wall_s"]
    for obs, sm, se, mm, me, ratio, word, speed in rows64:
        r = _p6e_rows(64, obs)
        check(sm, se, r["M=16"][2])
        check(mm, me, r["mcsolve"][2])
        assert word == ("worse" if r["M=16"][2] > r["mcsolve"][2] else "better"), obs
        assert _near(ratio, ratio_of(r["M=16"][2], r["mcsolve"][2])), obs
        assert _near(speed, native / r["M=16"][1]), obs

    cost = re.findall(r"^\| (`mcsolve`, (\d+) trajectories|SLB, `M=(\d+)`) \| ([\d,]+) s \| "
                      + sci + r" \| (—|\**([\d.]+)x (better|worse)\**) \|$", raw, re.M)
    assert len(cost) == 3, cost
    r128 = _p6e_rows(128, "energy")
    for _label, ntraj, m_value, wall, mant, exp, _cell, ratio, word in cost:
        row = r128[f"M={m_value}"] if m_value else r128["mcsolve"]
        if ntraj:
            assert int(ntraj) == row[4]
        assert _near(wall, row[1]), _label
        check(mant, exp, row[2])
        if m_value:
            assert word == ("worse" if row[2] > r128["mcsolve"][2] else "better")
            assert _near(ratio, ratio_of(row[2], r128["mcsolve"][2]))

    rows256 = re.findall(r"^\| `(\w+)` \| " + sci + r" \| " + sci
                         + r" \| \**([\d.]+)x (better|worse)\** \|$", raw, re.M)
    assert sorted(r[0] for r in rows256) == sorted(_p6e_point_of(128)["observables"])
    for obs, sm, se, mm, me, ratio, word in rows256:
        r = _p6e_rows(128, obs)
        check(sm, se, r["M=256"][2])
        check(mm, me, r["mcsolve"][2])
        assert word == ("worse" if r["M=256"][2] > r["mcsolve"][2] else "better"), obs
        assert _near(ratio, ratio_of(r["M=256"][2], r["mcsolve"][2])), obs


def test_p6e_zz_per_bond_repeats_zz(doc):
    """Result 3 says once, near its top, that zz_per_bond is zz over the bond
    count and is left out of every tally. In every Result 3 file its error and
    s.e.m. are zz's divided by the bond count (size - 1), for mcsolve and every
    SLB M, and no other Result 3 paragraph repeats the explanation."""
    import plot_method_comparison as pmc
    m = re.search(r"`zz_per_bond` is `zz` divided by the bond count, so every ratio on "
                  r"its row repeats `zz`'s\. The tables keep the row, but every count and "
                  r"tally in this section leaves it out and counts distinct observables "
                  r"only\.", _p6a_region(doc))
    assert m, "the zz_per_bond note at the top of Result 3 has changed shape"
    checked = 0
    for (system, dim), document in _p6a_files().items():
        point = document["point"]
        if "zz_per_bond" not in point["observables"]:
            continue
        zz = {(r[0], r[3]): r for r in pmc.method_errors(point, "zz")
              if r[0] in ("mcsolve", "slb")}
        per = {(r[0], r[3]): r for r in pmc.method_errors(point, "zz_per_bond")
               if r[0] in ("mcsolve", "slb")}
        assert zz.keys() == per.keys() and zz, (system, dim)
        bonds = point["size"] - 1  # an open chain of `size` spins
        assert bonds >= 1, (system, dim)
        for key in zz:
            assert zz[key][2] == pytest.approx(bonds * per[key][2], rel=1e-9), (system, dim, key)
            assert zz[key][5] == pytest.approx(bonds * per[key][5], rel=1e-9), (system, dim, key)
        checked += 1
    assert checked, "no Result 3 file carries zz_per_bond"
    start = doc.index("### Result 3 — accuracy versus cost")
    prose = " ".join(line for line in doc[start:doc.index("### Result 4", start)].splitlines()
                     if not line.startswith("|"))
    assert prose.count("`zz_per_bond`") == 1, "Result 3 explains zz_per_bond more than once"


def test_p6e_system_b_dim64_speed_and_tally(doc):
    """SLB at M=16 against mcsolve and native RK4 at dim 64: all three walls,
    both ratios, the matched substep count, and the verdicts under Result 3's
    gap rule -- coherence the only real difference, the other four ties."""
    text = _p6e_region(doc)
    m = re.search(
        r"At \$M=16\$ SLB's (\d+) realizations take ([\d.]+) s\. That is \*\*(\d+)x less "
        r"time\*\* than `mcsolve`'s (\d+) trajectories \(([\d,]+) s; `mcsolve` steps "
        r"adaptively, so the step counts differ\) and \*\*([\d.]+)x less\*\* than native "
        r"RK4 at the same (\d+) substeps \(([\d,]+) s\)\. On accuracy only `coherence` is "
        r"a real difference: SLB is \*\*([\d.]+)x worse\*\*\. On energy, zz and sz its "
        r"error is ([\d.]+)x to ([\d.]+)x smaller, and on `sx` ([\d.]+)x larger, but each "
        r"of those (\w+) gaps is smaller than the larger of the two s\.e\.m\.s, so they are "
        r"ties within noise\.", text)
    assert m, "the dim-64 speed sentence has changed shape"
    (n_runs, slb_s, r_mc, ntraj, mc_s, r_nat, subs, nat_s, q_coh, q_lo, q_hi, q_sx,
     q_ties) = m.groups()
    document = _p6e_file(64)
    point = document["point"]
    slb = next(r for r in point["methods"]["slb"] if r["M"] == 16)
    mc, nat = point["methods"]["mcsolve"], point["methods"]["native"]
    assert int(n_runs) == slb["n_runs"] and int(ntraj) == mc["ntraj"]
    assert int(subs) == document["meta"]["substeps"]
    assert _near(slb_s, slb["wall_s"]) and _near(mc_s, mc["wall_s"]) and _near(nat_s, nat["wall_s"])
    assert _near(r_mc, mc["wall_s"] / slb["wall_s"])
    assert _near(r_nat, nat["wall_s"] / slb["wall_s"])
    rows = {o: _p6e_rows(64, o) for o in _P6E_DISTINCT}
    verdict = {o: _p6g_verdict(rows[o]["M=16"], rows[o]["mcsolve"]) for o in _P6E_DISTINCT}
    assert verdict == {"energy": "tie", "zz": "tie", "sx": "tie", "sz": "tie",
                       "coherence": "loss"}, verdict
    ratio = {o: rows[o]["M=16"][2] / rows[o]["mcsolve"][2] for o in _P6E_DISTINCT}
    assert _near(q_coh, ratio["coherence"]) and _near(q_sx, ratio["sx"]) and ratio["sx"] > 1
    better = [1 / ratio[o] for o in ("energy", "zz", "sz")]
    assert min(better) > 1, "SLB's error is no longer smaller on energy, zz and sz"
    assert _near(q_lo, min(better)) and _near(q_hi, max(better))
    assert _P6E_COUNT[q_ties] == sum(v == "tie" for v in verdict.values())
    assert "modestly" not in text


def test_p6e_system_b_dim64_losses_are_noise(doc):
    """The sx and coherence losses at dim 64: all four points are hollow, the
    sample-budget factor, each method's one-sample spread, and the spread at
    M=256 that makes a larger M the second knob."""
    import plot_method_comparison as pmc
    text = _p6e_region(doc)
    m = re.search(
        r"Neither the `sx` gap nor the `coherence` loss shows a bias behind it\. All four "
        r"points behind those two ratios are hollow: SLB's error is ([\d.]+) and ([\d.]+) "
        r"times its s\.e\.m\., "
        r"`mcsolve`'s ([\d.]+) and ([\d.]+)\. So both gaps come from SLB's larger "
        r"sampling noise, "
        r"(\d+) realizations against (\d+) trajectories, and that budget alone is worth "
        r"a factor \$\\sqrt\{(\d+)/(\d+)\} = ([\d.]+)\$ in s\.e\.m\. Per sample, one SLB "
        r"realization is less noisy than one trajectory on `sx` \(spread ([\d.]+) against "
        r"([\d.]+)\) and ([\d.]+)x noisier on `coherence` \(([\d.]+) against ([\d.]+)\); "
        r"the spread is one sample's standard deviation, averaged over time\. More "
        r"realizations would narrow both gaps, and so would a larger \$M\$, which makes "
        r"each realization less noisy \(§4\): at \$M=(\d+)\$ the spread is ([\d.]+) on "
        r"`sx` and ([\d.]+) on `coherence`\.", text)
    assert m, "the dim-64 sx/coherence loss paragraph has changed shape"
    (s_sx, s_coh, m_sx, m_coh, n16, n500, a, b, budget,
     sp_s, sp_m, noisier, cp_s, cp_m, m_big, big_sx, big_coh) = m.groups()
    sx, coh = _p6e_rows(64, "sx"), _p6e_rows(64, "coherence")
    assert _p6g_verdict(sx["M=16"], sx["mcsolve"]) == "tie", "the paragraph calls sx a gap"
    assert _p6g_verdict(coh["M=16"], coh["mcsolve"]) == "loss", "the paragraph calls coherence a loss"
    assert sx["M=16"][5] > sx["mcsolve"][5] and coh["M=16"][5] > coh["mcsolve"][5], (
        "'SLB's larger sampling noise': SLB's s.e.m. is no longer the larger on both")
    for printed, row in ((s_sx, sx["M=16"]), (s_coh, coh["M=16"]),
                         (m_sx, sx["mcsolve"]), (m_coh, coh["mcsolve"])):
        assert _near(printed, _p6e_ratio(row))
        assert not pmc._bias_limited(row[2], row[5]), "the paragraph calls this point hollow"
    assert (int(n16), int(n500)) == (int(b), int(a)) == (sx["M=16"][4], sx["mcsolve"][4])
    assert _near(budget, math.sqrt(int(a) / int(b)))
    assert _near(sp_s, _p6e_spread(sx["M=16"])) and _near(sp_m, _p6e_spread(sx["mcsolve"]))
    assert _p6e_spread(sx["M=16"]) < _p6e_spread(sx["mcsolve"])
    assert _near(cp_s, _p6e_spread(coh["M=16"])) and _near(cp_m, _p6e_spread(coh["mcsolve"]))
    assert _near(noisier, _p6e_spread(coh["M=16"]) / _p6e_spread(coh["mcsolve"]))
    big = f"M={m_big}"
    assert _near(big_sx, _p6e_spread(sx[big])) and _near(big_coh, _p6e_spread(coh[big]))
    assert _p6e_spread(sx[big]) < _p6e_spread(sx["M=16"])
    assert _p6e_spread(coh[big]) < _p6e_spread(coh["M=16"])
    assert "resolves off-diagonal density-matrix elements better" not in text


def test_p6e_system_b_dim64_m16_energy_is_bias(doc):
    """'M=16 is the wrong setting at this size' rests on a filled energy point."""
    import plot_method_comparison as pmc
    m = re.search(r"Even here its energy error is ([\d.]+) times its s\.e\.m\., a filled "
                  r"point by the \$\\sqrt\{2\}\$ rule, so it is mostly bias", _p6e_region(doc))
    assert m, "the reason M=16 is the wrong setting at dim 64 has changed shape"
    row = _p6e_rows(64, "energy")["M=16"]
    assert _near(m.group(1), _p6e_ratio(row))
    assert pmc._bias_limited(row[2], row[5])


def test_p6e_system_b_dim128_walls_name_their_denominator(doc):
    """38x and 19x against native RK4 at 4 substeps and the 8-substep reference."""
    text = _p6e_region(doc)
    m = re.search(
        r"\*\*At dim 128\*\* \(\$N_L = ([\d{},]+)\$\) `mcsolve` takes ([\d,]+) s at "
        r"\$N_\{\\text\{traj\}\}=(\d+)\$, because every jump must test all ([\d,]+) collapse "
        r"operators\. That is \*\*(\d+)x\*\* the wall-clock of native RK4 at SLB's (\d+) "
        r"substeps \(([\d,]+) s\), and (\d+)x that of the (\d+)-substep certified reference "
        r"\(([\d,]+) s\)\.", text)
    assert m, "the dim-128 mcsolve wall sentence has changed shape"
    n_l, mc_s, ntraj, n_ops, r_nat, subs, nat_s, r_ref, ref_subs, ref_s = m.groups()
    document = _p6e_file(128)
    point = document["point"]
    mc, nat, ref = point["methods"]["mcsolve"], point["methods"]["native"], point["reference"]
    assert _printed(n_l) == _printed(n_ops) == point["n_l"]
    assert int(ntraj) == mc["ntraj"]
    assert int(subs) == document["meta"]["substeps"]
    assert int(ref_subs) == document["meta"]["params"]["ref_substeps"]
    assert ref["method"] == f"native_rk4_substeps{ref_subs}" and ref["selfcheck"]["passed"]
    assert _near(mc_s, mc["wall_s"]) and _near(nat_s, nat["wall_s"]) and _near(ref_s, ref["wall_s"])
    assert _near(r_nat, mc["wall_s"] / nat["wall_s"])
    assert _near(r_ref, mc["wall_s"] / ref["wall_s"])
    # mcsolve ran at every smaller dimension of the same job, so "for the
    # first time" must not come back
    job = document["meta"]["execution"]["slurm"]["job_id"]
    for dim in (4, 8, 16, 32, 64):
        smaller = _p6e_file(dim)
        assert smaller["meta"]["execution"]["slurm"]["job_id"] == job
        assert smaller["point"]["methods"]["mcsolve"]["wall_s"] > 0
    assert "for the first time" not in text and "25.6 hours" not in text


def test_p6e_system_b_dim128_headline_is_against_mcsolve_noise(doc):
    """The 13.5x caveat: mcsolve's energy error is noise because mcsolve is
    unbiased (not because of its marker), the 13.5x is a real win under the
    gap rule, the projected cost of four times the trajectories, and SLB's
    M=256 point is filled."""
    import plot_method_comparison as pmc
    text = _p6e_region(doc)
    m = re.search(
        r"`mcsolve`'s energy error in that table, ([\d.]+)×10⁻², is only ([\d.]+) times "
        r"its s\.e\.m\. of ([\d.]+)×10⁻², and like every `mcsolve` error here it is "
        r"noise\. So the ([\d.]+)x is against (\d+) trajectories, not a converged "
        r"`mcsolve`: (\w+) times the trajectories would roughly halve its error, and the "
        r"\4x with it, at a projected ([\d,]+) s: (\d+)x SLB's ([\d,]+) s\. SLB's "
        r"\$M=256\$ point is filled \(error ([\d.]+) times its s\.e\.m\.\)", text)
    assert m, "the dim-128 mcsolve-noise caveat has changed shape"
    err, ratio, sem, headline, ntraj, times, projected, vs, slb_s, slb_ratio = m.groups()
    assert "point in that table is hollow" not in text
    r = _p6e_rows(128, "energy")
    mc, slb = r["mcsolve"], r["M=256"]
    assert pmc._bias_limited(slb[2], slb[5])
    assert _p6g_verdict(slb, mc) == "win", "the 13.5x is no longer larger than the noise"
    assert _near(err, 100 * mc[2]) and _near(sem, 100 * mc[5]) and _near(ratio, _p6e_ratio(mc))
    assert _near(headline, mc[2] / slb[2]) and int(ntraj) == mc[4]
    factor = _P6E_COUNT[times]
    assert math.sqrt(factor) == 2, "'roughly halve' needs four times the trajectories"
    # Printed to the thousand, so half a unit in the last printed digit is 500 s;
    # a finer printed value would carry a finer tolerance.
    assert _printed(projected) % 1000 == 0, "the projection is quoted to the thousand"
    assert abs(_printed(projected) - factor * mc[1]) <= 500
    assert _near(slb_s, slb[1]) and _near(vs, factor * mc[1] / slb[1])
    assert _near(slb_ratio, _p6e_ratio(slb))


def test_p6e_system_b_dim128_m256_costs_and_tallies(doc):
    """SLB at M=256 against all three solves, and the verdicts under Result 3's
    gap rule over the five distinct observables at M=16 and at M=256."""
    m = re.search(
        r"\*\*At \$M=16\$ SLB is clearly behind only on `coherence`\*\* \(([\d.]+)x "
        r"worse\)\. On the other (\w+) it is ([\d.]+)x to ([\d.]+)x worse, but each gap is "
        r"smaller than the larger of the two s\.e\.m\.s, so those are ties within noise\. "
        r"At \$M=256\$ it takes ([\d,]+) s: still \*\*(\d+)x less time\*\* than `mcsolve` "
        r"\(not at matched steps\), ([\d.]+)x less than native RK4 at the same (\d+) "
        r"substeps, and ([\d.]+)x less than the (\d+)-substep reference\. It is clearly "
        r"ahead on energy, sz and zz \(([\d.]+)x, ([\d.]+)x and ([\d.]+)x better\); `sx` "
        r"and `coherence` are ties within noise\. Each ratio below is against (\d+) "
        r"`mcsolve` trajectories:", _p6e_region(doc))
    assert m, "the dim-128 M=256 cost and tally sentence has changed shape"
    (q_coh, q_other, lo, hi, slb_s, r_mc, r_nat, subs, r_ref, ref_subs, q_e, q_sz, q_zz,
     q_ntraj) = m.groups()
    document = _p6e_file(128)
    point = document["point"]
    assert int(subs) == document["meta"]["substeps"]
    assert int(ref_subs) == document["meta"]["params"]["ref_substeps"]
    rows = {o: _p6e_rows(128, o) for o in _P6E_DISTINCT}
    v16 = {o: _p6g_verdict(rows[o]["M=16"], rows[o]["mcsolve"]) for o in _P6E_DISTINCT}
    assert v16 == {"energy": "tie", "zz": "tie", "sx": "tie", "sz": "tie",
                   "coherence": "loss"}, v16
    worse16 = {o: rows[o]["M=16"][2] / rows[o]["mcsolve"][2] for o in _P6E_DISTINCT}
    assert _near(q_coh, worse16["coherence"])
    others = [r for o, r in worse16.items() if o != "coherence"]
    assert min(others) > 1 and _P6E_COUNT[q_other] == len(others)
    assert _near(lo, min(others)) and _near(hi, max(others))
    v256 = {o: _p6g_verdict(rows[o]["M=256"], rows[o]["mcsolve"]) for o in _P6E_DISTINCT}
    assert v256 == {"energy": "win", "zz": "win", "sx": "tie", "sz": "win",
                    "coherence": "tie"}, v256
    for name, printed in (("energy", q_e), ("sz", q_sz), ("zz", q_zz)):
        assert _near(printed, rows[name]["mcsolve"][2] / rows[name]["M=256"][2]), name
    assert int(q_ntraj) == rows["energy"]["mcsolve"][4]
    wall = rows["energy"]["M=256"][1]
    assert _near(slb_s, wall)
    assert _near(r_mc, point["methods"]["mcsolve"]["wall_s"] / wall)
    assert _near(r_nat, point["methods"]["native"]["wall_s"] / wall)
    assert _near(r_ref, point["reference"]["wall_s"] / wall)


def test_p6e_system_b_coherence_setting_paragraph(doc):
    """'A setting, not a property': the two coherence ratios, both SLB points
    hollow, the one-sample spreads, and the dim-64 -> 128 energy reversal."""
    import plot_method_comparison as pmc
    text = _p6e_region(doc)
    m = re.search(
        r"at this dimension it is ([\d.]+)x worse at \$M=16\$ and ([\d.]+)x at \$M=256\$\. "
        r"Both coherence points are hollow \(error ([\d.]+) and ([\d.]+) times the "
        r"s\.e\.m\.\), so raising \$M\$ closed most of the gap by cutting noise, not bias: "
        r"a larger \$M\$ makes each realization less noisy, with a spread of ([\d.]+) at "
        r"\$M=16\$ and ([\d.]+) at \$M=256\$, against ([\d.]+) for one `mcsolve` trajectory\. "
        r"At \$M=256\$ one realization is ([\d.]+)x less noisy than one trajectory, so what "
        r"remains of the \2x is the budget, (\d+) realizations against (\d+), and more "
        r"realizations would close it\. And \*\*\$M\$ must grow with the system\*\*: SLB's "
        r"own \$M=16\$ energy error grows ([\d.]+)x, from ([\d.]+)×10⁻² at dimension "
        r"(\d+) to ([\d.]+)×10⁻² at (\d+), and is mostly bias at both sizes \(([\d.]+) and "
        r"([\d.]+) times its s\.e\.m\.\)", text)
    assert m, "the coherence setting-not-property paragraph has changed shape"
    (worse16, worse256, e16, e256, sp16, sp256, sp_mc, less_noisy, n_runs, ntraj,
     grows, err64, d64, err128, d128, f64, f128) = m.groups()
    assert (int(d64), int(d128)) == (64, 128)
    coh = _p6e_rows(128, "coherence")
    lo, hi, mc = coh["M=16"], coh["M=256"], coh["mcsolve"]
    assert _near(worse16, lo[2] / mc[2]) and _near(worse256, hi[2] / mc[2])
    for printed, row in ((e16, lo), (e256, hi)):
        assert _near(printed, _p6e_ratio(row))
        assert not pmc._bias_limited(row[2], row[5])
    assert _near(sp16, _p6e_spread(lo)) and _near(sp256, _p6e_spread(hi))
    assert _near(sp_mc, _p6e_spread(mc))
    assert _near(less_noisy, _p6e_spread(mc) / _p6e_spread(hi))
    assert (int(n_runs), int(ntraj)) == (hi[4], mc[4])
    e64, e128 = _p6e_rows(64, "energy"), _p6e_rows(128, "energy")
    assert _near(grows, e128["M=16"][2] / e64["M=16"][2])
    assert _near(f64, _p6e_ratio(e64["M=16"])) and _near(f128, _p6e_ratio(e128["M=16"]))
    assert "edged out" not in text, "'M must grow' rests on a tie within noise again"
    assert _near(err64, 100 * e64["M=16"][2]) and _near(err128, 100 * e128["M=16"][2])
    assert pmc._bias_limited(e64["M=16"][2], e64["M=16"][5])
    assert pmc._bias_limited(e128["M=16"][2], e128["M=16"][5])
    assert "ample at dimension 64" not in text
    assert "not a larger" not in text, "a larger M still cuts the spread here"


# --- Part 6, unit U5: Result 3's System A text (dims 64, 1024, 2048) ---------
#
# The dim-64 paragraph said the exact solve "costs the same as bundling" (it is
# ~10x cheaper than the 16-realization ensemble) and that SLB "is worse than
# mcsolve on every observable" (two of the four distinct observables are ties
# within noise, and zz_per_bond repeats zz). The dim-1024 tally counted
# zz_per_bond as a sixth observable, and the dim-1024 ratio did not say its
# denominator was timed in another job. These tests pin the corrected text.

def _p6f_load(name: str) -> dict:
    path = DATA / f"method_comparison_{name}.json"
    assert path.exists(), f"BENCHMARKS.md quotes {path.name} but it is not committed"
    return json.loads(path.read_text(encoding="utf-8"))


def _p6f_region(doc: str) -> str:
    """Result 3's System A section, flattened."""
    text = _flat(doc)
    start = text.index("#### System A — TFIM chain (dim 64")
    return text[start:text.index("### Result 4", start)]


def _p6f_scores(point: dict, observable: str, bundles: int):
    """(slb row, mcsolve row) from plot_method_comparison.method_errors."""
    import plot_method_comparison as pmc
    rows = pmc.method_errors(point, observable)
    mc = next(r for r in rows if r[0] == "mcsolve")
    slb = next(r for r in rows if r[0] == "slb" and r[3] == f"M={bundles}")
    return slb, mc


def _p6f_is_rescaled_zz(point: dict) -> bool:
    """zz_per_bond is zz divided by one integer constant on the reference curve."""
    zz = np.asarray(point["reference"]["curves"]["zz"], dtype=float)
    per = np.asarray(point["reference"]["curves"]["zz_per_bond"], dtype=float)
    keep = np.abs(per) > 1e-12
    scale = zz[keep] / per[keep]
    return bool(np.allclose(scale, scale[0], rtol=1e-9)
                and abs(scale[0] - round(scale[0])) < 1e-9)


_p6f_words = {"four": 4, "five": 5, "six": 6}


def test_p6f_system_a_dim64_table_matches_the_data(doc):
    """Every cell of the System A dim-64 table, through method_errors."""
    point = _p6f_load("spin_chain_dim64")["point"]
    region = _p6f_region(doc)
    rows = re.findall(r"\| `(\w+)` \| ([^|]+) \| ([^|]+) \| \*{0,2}([\d.]+)x (better|worse)"
                      r"\*{0,2} \| ([\d.]+)x \|", region.split("This is Control 1")[0])
    assert [r[0] for r in rows] == point["observables"]
    nat = point["methods"]["native"]["wall_s"]
    for name, c_slb, c_mc, q_ratio, word, q_speed in rows:
        slb, mc = _p6f_scores(point, name, 16)
        v, half = _decode_with_precision(c_slb)
        assert abs(v - slb[2]) <= half, name
        v, half = _decode_with_precision(c_mc)
        assert abs(v - mc[2]) <= half, name
        assert word == ("worse" if slb[2] > mc[2] else "better")
        assert _near(q_ratio, max(slb[2], mc[2]) / min(slb[2], mc[2])), name
        assert _near(q_speed, nat / slb[1]), name


def test_p6f_control1_paragraph_matches_the_data(doc):
    """The exact solve is ~10x cheaper than SLB's 16-realization ensemble, not
    the same; SLB is clearly worse only on energy and sx; zz and coherence are
    ties within noise; every mcsolve error is noise-limited and every SLB error
    at M=16 bias-limited."""
    import plot_method_comparison as pmc
    document = _p6f_load("spin_chain_dim64")
    point, meta = document["point"], document["meta"]
    text = _p6f_region(doc)

    m = re.search(
        r"This is Control 1\. Davies grouping leaves only (\d+) operators, so the exact "
        r"solve is cheap\. In job (\d{8}), with both at (\d+) substeps, native RK4 took "
        r"([\d.]+) s and SLB's (\d+)-realization ensemble at \$M=(\d+)\$ took ([\d.]+) s\. "
        r"So the exact solve is about (\d+)x cheaper than the ensemble \(the table's "
        r"([\d.]+)x\), and ([\d.]+)x dearer than one realization \(([\d.]+) s\)\. SLB has "
        r"no speed advantage when \$N_L\$ is this small\.", text)
    assert m, "the Control 1 cost sentence has changed shape"
    (q_nl, q_job, q_sub, q_nat, q_runs, q_m, q_slb, q_x, q_tab, q_dear,
     q_one) = m.groups()
    assert int(q_nl) == point["n_l"]
    assert meta["execution"]["slurm"]["job_id"] == q_job
    assert meta["substeps"] == int(q_sub)
    nat = point["methods"]["native"]["wall_s"]
    row = next(r for r in point["methods"]["slb"] if r["M"] == int(q_m))
    assert row["n_runs"] == int(q_runs)
    one = row["wall_s"] / row["n_runs"]
    assert _near(q_nat, nat) and _near(q_slb, row["wall_s"]) and _near(q_one, one)
    assert _near(q_x, row["wall_s"] / nat) and _near(q_tab, nat / row["wall_s"])
    assert _near(q_dear, nat / one)

    e = re.search(
        r"On accuracy it is \*\*clearly worse than `mcsolve` on the energy \(([\d.]+)x\) "
        r"and `sx` \(([\d.]+)x\)\*\*: there the gap is larger than either method's "
        r"s\.e\.m\. On `zz` and `coherence` it is ([\d.]+)x worse, but the gaps "
        r"\(([\d.]+)×10⁻³ and ([\d.]+)×10⁻³\) are smaller than either method's "
        r"s\.e\.m\., so those two are ties within noise\. Every `mcsolve` point is "
        r"hollow \((\d+) trajectories, error ([\d.]+) to ([\d.]+) times its s\.e\.m\.\)\. "
        r"Every SLB error at \$M=(\d+)\$ is bias-limited \(filled\)", text)
    assert e, "the Control 1 accuracy sentence has changed shape"
    (q_e, q_sx, q_tie, q_gzz, q_gcoh, q_traj, q_lo, q_hi, q_m2) = e.groups()
    bundles = int(q_m)
    assert int(q_m2) == bundles
    for name, q in (("energy", q_e), ("sx", q_sx)):
        slb, mc = _p6f_scores(point, name, bundles)
        assert _near(q, slb[2] / mc[2]), name
        assert slb[2] - mc[2] > max(slb[5], mc[5]), f"{name}: the gap is not resolved"
        assert _p6g_verdict(slb, mc) == "loss", name
    for name, q_gap in (("zz", q_gzz), ("coherence", q_gcoh)):
        slb, mc = _p6f_scores(point, name, bundles)
        assert _near(q_tie, slb[2] / mc[2]), name
        assert _near(q_gap, 1000 * (slb[2] - mc[2])), name
        assert 0 < slb[2] - mc[2] < min(slb[5], mc[5]), f"{name}: the gap is resolved"
        assert _p6g_verdict(slb, mc) == "tie", name
    assert _p6f_is_rescaled_zz(point)
    zz, zzb = _p6f_scores(point, "zz", bundles), _p6f_scores(point, "zz_per_bond", bundles)
    assert math.isclose(zz[0][2] / zz[1][2], zzb[0][2] / zzb[1][2], rel_tol=1e-9)
    ratios = []
    for name in point["observables"]:
        slb, mc = _p6f_scores(point, name, bundles)
        assert mc[4] == int(q_traj)
        assert not pmc._bias_limited(mc[2], mc[5]), f"{name}: mcsolve point is filled"
        assert pmc._bias_limited(slb[2], slb[5]), f"{name}: SLB point is hollow"
        ratios.append(mc[2] / mc[5])
    assert _near(q_lo, min(ratios)) and _near(q_hi, max(ratios))


def test_p6f_dim1024_tally_counts_distinct_observables(doc):
    """At dim 1024 mcsolve's ratio is below the sqrt(2) line on every distinct
    observable (zz_per_bond is zz rescaled, so five, not six), and the smaller
    sizes' energy ratios straddle the line."""
    import plot_method_comparison as pmc
    text = _p6f_region(doc)
    m = re.search(
        r"— a ratio of ([\d.]+), and below the \$\\sqrt\{2\}\$ line on all (\w+) "
        r"distinct observables\. At smaller sizes on this system the energy point sits "
        r"either side of that line, at ratios ([\d.]+) to ([\d.]+);", text)
    assert m, "the dim-1024 below-the-line tally has changed shape"
    q_energy, q_n, q_lo, q_hi = m.groups()
    assert "independent observables" not in text
    point = _p6f_load("spin_chain_dim1024")["point"]
    assert _p6f_is_rescaled_zz(point)
    independent = [o for o in point["observables"] if o != "zz_per_bond"]
    assert _p6f_words[q_n] == len(independent)
    for name in point["observables"]:
        mc = next(r for r in pmc.method_errors(point, name) if r[0] == "mcsolve")
        assert not pmc._bias_limited(mc[2], mc[5]), name
        if name == "energy":
            assert _near(q_energy, mc[2] / mc[5])
    smaller = []
    for dim in pmc.discover_dims("spin_chain"):
        if dim >= 1024:
            continue
        p = _p6f_load(f"spin_chain_dim{dim}")["point"]
        mc = next(r for r in pmc.method_errors(p, "energy") if r[0] == "mcsolve")
        smaller.append(mc[2] / mc[5])
    assert _near(q_lo, min(smaller)) and _near(q_hi, max(smaller))
    assert min(smaller) < pmc.BIAS_LIMITED_RATIO < max(smaller)


def test_p6f_dim1024_ratio_names_its_cross_job_denominator(doc):
    """The 10-spin 3.5x divides mcsolve (job 19606788) by section 5.2's
    8-substep grid timing, which ran in another job on another node; the
    sentence must say so and quote both walls and the ratio correctly."""
    m = re.search(
        r"It took \*\*([\d,]+) s\*\*, against ([\d,]+) s for native RK4 at (\d+) "
        r"substeps at the same size on the same grid \(§5\.2\), timed in another job "
        r"on another node \(§5\.3\)\. So on the control system `mcsolve` is "
        r"\*\*([\d.]+)× slower than the exact solve\*\*", _p6f_region(doc))
    assert m, "the dim-1024 mcsolve-vs-exact sentence has changed shape"
    q_mc, q_grid, q_sub, q_ratio = m.groups()
    d10 = _p6f_load("spin_chain_dim1024")
    pgrid = DATA / "solver_timing_spin_chain.json"
    assert pgrid.exists(), "BENCHMARKS.md quotes solver_timing_spin_chain.json"
    dgrid = json.loads(pgrid.read_text(encoding="utf-8"))
    grid = {p["dim"]: p for p in dgrid["points"]}[1024]
    mc = d10["point"]["methods"]["mcsolve"]
    g10 = grid["timings"]["native"]["median_s"]
    assert int(q_sub) == grid["native_substeps"]
    assert _near(q_mc, mc["wall_s"]) and _near(q_grid, g10)
    assert _near(q_ratio, mc["wall_s"] / g10)
    e10, eg = d10["meta"]["execution"], dgrid["meta"]["execution"]
    assert e10["slurm"]["job_id"] != eg["slurm"]["job_id"]
    assert e10["hostname"] != eg["hostname"]


# --- Part 6 follow-up (p6g): one gap rule, and the text outside Result 3 ------
#
# Result 3 scored wins and losses by two standards: System A called a gap
# smaller than the s.e.m.s a tie, System B counted such gaps as wins or losses.
# The document now states one rule, once. The walls table in section 5.2, the
# seed-robustness note in section 6, section 3.3's sampling table and the
# plotter's M=1 comment had drifted from Result 3; these pin them.

def _p6g_verdict(slb_row, mc_row) -> str:
    """Result 3's rule on two method_errors rows: a gap between the errors is
    a win or a loss for SLB only when it is larger than the larger of the two
    s.e.m.s; otherwise it is a tie within noise."""
    gap = slb_row[2] - mc_row[2]
    if abs(gap) <= max(slb_row[5], mc_row[5]):
        return "tie"
    return "loss" if gap > 0 else "win"


def _p6g_result3(doc: str) -> str:
    """All of Result 3, flattened."""
    text = _flat(doc)
    start = text.index("### Result 3 — accuracy versus cost: SLB against mcsolve")
    return text[start:text.index("### Result 4", start)]


def _p6g_mcsolve_ratios() -> list:
    """error/s.e.m. of every mcsolve point in every Result 3 file, distinct
    observables only, through method_errors."""
    ratios = []
    for system, dim in _p6c_files():
        point = _p6c_point(system, dim)["point"]
        for obs in point["observables"]:
            if obs == "zz_per_bond":
                continue
            scored = _p6c_mcsolve(point, obs)
            if scored is not None:
                ratios.append(scored[0] / scored[1])
    assert ratios, "no Result 3 file ran mcsolve"
    return ratios


def test_p6g_result3_states_the_gap_rule_once(doc):
    """The win/loss/tie rule is stated exactly once, next to the paragraph
    that explains mcsolve's noise, and every verdict test uses _p6g_verdict."""
    rule = (r"\*\*So a gap between the two errors counts as a win or a loss only if it "
            r"is larger than the larger of the two s\.e\.m\.s; a smaller gap is a tie "
            r"within noise\.\*\*")
    assert len(re.findall(rule, _p6g_result3(doc))) == 1, "state the gap rule once"
    assert re.search(rule, _flat(_p6c_region(doc))), (
        "the gap rule no longer sits with the mcsolve-noise paragraph")
    # the helper is the rule as written: ties at and below the larger s.e.m.
    assert _p6g_verdict((0, 0, 1.0, 0, 0, 0.5), (0, 0, 1.4, 0, 0, 0.4)) == "tie"
    assert _p6g_verdict((0, 0, 1.0, 0, 0, 0.1), (0, 0, 1.4, 0, 0, 0.3)) == "win"
    assert _p6g_verdict((0, 0, 1.4, 0, 0, 0.3), (0, 0, 1.0, 0, 0, 0.1)) == "loss"


def test_p6g_sampling_table_result3_row_matches_the_sweep(doc):
    """Section 3.3's sampling table: Result 3's M range is the files' sweep,
    the figures' first M is the plotter's, and the realization count holds."""
    import plot_method_comparison as pmc
    m = re.search(r"^\| four-method comparison \(Result 3\) \| (\d+)–(\d+) \(swept; "
                  r"figures from (\d+)\) \| (\d+) \| S/√N_r \(SEM\) \|$", doc, re.M)
    assert m, "section 3.3's Result 3 sampling row has changed shape"
    low, high, first, n_runs = map(int, m.groups())
    slb = [d["point"]["methods"]["slb"] for d in _p6a_files().values()
           if d["point"]["methods"].get("slb")]
    swept = [int(r["M"]) for rows in slb for r in rows]
    assert (min(swept), max(swept)) == (low, high)
    assert first == pmc.MIN_M_PLOTTED
    assert all(int(r["n_runs"]) == n_runs for rows in slb for r in rows)


def test_p6g_section52_mcsolve_wall_reads_filled_as_noise(doc):
    """The walls table's mcsolve row: its error/s.e.m. range over every
    Result 3 point, and a point past the sqrt(2) line read as noise."""
    import plot_method_comparison as pmc
    row = next((line for line in doc.splitlines()
                if line.startswith("| **QuTiP `mcsolve`**")), None)
    assert row, "section 5.2's mcsolve wall row is missing"
    m = re.search(r"\(error/s\.e\.m\. ([\d.]+) to ([\d.]+) across observables\), so some "
                  r"points land past Result 3's \$\\sqrt\{2\}\$ line, which for an unbiased "
                  r"method is noise, not bias;", row)
    assert m, "section 5.2's mcsolve wall reading has changed shape"
    ratios = _p6g_mcsolve_ratios()
    assert _near(m.group(1), min(ratios)) and _near(m.group(2), max(ratios))
    assert min(ratios) < pmc.BIAS_LIMITED_RATIO < max(ratios), "'some points' past the line"
    assert "bias-limited by Result 3" not in row


def test_p6g_section6_seed_sentence_uses_the_gap_rule(doc):
    """Section 6 describes Result 3's System A comparison with the same
    verdicts Result 3 gives: energy and sx losses, zz and coherence ties."""
    m = re.search(r"Result 3's System A comparison at dimension (\d+) \(\$M=(\d+)\$ against "
                  r"\$\\texttt\{ntraj\}=(\d+)\$\), where SLB is clearly worse on the energy "
                  r"and `sx` and ties within noise on `zz` and `coherence`\.", _flat(doc))
    assert m, "section 6's seed-robustness sentence has changed shape"
    dim, bundles, ntraj = map(int, m.groups())
    point = _p6f_load(f"spin_chain_dim{dim}")["point"]
    verdict = {}
    for name in point["observables"]:
        if name == "zz_per_bond":
            continue
        slb, mc = _p6f_scores(point, name, bundles)
        assert mc[4] == ntraj, name
        verdict[name] = _p6g_verdict(slb, mc)
    assert verdict == {"energy": "loss", "sx": "loss", "zz": "tie", "coherence": "tie"}, verdict
    assert "loses on every observable" not in _flat(doc)


def test_p6g_plotter_m1_comment_matches_the_data():
    """The comment above MIN_M_PLOTTED: how many dimensions timed M=1 slower
    than M=2, and how many curves fall from M=1 to M=2, with every rise under
    one s.e.m. -- recomputed as Result 3's M=1 paragraph is."""
    import plot_method_comparison as pmc
    source = (BENCHMARKS / "plot_method_comparison.py").read_text(encoding="utf-8")
    block = source[:source.index("\nMIN_M_PLOTTED = ")]
    comment = re.sub(r"\s*\n#\s*", " ", block[block.rindex("# M=1 is one bundle"):])
    assert "monotone" not in comment, "the plotter comment says accuracy is monotone in M"
    m = re.search(r"It also timed slower than M=2 at (\d+) of the (\d+) dimensions despite "
                  r"doing less work", comment)
    f = re.search(r"The error falls from M=1 to M=2 on (\d+) of the (\d+) curves, and every "
                  r"rise in M, those two included, is smaller than one s\.e\.m\.", comment)
    assert m and f, "the plotter's M=1 comment has changed shape"
    slower = total = 0
    for document in _p6a_files().values():
        slb = {int(r["M"]): r for r in document["point"]["methods"].get("slb", [])}
        if slb:
            total += 1
            slower += slb[1]["wall_s"] > slb[2]["wall_s"]
    assert (int(m.group(1)), int(m.group(2))) == (slower, total)
    curves = _p6a_curves(1)
    falls = sum(rows[1][2] < rows[0][2] for rows in curves.values())
    assert (int(f.group(1)), int(f.group(2))) == (falls, len(curves))
    rises = [(b[2] - a[2]) / b[5] for rows in curves.values()
             for a, b in zip(rows, rows[1:]) if b[2] >= a[2]]
    assert max(rises) < 1, "a rise now exceeds the s.e.m. of the point it reaches"
    assert pmc.MIN_M_PLOTTED == 2
