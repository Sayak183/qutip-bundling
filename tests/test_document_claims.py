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
