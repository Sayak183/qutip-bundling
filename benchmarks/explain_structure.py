r"""
explain_structure.py
====================

The two checks from BENCHMARKS.md section 1, run on the benchmark systems and
printed as numbers you can read -- and, more usefully, as a template for
running the same checks on your own model.

It answers three questions about a Hamiltonian and a coupling operator:

  1. How many collapse operators are there (N_L), and why that many? The gap
     census shows how the d^2 ordered level pairs collapse first by symmetry
     (pairs X cannot connect at all) and then by frequency degeneracy (distinct
     pairs sharing one Bohr frequency, which the bath cannot tell apart).

  2. Where does the dissipation sit in the energy eigenbasis? An ASCII map of
     the total transition weight, plus \bar d, the strength-weighted mean
     distance from the diagonal. A ladder gives \bar d ~ 1; an operator spread
     across the spectrum gives \bar d ~ dim/3.

  3. For the chains: a direct test of whether the 2^n many-body energies are
     nothing more than the subset sums of n single-particle energies. That is
     what "free fermions" means concretely, and it is the reason the gaps
     repeat. The integrable chain passes exactly; the mixed-field chain fails.

Run:  python explain_structure.py [--systems ...] [--n-sites N] [--show K]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from common import (
    build_spin_chain,
    build_mixed_field_chain,
    build_oscillator_bath,
    build_davies_operators,
)

# Numerical tolerances. GAP_TOL is deliberately looser than machine epsilon:
# two Bohr frequencies that agree to 1e-9 are one bath channel in practice, and
# it is the same order the Davies construction itself uses to group them.
GAP_TOL = 1e-9
ZERO_TOL = 1e-10
FIT_TOL = 1e-7      # two excitation energies this close are the same level

SHADES = " .:-=+*#%@"


def count_distinct(values, tol=GAP_TOL):
    """How many distinct values, treating anything closer than `tol` as one."""
    distinct = 0
    previous = None
    for value in np.sort(np.asarray(values, dtype=float)):
        if previous is None or abs(value - previous) > tol:
            distinct += 1
            previous = value
    return distinct


def gap_census(H, X):
    """Where the d^2 level pairs go: connected by X, then merged by frequency.

    Returns (n_pairs, n_distinct_gaps, n_connected_pairs, n_connected_gaps).
    The last number is N_L.
    """
    energies, vectors = np.linalg.eigh(H.full())
    coupling = vectors.conj().T @ X.full() @ vectors
    all_gaps, connected_gaps = [], []
    for a in range(len(energies)):
        for b in range(len(energies)):
            gap = energies[b] - energies[a]
            all_gaps.append(gap)
            if abs(coupling[a, b]) > ZERO_TOL:
                connected_gaps.append(gap)
    return (len(all_gaps), count_distinct(all_gaps),
            len(connected_gaps), count_distinct(connected_gaps))


def transition_weight(H, c_ops):
    """Total |<a|L|b>|^2 over all collapse operators, in the energy eigenbasis.

    One matrix summarising where the dissipation acts: entry (a, b) is how
    strongly the bath moves population and coherence between energy levels a
    and b, regardless of which collapse operator does it.
    """
    _, vectors = np.linalg.eigh(H.full())
    weight = np.zeros((H.shape[0], H.shape[0]))
    for op in c_ops:
        weight += np.abs(vectors.conj().T @ op.full() @ vectors) ** 2
    return weight


def mean_offdiagonal_distance(weight):
    r"""\bar d: mean |a - b| weighted by transition strength.

    The locality number from section 1. Small means the bath walks the system
    down the ladder one rung at a time; large means it connects distant levels.
    """
    index = np.arange(weight.shape[0])
    distance = np.abs(index[:, None] - index[None, :])
    total = weight.sum()
    return float((weight * distance).sum() / total) if total > 0 else 0.0


def render(weight, show):
    """An ASCII map of the transition weight, brightest entry = '@'."""
    block = weight[:show, :show]
    peak = block.max()
    if peak <= 0:
        return ["(no transitions)"]
    rows = []
    for row in block / peak:
        rows.append("".join(
            SHADES[min(len(SHADES) - 1, int(np.ceil(v * (len(SHADES) - 1))))]
            if v > 1e-12 else " "
            for v in row
        ))
    return rows


def free_fermion_check(H, n_sites):
    r"""Test whether the 2^n energies are all subset sums of n numbers.

    Measure every energy from the ground state. Walk up the resulting list of
    excitation energies; whenever one is NOT already a sum of modes found so
    far, it is a new mode. If the model really is free, exactly ``n_sites``
    modes appear and every remaining level is accounted for -- so the 2^n
    many-body energies carry no more information than n numbers, which is what
    makes the gaps repeat. No fitting is involved, and nothing is assumed: if
    the model is not free the search simply runs out of levels or the rebuilt
    spectrum fails to match.

    Returns (eps, max_deviation); a deviation at machine precision means the
    decomposition is exact.
    """
    excitation = np.sort(np.real(H.eigenenergies()))
    excitation = excitation - excitation[0]

    eps, reachable = [], [0.0]
    for value in excitation[1:]:
        if min(abs(value - r) for r in reachable) < FIT_TOL:
            continue                      # already a sum of known modes
        eps.append(value)
        reachable = sorted({round(r + s, 9)
                            for r in reachable for s in (0.0, value)})
        if len(eps) == n_sites:
            break

    if len(eps) < n_sites:
        return np.array(eps), float("inf")
    rebuilt = np.sort([sum(choice) for choice in
                       itertools.product(*[(0.0, e) for e in eps])])
    if len(rebuilt) != len(excitation):
        return np.array(eps), float("inf")
    return np.array(eps), float(np.abs(rebuilt - excitation).max())


def report(label, H, X, show, n_sites=None):
    dim = H.shape[0]
    c_ops = build_davies_operators(H, X)
    pairs, gaps, connected, n_l = gap_census(H, X)
    weight = transition_weight(H, c_ops)
    dbar = mean_offdiagonal_distance(weight)

    print(f"\n{'=' * 72}\n{label}   dim = {dim}\n{'=' * 72}")
    print(f"  {pairs:>7d} ordered level pairs (dim^2)")
    print(f"  {connected:>7d} of them the coupling operator can connect at all"
          f"   ({100 * connected / pairs:.0f}%)")
    print(f"  {gaps:>7d} distinct Bohr frequencies among all pairs")
    print(f"  {n_l:>7d} distinct Bohr frequencies among the connected pairs"
          f"  <-- N_L")
    assert n_l == len(c_ops), (n_l, len(c_ops))
    print(f"  compression from grouping: {connected} transitions -> {n_l} "
          f"collapse operators ({connected / max(n_l, 1):.1f} per operator)")

    print(f"\n  transition weight in the energy eigenbasis "
          f"(lowest {min(show, dim)} levels, '@' = strongest):")
    for row in render(weight, show):
        print("    " + row)
    print(f"\n  mean distance from the diagonal  d_bar = {dbar:.2f} levels "
          f"(dim - 1 = {dim - 1} would be maximal)")

    if n_sites is not None:
        eps, deviation = free_fermion_check(H, n_sites)
        print(f"\n  free-particle test: are all {dim} energies subset sums "
              f"of {n_sites} numbers?")
        print("    modes found, eps_k = " + np.array2string(eps, precision=4))
        if deviation < 1e-8:
            print(f"    every level reproduced, largest error {deviation:.1e}"
                  f"   <-- YES: the spectrum is {n_sites} numbers, not {dim}")
        else:
            print("    the decomposition fails"
                  f"   <-- NO: genuinely {dim} independent numbers")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--systems", nargs="+",
                        choices=["A", "B", "C"], default=["A", "B", "C"],
                        help="A = integrable chain, B = mixed-field chain, "
                             "C = oscillator + spin (default: all three)")
    parser.add_argument("--n-sites", type=int, default=4,
                        help="spins in the chains (default: 4, i.e. dim 16)")
    parser.add_argument("--osc-levels", type=int, default=8,
                        help="oscillator levels; dim is twice this (default: 8)")
    parser.add_argument("--show", type=int, default=16,
                        help="levels to draw in the ASCII map (default: 16)")
    args = parser.parse_args()

    if "A" in args.systems:
        H, X, _ = build_spin_chain(args.n_sites, g=0.0)
        report(f"System A: integrable Ising chain, n={args.n_sites}, "
               f"X = sum_i sigma^x_i", H, X, args.show, n_sites=args.n_sites)
    if "B" in args.systems:
        H, X, _ = build_mixed_field_chain(args.n_sites)
        report(f"System B: mixed-field Ising chain, n={args.n_sites}, "
               f"X = sum_i sigma^x_i", H, X, args.show, n_sites=args.n_sites)
    if "C" in args.systems:
        H, X, _ = build_oscillator_bath(args.osc_levels)
        report(f"System C: anharmonic oscillator + spin, "
               f"{args.osc_levels} levels, X = x (x) I", H, X, args.show)


if __name__ == "__main__":
    main()
