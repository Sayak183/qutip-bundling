# Changelog

## Unreleased
- **Benchmarks (result change):** a four-method comparison run in one Slurm
  allocation (native RK4, `mesolve`, `mcsolve`, SLB; identical Hamiltonian,
  Davies construction, initial state, time grid, and observables) shows the
  bundling advantage is governed by `N_L`, not by Hilbert dimension. On the
  oscillator (`N_L=408` at dimension 32) SLB reaches `1e-3` error ~27x cheaper
  than the exact full-dissipator solve and ~300x cheaper than `mcsolve` at 500
  trajectories, while being ~100x more accurate than it. On the spin chain
  (`N_L=43` at dimension 128) it is **not** competitive: the exact solve costs
  about the same as the cheapest useful bundle and is far more accurate. This
  holds for single realizations as well as 16-run ensembles.
  The previous README claim that bundling on the spin chain "is never slower
  and typically a few-fold faster" than `mcsolve` rested on pre-0.6.4 data,
  where that system's `N_L` was inflated to 113 and 325 by the roundoff-only
  projector blocks the 0.6.4 floor removes. The chain is now documented as a
  control for where the method does not help.
- **Performance:** `bundle_from_phases` builds its `M` bundles with a single
  BLAS matrix product instead of a Python loop of `M * N_L` Qobj additions. The
  operators are stacked once into an `(N_L, N*N)` array and every bundle comes
  from one `(M, N_L) @ (N_L, N*N)` product. Output is identical to the explicit
  summation (max deviation `8e-16`, machine epsilon), pinned by 17 new
  regression tests. Measured on the spin chain at `M=8`: 3.9x at dim 16, 2.5x
  at dim 32, 1.5x at dim 64 — the win is loop overhead, so it shrinks as
  genuine arithmetic starts to dominate.
- **Benchmarks:** the `run_*.py` data generators accept `--max-full-dim` to
  raise the exact-`mesolve` dimension cap, which was previously a module
  constant with no override; a run on a larger machine silently recorded no
  exact reference above dimension 32. The effective value is stamped into the
  run metadata.
- **Benchmarks:** run metadata now records an `execution` block — hostname, CPU
  count, thread environment variables, and the Slurm job, nodelist, and
  partition when present. Wall-clock times are only comparable within one job
  on one node, and this makes that checkable from the data instead of assumed.
- **Benchmarks:** added `build_high_dim_sheet.py`, which regenerates the
  `High-dim Ref` sheet of `benchmark_results.xlsx` from the certified reference
  JSON files, and `plot_high_dim_spin_reference.py`, which draws the certified
  references and their full-state convergence check. Both default to every
  dimension present in `data/` rather than a hard-coded list.

## 0.6.4 — 2026-07-29
- **Fixed:** `davies_operators` now discards projector blocks below a
  scale-covariant backward-error floor,
  `512 * eps * dim * ||X||_F` in the energy eigenbasis, instead of an absolute
  cutoff. A fixed floor was not reproducible: different LAPACK builds place
  symmetry-forbidden zeros on either side of it, so the same physical system
  produced different numbers of Davies operators on different machines.
- **The dissipator is unchanged to double precision.** Against a reference that
  keeps every numerically nonzero block, the floored construction differs by a
  relative Frobenius norm of `1e-24` to `1e-33` in the full dissipator
  superoperator, on both benchmark systems at every dimension tested — eight or
  more orders of magnitude below machine epsilon. Applied to a density matrix
  the difference is *exactly* zero: the perturbation is far too small to change
  any floating-point sum. The discarded blocks have Frobenius norm ~`1e-11` or
  below against a smallest retained block of ~`3e-2` (nine orders of
  separation) and enter the dissipator quadratically. Accuracy-type results are
  therefore unaffected.
- **Behavior change:** operator *counts* drop substantially, because those
  blocks were being promoted into whole spurious frequency sectors. For the
  benchmark spin chain `N_L` is now exactly `n**2 - n + 1` in the number of
  sites (13, 21, 31, 43, 57 at dimensions 16-256, against 15, 41, 113, 325, 839
  before); for the oscillator it goes 128, 408, 890 at dimensions 16/32/64
  against 128, 478, 1172.
- **Benchmarks:** any measurement whose cost depends on `N_L` must be
  regenerated — the exact-solver time `t_full` is roughly linear in the number
  of collapse operators, so cost scaling (Result 2), iso-accuracy cost
  (Result 4), and the cost axis of the frontier (Result 3) were measured
  against an inflated baseline. Accuracy-versus-`M`, convergence, jackknife,
  and seed-robustness data remain valid by the bit-identity above.
- `coupling_threshold` is still applied as `max(coupling_threshold,
  numerical_floor)`, so it can only prune more aggressively than the automatic
  floor, never less.

## 0.6.3 — 2026-07-28
- **Fixed:** `davies_operators` now implements the strict secular construction
  for degenerate systems: it forms energy-eigenspace projectors and returns one
  summed `A(omega)` per Bohr-frequency sector. The previous eigenstate-pair
  decomposition dropped cross terms and depended on the numerical basis chosen
  inside degenerate subspaces.
- Added `degeneracy_tol` (default `1e-10`) for numerical energy/frequency
  equality, with regression tests for basis covariance, Gibbs stationarity,
  trace preservation, a harmonic ladder, and the benchmark spin chain.
- **Benchmarks:** future JSON metadata records the grouped construction and its
  tolerance. Existing spin-chain results generated with 0.6.2 must be
  regenerated; the anharmonic oscillator dissipator is unchanged to roundoff.

## 0.6.2 — 2026-06-26
- **Changed:** the native RK4 backend (`rk4_mesolve`) now raises
  `SolverInstabilityError` when the integration diverges to a non-finite
  state, instead of silently returning a NaN-filled result. Under-resolved
  stiff systems (e.g. the anharmonic oscillator at a large Fock cutoff) now
  fail loudly with an actionable message (increase `substeps`).
  `SolverInstabilityError` is exported from `qutip_bundling`.
- **Benchmarks:** replaced the combined cost+error scaling figure (Result 1)
  with a dedicated cost-scaling figure (`benchmark_cost_scaling.py`) that
  compares SLB against the exact solver only (measured large-$N$ slopes
  ~$N^2$ for SLB vs ~$N^5$ for full `mesolve`). The SLB-vs-`mcsolve`
  comparison is the accuracy-cost frontier (Result 3), where the relevant
  axis is accuracy-per-cost rather than raw scaling.
- **Docs:** fixed the cost-scaling figure link in the README (it pointed
  at the pre-`bc74ccb` filename `benchmark_scaling_spin_chain.png`, now
  `benchmark_cost_scaling_spin_chain.png`).

## 0.6.1 — 2026-06-22
- **Fixed** the Bohr-frequency sign in `examples/oscillator_demo.py`: it
  used `E_n - E_m` for `L = |n><m|`, the opposite of the package
  convention (`E_m - E_n`) documented in `CONVENTIONS.md` and enforced by
  `davies_operators`. The demo's checks pass either way (they test
  bundling fidelity, not relaxation direction), but the example now
  matches the convention so it is safe to use as a template.
- **Docs:** `CONVENTIONS.md` now documents the pairwise-vs-grouped
  operator choice for degenerate Bohr-frequency sectors (the strict
  secular Davies construction groups transitions by frequency; this
  package builds one operator per eigenstate pair, which agrees for
  non-degenerate spectra).
- **Docs:** fixed README rendering — the API table no longer embeds
  display math in cells (a malformed brace in the Lamb-shift entry is
  gone), and a stray Sphinx `:func:` role was replaced with plain
  Markdown so it renders on GitHub and PyPI.
- **Packaging:** added `MANIFEST.in` so the source distribution now
  includes `CITATION.cff`, `CHANGELOG.md`, `CONVENTIONS.md`,
  `PUBLISHING.md`, `examples/`, and the benchmark scripts. No code or API
  changes.

## 0.6.0 — 2026-06-15
- `mesolve_ensemble` / `mesolve_jackknife` now raise `ValueError` when
  `options` is passed together with `backend="native"`, instead of silently
  ignoring it. The native RK4 backend does not consume qutip integrator
  `options`; use `substeps=` to set its resolution. Only affects callers that
  previously passed `options` with the native backend (where it had no effect).
- The native RK4 backend (`rk4_mesolve`) is now covered by tests asserting
  agreement with `qutip.mesolve` and trace preservation.
- **Behavior change:** `build_collapse_ops` now applies `threshold` to the
  full collapse-operator weight `sqrt(gamma) * ||L||`, not to `sqrt(gamma)`
  alone. This makes `threshold` mean the same thing as in
  `davies_operators` (rate times coupling strength) and matches the single
  sparsity scalar of the StochLind C++ reference. Behavior is unchanged for
  `threshold=0.0` (the default) and for the rate=0 drop; only nonzero
  thresholds applied to operators of differing norm are affected.
- `davies_operators` gained two opt-in keyword arguments, both defaulting to
  current behavior:
  - `coupling_threshold` (default `0.0`): prunes Bohr pairs whose bare
    coupling element `|<a|X|b>|` is below the cutoff *before* the spectral
    function is evaluated or any operator is built. Because the coupling
    operator is typically sparse in the energy eigenbasis, the build now
    iterates only the significant entries (O(nnz) instead of O(N**2)),
    mirroring the first sparsity cut in the StochLind C++ reference. With
    the default `0.0` the surviving operator set is bit-for-bit identical to
    before.
  - `lamb_shift_threshold` (default `None` = inherit `threshold`): lets the
    Lamb-shift term use its own cutoff. Previously an aggressive operator
    `threshold` could silently drop Lamb-shift terms, because that filter
    compares against `|imag_gamma|`, an unrelated quantity. Set this
    explicitly to decouple the two.

## 0.5.0
- `mesolve_ensemble` / `mesolve_jackknife` now expose `.std` (trajectory
  standard deviation) and `.samples` (raw per-realization data) in addition
  to `.expect` and `.sem`.
- Jackknife de-emphasized in docs as an optional advanced step.
- Documented that bundling accepts arbitrary collapse operators (not only
  Davies) and arbitrary spectral functions (not only ohmic).

## 0.4.0
- Added `davies_operators(H, X, gamma)`: builds Davies/Bohr collapse
  operators directly from a Hamiltonian and coupling operator, with the
  correct Bohr-frequency sign convention baked in.
- Added `CONVENTIONS.md` documenting the sign convention, detailed balance,
  operator scaling, and the Lamb shift.

## 0.3.0
- Added `rk4_mesolve`, a native RK4 propagator that avoids building the
  Liouvillian superoperator, enabling large Hilbert spaces.
- `backend="native"` option on the ensemble/jackknife solvers.

## 0.2.0
- Refactored into pure operator transforms: `build_collapse_ops`, `bundle`,
  `lamb_shift_hamiltonian`, and the `prepare_bundled_dynamics` composer.
- Lamb shift built from bare operators, never bundled.

## 0.1.0
- Initial implementation of stochastic Lindblad operator bundling.
