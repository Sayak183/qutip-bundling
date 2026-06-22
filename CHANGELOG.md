# Changelog

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
