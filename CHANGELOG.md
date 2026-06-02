# Changelog

## Unreleased
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
- Benchmark scripts now expose the same Davies-construction thresholds through
  environment variables (`DAVIES_THRESHOLD`, `COUPLING_THRESHOLD`, and
  `LAMB_SHIFT_THRESHOLD`) and report retained/full `N_L` counts.

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
