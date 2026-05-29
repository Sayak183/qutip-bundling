# Changelog

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
