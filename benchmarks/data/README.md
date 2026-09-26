# Benchmark data manifest

The JSON files directly in this directory are the current machine-readable
sources of truth. Plotters and the default `python export_csv.py` command read
only these top-level files.

## Canonical JSON

| Result | Canonical files |
|---|---|
| 1. Accuracy versus bundle size | `accuracy_vs_M_<system>_dim<D>.json` |
| 2. Cost scaling | `cost_scaling_<system>.json` |
| 3. Method comparison | `method_comparison_<system>_dim<D>.json` |
| 4. Iso-accuracy cost | `isocost_vs_dim_<system>.json` |

The dimension suffix is required for Results 1 and 3 because each dimension is
an independent run with its own solver provenance and numerical checks.

`csv/` holds derived, Excel-friendly exports, all stale for now (see below). CSV
is a convenience view; JSON remains authoritative because it retains complete
metadata and raw samples.

Three top-level files are not current: `frontier_oscillator_bath_dim16.json`,
`_dim32.json` and `_dim64.json` were written on Jul 18, before 0.6.4. They
record no `degeneracy_tol`, package version or Slurm job, and at dims 32 and 64
they hold 478 and 1,172 operators where the shipped code builds 408 and 890.
No Result reads them; only the superseded `plot_frontier.py` and the CSV export
do. Every file in `csv/` dates from 29 July, before 0.6.4, so all of them are
stale until `python export_csv.py` is re-run, and it has no exporter for
`method_comparison_*` yet.

## High-dimensional reference artifacts

`high_dim_reference_spin_chain_dim<D>.json` records an isolated native-RK4
reference and its full-state convergence certification. Its companion
`high_dim_reference_spin_chain_dim<D>.npz` stores the reusable density-matrix
trajectory. The JSON contains the NPZ filename, shape, dtype, and SHA-256
checksum.

NPZ archives are ignored by Git because the dimension-512 file can exceed
GitHub's normal file limit. Preserve them in persistent research storage.

## Legacy data

`legacy/` preserves superseded or incomplete runs for provenance. Its
unsuffixed `accuracy_vs_M_*` and old-frontier `frontier_*` files predate the
per-dimension format and lack
newer reference-solver metadata. The `.PARTIAL.json` file is an interrupted
Result 2 run. Neither is selected by current plotters or the default CSV export.
