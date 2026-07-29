# Benchmark data manifest

The JSON files directly in this directory are the current machine-readable
sources of truth. Plotters and the default `python export_csv.py` command read
only these top-level files.

## Canonical JSON

| Result | Canonical files |
|---|---|
| 1. Accuracy versus bundle size | `accuracy_vs_M_<system>_dim<D>.json` |
| 2. Cost scaling | `cost_scaling_<system>.json` |
| 3. Accuracy/cost frontier | `frontier_<system>_dim<D>.json` |
| 4. Iso-accuracy cost | `isocost_vs_dim_<system>.json` |

The dimension suffix is required for Results 1 and 3 because each dimension is
an independent run with its own solver provenance and numerical checks.

`csv/` contains derived, Excel-friendly exports of these canonical files. CSV
is a convenience view; JSON remains authoritative because it retains complete
metadata and raw samples.

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
unsuffixed Result 1 and Result 3 files predate the per-dimension format and lack
newer reference-solver metadata. The `.PARTIAL.json` file is an interrupted
Result 2 run. Neither is selected by current plotters or the default CSV export.
