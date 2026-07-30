# Benchmark development archive

Nothing in this directory is required to reproduce the published figures.

- `run-logs/` contains historical console transcripts and timing notes. Local
  usernames and checkout paths have been redacted.
- `dev-tools/` contains small one-off inspection scripts superseded by the
  maintained benchmark and plotting entry points.
- `PATCH_NOTES.txt` records an earlier internal rerun procedure and is not
  current user documentation.
- `PATCH_NOTES_dim64_prose.txt` and `FIX_NOTES_v8.txt` are the equivalent
  internal notes for the dimension-64 prose update and the substeps-4 fix kit.
  Both were previously kept at the repository root; they describe one-off
  local procedures and are superseded by the maintained runner scripts.
- `PATCH_NOTES_bundle_vectorization.txt` records the measured speedups behind
  the BLAS rewrite of `bundle_from_phases`, kept for the timings; the change
  itself now lives in `src/` with its own regression tests.

The archive is kept for provenance. New users should start at
[`../README.md`](../README.md).
