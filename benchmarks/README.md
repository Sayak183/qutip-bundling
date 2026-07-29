# Benchmark quick start

This directory contains the scripts, saved data, and figures for the benchmark
study. Start here to reproduce a figure safely; read [BENCHMARKS.md](BENCHMARKS.md)
for the systems, methodology, and interpretation of every result.

## Replot one result without rerunning a benchmark

From the repository root:

```bash
python -m pip install -e ".[examples]"
cd benchmarks
python plot_accuracy_vs_M.py --system spin_chain
```

The last command reads the committed `data/*.json` files and regenerates the
spin-chain Result 1 PNGs. It does **not** run the numerical benchmark or modify
the saved JSON.

Run benchmark commands from this directory. Several plotters intentionally
resolve `data/` relative to the current working directory.

## Replot all four main results

These commands only analyze the existing JSON data and normally finish in
seconds:

```bash
python plot_accuracy_vs_M.py
python plot_cost_scaling.py
python plot_frontier.py
python plot_isocost_vs_dim.py
```

They regenerate the corresponding tracked PNG files in this directory.

| Result | Data generator | Plotter | Saved data |
|---|---|---|---|
| 1. Accuracy versus bundle size | `run_accuracy_vs_M.py` | `plot_accuracy_vs_M.py` | `data/accuracy_vs_M_<system>_dim<D>.json` |
| 2. Cost scaling | `run_cost_scaling.py` | `plot_cost_scaling.py` | `data/cost_scaling_<system>.json` |
| 3. Accuracy/cost frontier | `run_frontier.py` | `plot_frontier.py` | `data/frontier_<system>_dim<D>.json` |
| 4. Iso-accuracy cost | `run_isocost_vs_dim.py` | `plot_isocost_vs_dim.py` | `data/isocost_vs_dim_<system>.json` |

## Before generating new data

The `run_*.py` scripts perform the expensive simulations. They require an
explicit `--system SYSTEM` or `--all`; running one with no scope is an error.
They also refuse to replace an existing JSON file unless `--overwrite` is
present. Some configurations take hours or require workstation-scale memory,
so use `--dry-run` to inspect the plan and work on a separate Git branch before
intentional replacement.

Inspect the supported filters before starting:

```bash
python run_accuracy_vs_M.py --help
python run_cost_scaling.py --help
python run_frontier.py --help
python run_isocost_vs_dim.py --help
```

For example, preview only the spin-chain, dimension-64 frontier with:

```bash
python run_frontier.py --system spin_chain --dims 64 --dry-run
```

To actually replace its existing canonical JSON after reviewing that plan:

```bash
python run_frontier.py --system spin_chain --dims 64 --overwrite
```

The real command is intentionally **not** a quick test: it is a heavy
workstation run. There is no `--preset` option.

## High-dimensional native references

`run_high_dim_spin_reference.py` extends the deterministic spin-chain
reference beyond the practical `qutip.mesolve` wall. It saves two companion
files:

- `data/high_dim_reference_spin_chain_dim<D>.json` contains provenance,
  timings, energy, state diagnostics, and the convergence certification.
- `data/high_dim_reference_spin_chain_dim<D>.npz` contains `times` and the
  complete complex density-matrix array `states` with shape
  `(n_times, D, D)`. The JSON records its SHA-256 checksum.

The primary and comparison integrations are certified using both the energy
difference and the trace distance between their full density matrices. NPZ
archives are intentionally ignored by Git because large dimensions exceed
normal repository file limits; copy them to persistent research storage.

On a direct-SSH server using `tcsh`, prepare a separate checkout and start a
dimension-256 run that survives logout with:

```tcsh
git clone https://github.com/Sayak183/qutip-bundling.git qutip-bundling-`hostname -s`
cd qutip-bundling-`hostname -s`
python3 -m venv .venv
source .venv/bin/activate.csh
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
python benchmarks/probe_high_dim_spin.py --dims 128 256 512
mkdir -p logs
nohup .venv/bin/python benchmarks/run_high_dim_spin_reference.py --dim 256 \
    >&! logs/dim256.log &
```

Monitor without stopping the calculation:

```tcsh
tail -f logs/dim256.log
```

The default 8 GiB preflight limit admits dimension 256 and deliberately
refuses dimension 512. Raise `--max-core-gib` only after reviewing the probe
on a machine with sufficient dedicated RAM.

## Other files

- [`data/README.md`](data/README.md) is the canonical-data manifest. Current
  JSON stays at the top of `data/`; superseded and partial runs are preserved
  under `data/legacy/`.
- `common.py` contains shared system definitions, grids, metrics, metadata, and
  JSON helpers.
- `benchmark_cli.py` provides the shared explicit-scope, dry-run, and overwrite
  safety controls used by every `run_*.py` command.
- `benchmark_*.py` scripts produce the validation and robustness figures
  discussed in section 6 of [BENCHMARKS.md](BENCHMARKS.md).
- `convergence_progress_*.json` files are saved inputs for those validation
  plots, not stray temporary files.
- `export_csv.py` converts saved JSON results to tidy files under `data/csv/`.
- [`benchmark_results.xlsx`](benchmark_results.xlsx) combines all canonical
  Result 1-4 JSON/CSV outputs into formatted summary and dynamics sheets with
  provenance, charts, and the dimension-64 spin-chain energy trace.
- [`archive/`](archive/) contains redacted historical transcripts, internal
  patch notes, and superseded one-off inspection scripts.
- Extra frontier PNGs ending in `_perUnit` or `_noEns_Single` are alternate
  views controlled by toggles in `plot_frontier.py`; the unsuffixed PNGs are
  the main figures embedded in [BENCHMARKS.md](BENCHMARKS.md).
