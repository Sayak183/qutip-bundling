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
- [`archive/`](archive/) contains redacted historical transcripts, internal
  patch notes, and superseded one-off inspection scripts.
- Extra frontier PNGs ending in `_perUnit` or `_noEns_Single` are alternate
  views controlled by toggles in `plot_frontier.py`; the unsuffixed PNGs are
  the main figures embedded in [BENCHMARKS.md](BENCHMARKS.md).
