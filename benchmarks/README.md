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
| Four-method comparison | `run_method_comparison.py` | (plotter pending) | `data/method_comparison_<system>_dim<D>.json` |

The four-method comparison puts native RK4, `mesolve`, `mcsolve`, and SLB on
one footing at each dimension: same Hamiltonian, Davies construction, initial
state, time grid, and observables, all scored against the same certified
reference. `mcsolve` runs at a fixed `--ntraj` budget and reports whatever
accuracy that buys. Because it compares wall-clock times across methods, every
dimension must run inside a single job on a single node — see the caveat under
"Running these on a cluster" below.

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

The committed dimension-128 and dimension-256 JSON artifacts can be plotted
together with:

```bash
python benchmarks/plot_high_dim_spin_reference.py
```

The same data are summarized on the `High-dim Ref` sheet of
`benchmark_results.xlsx`.

### Running these on a cluster

High-dimensional runs belong in a batch job. Prepare a separate checkout and
verify the environment on the login node:

```tcsh
git clone https://github.com/Sayak183/qutip-bundling.git qutip-bundling-`hostname -s`
cd qutip-bundling-`hostname -s`
python -m pytest -q
python benchmarks/probe_high_dim_spin.py --dims 128 256 512
```

Then submit the calculation itself through the scheduler — never run it on a
login node, and do not detach it with `nohup`, which neither reserves the CPUs
and memory it uses nor survives node maintenance:

```tcsh
mkdir -p logs
sbatch --job-name=qutip-d256 --time=08:00:00 \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
    --output=logs/dim256-%j.log --error=logs/dim256-%j.log \
    --export=ALL,OMP_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8,MKL_NUM_THREADS=8,NUMEXPR_NUM_THREADS=8 \
    --wrap='python benchmarks/run_high_dim_spin_reference.py --dim 256'
```

Add your site's `--account`, `--partition`, and `--qos` as required. Monitor
with `squeue -u $USER` and `tail -f logs/dim256-<jobid>.log`.

**Comparing wall times.** Times measured in different jobs, or on a shared node
under different load, are not comparable: the same certified dimension-256
reference took 2,744 s as a standalone run and 86.6 s inside a sequential
sweep on the same node. Any timing table meant to be read as a scaling result
must come from one job on one node, pinned with `--nodelist` and a fixed thread
count, and be labelled as a single shared-node sweep.

With the reproducible Davies block floor, the probe estimates native-core
storage of about 0.03, 0.17, 0.86, and 4.27 GiB for dimensions 128, 256, 512,
and 1024. These estimates cover retained operator arrays, not all solver
temporaries, and say nothing about wall time. Review both memory headroom and
the requested batch time before raising `--max-core-gib`.

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
  Result 1-4 JSON/CSV outputs with the certified high-dimensional references
  into formatted summary and dynamics sheets with provenance and charts.
- `build_high_dim_sheet.py` regenerates that workbook's `High-dim Ref` sheet
  from the certified reference JSON files, leaving the other sheets untouched.
  Re-run it after adding a dimension, so the sheet cannot drift from the data:

  ```bash
  python benchmarks/build_high_dim_sheet.py --run-note "one-sweep provenance"
  ```
- [`archive/`](archive/) contains redacted historical transcripts, internal
  patch notes, and superseded one-off inspection scripts.
- Extra frontier PNGs ending in `_perUnit` or `_noEns_Single` are alternate
  views controlled by toggles in `plot_frontier.py`; the unsuffixed PNGs are
  the main figures embedded in [BENCHMARKS.md](BENCHMARKS.md).
