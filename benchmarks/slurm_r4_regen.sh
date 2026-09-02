#!/bin/bash
#SBATCH --job-name=r4-regen
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-regen-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-regen-%j.err

# Regenerate Result 4 so its mcsolve projection uses the corrected S estimator.
#
# WHY. ntraj* = (S/target)^2 needs S, the per-trajectory spread.
# run_isocost_vs_dim.py estimated it as rmse^2 * ntraj, where rmse is tavg_rmse
# -- which combines (sample mean - reference)^2 with SEM^2. mcsolve is unbiased,
# so the realized deviation of its sample mean IS sampling fluctuation with
# variance SEM^2: the two terms are one quantity counted twice.
#
# Measured on the mixed chain at dim 16, four repeats per point:
#
#     ntraj   S via tavg_rmse   S via spread   ratio
#       100        0.4183          0.3267      1.280
#       200        0.4984          0.3412      1.461
#       400        0.4506          0.3670      1.228
#
# Since ntraj* goes as S^2, the projections inflate by 1.5-2.1x and every
# speedup with them. The runner now records `s_repeats` -- the spread across
# trajectories -- and the plotter prefers it. The committed files predate that
# field, which is what this job fixes. Expect the published speedups to FALL:
# 468x -> roughly 250-310x on the mixed chain, 1,739x -> roughly 830-1,150x on
# the oscillator.
#
# SETTINGS. Deliberately none. The committed runs used no overrides -- substeps
# come from the runner's own per-size table, recorded as substeps_by_size in the
# metadata (spin and mixed all 4; oscillator 4/4/4/16/32), and max_full_dim is
# the default 32. Adding a flag here would change what is measured rather than
# how well, which is exactly how two earlier re-runs went wrong. The script
# checks the regenerated settings against the committed ones at the end.
#
# --exclusive because Result 2's re-timing showed shared nodes inflating the
# heaviest solves by up to 27x. Accuracy is unaffected by that -- every RMSE
# reproduced exactly -- but this Result quotes costs.
#
# COST. The previous full run (19597387) took 18 h on a shared node for all
# three systems, and the dim-128 extensions took two days each. MC_TIME_BUDGET_S
# caps mcsolve at one hour per dimension, so the ceiling is bounded. Budget
# 20-40 h and let roibq's lack of a wall clock absorb the error; my estimates on
# this project have been wrong by 10x in one direction and 4x in the other.
#
# WRITES TO A SCRATCH COPY. Compare against the committed files with
# benchmarks/merge_retimed.py before merging anything.

set -e

REPO=/usr/people/roib/sayak/qutip-bundling-landau
PY=$REPO/.conda-env/bin/python
WORK=/usr/people/roib/sayak/r4-regen-$SLURM_JOB_ID

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p "$WORK"
cp -r "$REPO/benchmarks" "$WORK/benchmarks"
rm -rf "$WORK/benchmarks/data"
mkdir -p "$WORK/benchmarks/data"
cd "$WORK/benchmarks"

echo "node $(hostname), $SLURM_CPUS_PER_TASK cpus, exclusive"
echo "started $(date)"
echo

# Cheapest first, so a failure surfaces in minutes rather than after a day.
for SYSTEM in spin_chain oscillator_bath mixed_chain; do
    echo "=== $SYSTEM ==="
    $PY -u run_isocost_vs_dim.py --system "$SYSTEM" --overwrite
    echo "finished $SYSTEM at $(date)"
    echo
done

echo "=== did the corrected S estimator actually land, and how much did it move? ==="
$PY - "$REPO/benchmarks/data" <<'PYEOF'
import glob, json, sys
import numpy as np

committed = sys.argv[1]
for path in sorted(glob.glob("data/isocost_vs_dim_*.json")):
    name = path.split("/")[-1]
    doc = json.load(open(path))
    old = json.load(open(f"{committed}/{name}"))

    m, o = doc["meta"], old["meta"]
    same = (m.get("substeps") == o.get("substeps")
            and m["params"].get("substeps_by_size")
            == o["params"].get("substeps_by_size"))
    print(f"{name}   settings match committed: {same}")
    if not same:
        print(f"   regenerated {m.get('substeps')} "
              f"{m['params'].get('substeps_by_size')}")
        print(f"   committed   {o.get('substeps')} "
              f"{o['params'].get('substeps_by_size')}")

    for p in doc["points"]:
        rows = p.get("mc_fit") or []
        if not rows or "s_repeats" not in rows[0]:
            print(f"   dim {p['dim']:4d}: s_repeats MISSING -- runner is stale")
            continue
        r = rows[0]
        direct = float(np.mean(np.asarray(r["s_repeats"], dtype=float)))
        viarmse = float(np.mean(np.asarray(r["rmse_repeats"], dtype=float)
                                ) * np.sqrt(r["ntraj"]))
        print(f"   dim {p['dim']:4d}: S direct {direct:.4f}  via rmse "
              f"{viarmse:.4f}  ratio {viarmse/direct:.3f}  "
              f"=> ntraj* falls {(viarmse/direct)**2:.2f}x")
PYEOF

echo
echo "done. scratch dir: $WORK"
echo "verify with:  python benchmarks/merge_retimed.py <file>   (report only)"
echo "NOTE: merge_retimed.py is written for cost_scaling files; for Result 4"
echo "      compare the settings printed above and the m_star ladders by hand."
