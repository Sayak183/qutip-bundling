#!/bin/bash
#SBATCH --job-name=r2-retime
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r2-retime-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r2-retime-%j.err

# Re-time Result 2. Its accuracy numbers are fine; its wall-clocks are not.
#
# WHY. Job 19599549 re-measured the mixed chain at dim 64 on this cluster, twice
# in one allocation. Every RMSE in the sweep reproduced to the last printed
# digit. The timings did not:
#
#     native reference   committed 1186.5 s   fresh 205.6 s   5.8x
#     SLB one solve      committed   2.057 s  fresh   0.632 s 3.3x
#     published ratio        576.8x           fresh 325.2x    1.8x
#
# The two halves were inflated unequally, so the ratio is inflated too. Thread
# pinning was the first suspect and is NOT the cause -- the same probe ran
# pinned and unpinned back to back and got 204.9 s against 206.2 s. Whatever it
# was (node contention, a long sequential job's memory behaviour), one timing is
# not a measurement here.
#
# WHAT THIS DOES DIFFERENTLY.
#   --timing-repeats 3   every timed solve is run three times; the median is
#                        published and all three samples are kept beside it, so
#                        the spread is visible instead of assumed
#   --exclusive          no other job shares the node, removing the most likely
#                        source of the original inflation
#   one job, one node    all three systems together, so cross-system wall-clocks
#                        are comparable for once
#
# COST. The committed reference times total 28.3 h, and the fresh dim-64
# measurement came back 5.8x under its committed value. If that holds, one pass
# is roughly 5 h and three passes of the reference put this near 15 h. roibq has
# no wall-clock limit. The mixed chain at dim 128 dominates: 88,443 s committed,
# so plan for it to be the long pole either way.
#
# WRITES TO A SCRATCH COPY. The committed data/ is not touched. Compare the
# result against it before merging anything.

set -e

REPO=/usr/people/roib/sayak/qutip-bundling-landau
PY=$REPO/.conda-env/bin/python
WORK=/usr/people/roib/sayak/r2-retime-$SLURM_JOB_ID

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

# Cheap systems first, so a failure surfaces in minutes rather than hours.
for SYSTEM in spin_chain oscillator_bath mixed_chain; do
    echo "=== $SYSTEM ==="
    $PY -u run_cost_scaling.py --system "$SYSTEM" --timing-repeats 3 --overwrite
    echo "finished $SYSTEM at $(date)"
    echo
done

echo "=== spread of the three timing samples, per point ==="
$PY - <<'PYEOF'
import glob, json
for path in sorted(glob.glob("data/cost_scaling_*.json")):
    print(path.split("/")[-1])
    for p in json.load(open(path))["points"]:
        reps, med = p.get("t_native_ref_repeats"), p.get("t_native_ref")
        if reps and med is not None:
            print(f"   dim {p['dim']:4d}  native ref median {med:9.1f} s"
                  f"   samples {[round(x, 1) for x in reps]}"
                  f"   spread {max(reps)/min(reps):.2f}x")
PYEOF

echo
echo "done. scratch dir: $WORK"
echo "compare against the committed data before merging."
