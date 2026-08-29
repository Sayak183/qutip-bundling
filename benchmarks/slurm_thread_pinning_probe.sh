#!/bin/bash
#SBATCH --job-name=pin-probe
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/pin-probe-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/pin-probe-%j.err

# Does thread pinning explain Result 2's cost numbers?
#
# All three committed cost_scaling files ran with OMP_NUM_THREADS and friends
# UNSET, while every method_comparison file ran with them pinned to 4. The two
# disagree by 37x on the same quantity -- the mixed chain's dim-128 exact solve
# is 88,443 s in cost_scaling and 2,413 s in method_comparison. Substeps account
# for 2x of that. Oversubscription is the obvious suspect for the rest: with
# --cpus-per-task=4 on a 72-core node, an unpinned OpenBLAS will happily start
# 72 threads and thrash against the 4 CPUs Slurm actually granted.
#
# This settles it by running the SAME measurement twice in ONE allocation on ONE
# node -- so node speed, BLAS build, and file system are all held fixed and the
# only difference is the environment.
#
# Dimension 64 is the probe rather than 128: its committed reference is 1,186.5 s
# unpinned, so the whole job is minutes rather than a day, and 1,186.5 is the
# number to beat.
#
# Everything runs in a scratch copy. The committed data/ is never written to.

set -e

REPO=/usr/people/roib/sayak/qutip-bundling-landau
PY=$REPO/.conda-env/bin/python
WORK=/usr/people/roib/sayak/pin-probe-$SLURM_JOB_ID

mkdir -p "$WORK"
cp -r "$REPO/benchmarks" "$WORK/benchmarks"
rm -rf "$WORK/benchmarks/data"
mkdir -p "$WORK/benchmarks/data"
cd "$WORK/benchmarks"

echo "node: $(hostname)   cpus granted: $SLURM_CPUS_PER_TASK"
echo "committed unpinned value for this measurement: 1186.5 s"
echo

for MODE in unpinned pinned; do
    if [ "$MODE" = "pinned" ]; then
        export OMP_NUM_THREADS=4
        export OPENBLAS_NUM_THREADS=4
        export MKL_NUM_THREADS=4
        export NUMEXPR_NUM_THREADS=4
    else
        unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
    fi

    echo "=== $MODE ==="
    rm -f data/cost_scaling_mixed_chain.json
    $PY -u run_cost_scaling.py --system mixed_chain --sizes 6 \
        --native-ref-max 64 --overwrite
    $PY - <<'PYEOF'
import json
pts = json.load(open("data/cost_scaling_mixed_chain.json"))["points"]
for p in pts:
    if p.get("t_native_ref"):
        print(f"  RESULT  dim {p['dim']}  native_ref {p['t_native_ref']:.1f} s  "
              f"slb_fixed {p['t_slb_fixed']:.3f} s  "
              f"ratio {p['t_native_ref']/p['t_slb_fixed']:.1f}x")
PYEOF
    echo
done

echo "done. scratch dir: $WORK"
