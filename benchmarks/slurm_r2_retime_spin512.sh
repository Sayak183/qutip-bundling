#!/bin/bash
#SBATCH --job-name=r2-spin512
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r2-spin512-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r2-spin512-%j.err

# Completes the spin chain half of the Result 2 re-timing.
#
# slurm_r2_retime.sh omitted --native-ref-max, so it took the default of 128 and
# stopped the spin chain's exact reference there. The committed spin_chain run
# used an override to 512 (recorded as native_ref_max_dim: 512 in its metadata),
# so that job re-timed dims 4-128 and silently skipped the two most expensive
# points on the curve -- dim 256 at 280.3 s and dim 512 at 2,799.8 s committed.
#
# The other two systems are unaffected: both committed at native_ref_max_dim
# 128, which IS the default, so job 19599550 reproduces their coverage exactly.
# In particular the mixed chain reaches dim 128, which is the measurement this
# whole exercise exists for.
#
# The spin chain is re-run WHOLE rather than just the two missing points, so its
# curve comes from one allocation on one node and its slope fit is internally
# consistent. Result 2 does not compare wall-clocks across systems, so it does
# not matter that this is a different node from 19599550.
#
# COST. Committed spin-chain references total 3,116 s, and 19599550 measured the
# fresh speedup on this system at only 1.4x (dim 64) to 2.0x (dim 128) -- much
# smaller than the mixed chain's 5.8x, because the spin chain has 31 operators
# where the mixed chain has 2,017. Budget three repeats at ~2x: roughly 1.5 h.

set -e

REPO=/usr/people/roib/sayak/qutip-bundling-landau
PY=$REPO/.conda-env/bin/python
WORK=/usr/people/roib/sayak/r2-spin512-$SLURM_JOB_ID

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
echo "committed values to beat: dim 256 = 280.3 s, dim 512 = 2799.8 s"
echo

$PY -u run_cost_scaling.py --system spin_chain \
    --native-ref-max 512 --timing-repeats 3 --overwrite

echo
echo "=== spread of the three timing samples, per point ==="
$PY - <<'PYEOF'
import json
for p in json.load(open("data/cost_scaling_spin_chain.json"))["points"]:
    reps = p.get("t_native_ref_repeats")
    if reps:
        print(f"   dim {p['dim']:4d}  native ref median {p['t_native_ref']:9.1f} s"
              f"   samples {[round(x, 1) for x in reps]}"
              f"   spread {max(reps)/min(reps):.2f}x")
PYEOF

echo
echo "done. scratch dir: $WORK"
