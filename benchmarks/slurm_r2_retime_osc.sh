#!/bin/bash
#SBATCH --job-name=r2-retime-osc
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r2-retime-osc-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r2-retime-osc-%j.err

# Redo the oscillator half of the Result 2 re-timing, at the right substeps.
#
# Job 19599550 ran all three systems on DEFAULTS. That matched the committed
# settings for the mixed chain, so its numbers stand. It did not match the
# oscillator: that system is committed at 32 substeps (recorded in its metadata)
# and the job ran it at the default 4. An eighth of the integration work is not
# a faster measurement of the same thing, it is a measurement of something else,
# so those oscillator numbers are discarded rather than compared.
#
# The tell was in the output: the oscillator appeared to gain 15x at dim 32
# (117.0 s committed against 7.7 s) where the mixed chain gained 5.8x at dim 64.
# Roughly 8x of that apparent gain is simply the missing substeps.
#
# For the record, the mixed chain result that job DID produce, with three
# samples per point and spreads of 1.02-1.03x:
#
#     dim  64   reference 1186.5 s -> 212.5 s     ratio 576.8x  -> 352.6x
#     dim 128   reference 88442.7 s -> 3249.3 s   ratio 2263.5x -> 1013.2x
#
# Both halves were inflated, unequally: the reference by 27x and the SLB solve
# by 12x. On an exclusive node the timings are stable to a few percent, so the
# original numbers were not noisy -- they were inflated by sharing a node, and
# by an amount that grows with how heavy the solve is.
#
# COST. Committed oscillator references total 9,052 s, dominated by 8,136 s at
# dim 128. If it gains anything like the mixed chain, three repeats should be
# 1-2 h. Budget generously; roibq has no wall clock.

set -e

REPO=/usr/people/roib/sayak/qutip-bundling-landau
PY=$REPO/.conda-env/bin/python
WORK=/usr/people/roib/sayak/r2-retime-osc-$SLURM_JOB_ID

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
echo "committed values to beat, at 32 substeps:"
echo "   dim 32 = 117.0 s,  dim 64 = 786.5 s,  dim 128 = 8136.2 s"
echo

$PY -u run_cost_scaling.py --system oscillator_bath \
    --slb-substeps 32 --timing-repeats 3 --overwrite

echo
echo "=== spread of the three timing samples, per point ==="
# Guarded against None: a point whose SLB solve went unstable records the
# reference it did obtain and a null timing, and formatting that null is what
# made job 19599550 exit 1 after all its data had been written safely.
$PY - <<'PYEOF'
import json
for p in json.load(open("data/cost_scaling_oscillator_bath.json"))["points"]:
    reps, med = p.get("t_native_ref_repeats"), p.get("t_native_ref")
    if not reps or med is None:
        continue
    slb = p.get("t_slb_fixed")
    ratio = f"{med / slb:8.1f}x" if slb else "      n/a"
    print(f"   dim {p['dim']:4d}  ref median {med:9.1f} s  ratio {ratio}"
          f"   samples {[round(x, 1) for x in reps]}"
          f"   spread {max(reps) / min(reps):.2f}x")
PYEOF

echo
echo "done. scratch dir: $WORK"
