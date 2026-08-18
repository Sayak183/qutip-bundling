#!/bin/bash
#SBATCH --job-name=osc256
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/osc256-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/osc256-%j.err

# Does SLB reach oscillator dim 256 at all? Result 2 records
# `slb_unstable_at_substeps: 32` there -- it tried and the RK4 diverged. Per the
# stiffness ceiling in BENCHMARKS.md 5.2 the requirement doubles per octave
# (32 at dim 64, 64 at 128, 128 at 256), so this asks for 128.
#
# NOT merged into Result 2's cost curve, deliberately. run_cost_scaling.py
# requires substeps be raised UNIFORMLY across dimensions or the points stop
# being cost-comparable, and raising the whole oscillator sweep to 128 would
# multiply every published SLB cost by 4 (300x advantage at dim 128 becomes
# 75x) and cost ~15 h, to gain one dot. This run answers the narrower question
# -- can it run there -- and leaves every published number untouched.

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks

# run_cost_scaling.py always writes cost_scaling_<system>.json, so the real file
# is moved aside first and restored after. The published data is never at risk:
# it is committed, but a job that clobbers it and then fails would still cost an
# afternoon of confusion.
KEEP=data/cost_scaling_oscillator_bath.json
SAVED=data/.cost_scaling_oscillator_bath.published
cp "$KEEP" "$SAVED"

/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u \
    run_cost_scaling.py --system oscillator_bath --sizes 128 \
    --slb-substeps 128 --overwrite
STATUS=$?

mv "$KEEP" data/osc_dim256_reach_substeps128.json 2>/dev/null
mv "$SAVED" "$KEEP"

echo "--- published cost_scaling_oscillator_bath.json restored ---"
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python - <<'EOF'
import json, pathlib
p = pathlib.Path("data/cost_scaling_oscillator_bath.json")
dims = sorted(q["dim"] for q in json.loads(p.read_text())["points"])
print("  dims present:", dims, "(expect [8, 16, 32, 64, 128, 256])")
EOF
exit $STATUS
