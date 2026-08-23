#!/bin/bash
#SBATCH --job-name=r4-osc128
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-osc128-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-osc128-%j.err

# Result 4, System C, extended to dimension 128.
#
# The oscillator's iso-cost curve stopped at dim 64 while the spin chain spanned
# 4-512, so its fitted slope rested on four points. This adds a fifth.
#
# CHEAPEST OF THE THREE REMAINING EXTENSIONS, which is why it is worth doing
# first. The whole cost is the dim-128 reference, measured at 8,136 s (2.3 h) in
# Result 2, plus its substep-halving certification. The four existing dimensions
# re-run in minutes. Budget ~5 h.
#
# dim 128 uses 32 substeps, not the 16 that dim 64 uses: the anharmonic ladder
# needs roughly double per octave, and 16 is what diverged at this size in
# Result 2. The reference gets 2x that.
#
# --overwrite is required because the runner writes one file per system, so the
# existing dims 8-64 are regenerated in the same allocation -- which is what
# keeps their wall-clocks comparable with the new point.

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u \
    run_isocost_vs_dim.py --system oscillator_bath --overwrite
