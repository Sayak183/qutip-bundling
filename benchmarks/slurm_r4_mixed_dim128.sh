#!/bin/bash
#SBATCH --job-name=r4-mix128
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-mix128-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-mix128-%j.err

# Result 4, System B, extended to dimension 128.
#
# THE EXPENSIVE ONE -- roughly 40 h, and worth knowing why before submitting.
# The dim-128 reference on this system took 88,443 s (24.6 h) in Result 2, and
# its substep-halving certification took most of a further day. That single
# point is the entire cost; everything else re-runs in about an hour.
#
# It buys a fifth point on System B's iso-cost curve, which currently spans
# dims 4-64. Whether that is worth a day and a half of a node is a judgement
# call -- the existing four already give a clean fit, and the slope is quoted
# with its point count.
#
# Submit the oscillator (slurm_r4_osc_dim128.sh, ~5 h) first if you want the
# cheaper half of the same improvement.

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u \
    run_isocost_vs_dim.py --system mixed_chain --overwrite
