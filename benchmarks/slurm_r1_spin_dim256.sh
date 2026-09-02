#!/bin/bash
#SBATCH --job-name=r1-spin256
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r1-spin256-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r1-spin256-%j.err

# Result 1, System A, at dimension 256 -- a fifth point for the size-invariance
# claim, which currently rests on four (dims 16, 32, 64, 128).
#
# The spin chain is the cheapest system to extend here because its N_L grows
# slowly (57 at dim 256) so the exact reference stays affordable, and M is
# capped at N_L anyway. Budget ~22 h.
#
# NO --overwrite, deliberately: dim 256 writes its own new file and the four
# existing dimensions are left exactly as they are. Result 1 compares SLOPES
# across dimensions rather than wall-clocks, so the points do not need to share
# an allocation, and not re-running them removes any risk to data already
# published.

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u \
    run_accuracy_vs_M.py --system spin_chain --dims 256
