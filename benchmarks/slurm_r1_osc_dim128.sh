#!/bin/bash
#SBATCH --job-name=r1-osc128
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r1-osc128-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r1-osc128-%j.err

# Result 1 at dimension 128 for the oscillator_bath.
#
# The size-invariance claim now rests on FIVE dimensions for the spin chain
# (slopes -0.91, -0.97, -0.98, -0.98, -0.97 across a 16-fold range) and THREE
# for this system. This adds a fourth.
#
# oscillator, N_L=1686, 32 substeps. Its 64-substep reference was measured at 16,441 s (4.6 h) by job 19597388, so that part is known; the 200-realization M sweep at 32 substeps is the unknown. Budget 20-30 h.
#
# NO --overwrite: dim 128 writes its own new file and the existing dimensions
# are untouched. Result 1 compares SLOPES across dimensions rather than
# wall-clocks, so its points need not share an allocation, and leaving the
# published files alone removes any risk to them.

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u     run_accuracy_vs_M.py --system oscillator_bath --dims 128
