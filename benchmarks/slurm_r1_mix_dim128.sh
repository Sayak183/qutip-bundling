#!/bin/bash
#SBATCH --job-name=r1-mix128
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r1-mix128-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r1-mix128-%j.err

# Result 1 at dimension 128 for the mixed_chain.
#
# The size-invariance claim now rests on FIVE dimensions for the spin chain
# (slopes -0.91, -0.97, -0.98, -0.98, -0.97 across a 16-fold range) and THREE
# for this system. This adds a fourth.
#
# mixed chain, N_L=8,193 -- THE EXPENSIVE ONE. Its dim-128 exact reference cost 24.6 h in Result 2 and the 200-realization sweep at this N_L is heavier still. Budget 40-60 h, and it may be the least valuable of the three: System B already shows M^-1.00, M^-1.00, M^-1.02 across its existing sizes, which is about as flat as a slope gets.
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
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u     run_accuracy_vs_M.py --system mixed_chain --dims 128
