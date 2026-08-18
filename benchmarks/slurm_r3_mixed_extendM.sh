#!/bin/bash
#SBATCH --job-name=r3-mixed-M
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r3-mixed-M-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r3-mixed-M-%j.err

# Result 3, System B, with the SLB sweep extended to M = 64, 128, 256.
#
# WHY ONLY SYSTEM B. The point of extending is to reach the M where bundling
# bias falls below sampling noise -- where the marker flips from filled to
# hollow -- because that is where raising M stops paying and the useful range of
# the method ends. Extrapolating error/s.e.m. against M from the committed data:
#
#     A spin        N_L =   31   ratio ~ M^-0.34   crossover at M ~  1161
#     B mixed       N_L = 2017   ratio ~ M^-0.48   crossover at M ~   193
#     C oscillator  N_L =  890   ratio ~ M^-0.16   crossover at M ~ 18266
#
# A bundle cannot hold more operators than exist, so on A and C the crossover
# sits beyond N_L and can never be reached: those systems stay bias-limited even
# at M = N_L. B is the only one where it is reachable, so B is the only one
# extended. Running the others would burn hours to add points that cannot show
# anything new.
#
# WHY THE WHOLE SYSTEM RE-RUNS, not just the new M points. Result 3 compares
# wall-clocks, and plot_method_comparison.py refuses to draw a cost axis across
# files from different allocations. Adding M points in a second job would make
# every existing point in that panel incomparable with the new ones. The extra
# M values are only ~34 min of the ~4.3 h; the rest is re-deriving the
# references and mcsolve in the SAME allocation, which is what buys the right to
# put them on one axis.
#
# The dim-128 reference is the long pole at ~1.5 h on its own.
#
# Note: M values above N_L are clamped to N_L and de-duplicated, so the small
# dimensions (N_L = 7, 33) simply gain nothing rather than repeating a point.

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u \
    run_method_comparison.py --system mixed_chain \
    --m-grid 1 2 4 8 16 32 64 128 256 --overwrite
