#!/bin/bash
#SBATCH --job-name=r4-multiobs
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-multiobs-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-multiobs-%j.err

# Result 4 scored on every observable instead of the energy alone.
#
# WHY. M* is the cheapest bundle count reaching the accuracy target, and it was
# decided from <H> because every solve passed e_ops=[H]. The energy is the
# EASIEST observable this suite measures, so that M* is the cost of the best
# case rather than the cost of using the method. Measured locally on System B:
#
#     dim  4, 8, 16   M* unchanged        binding observable sx or sz
#     dim 32          M* 16 -> 32         binding sx
#     dim 64          M* unchanged (32)   binding sz
#
# The binding observable is NEVER the energy at any dimension, and at dimension
# 32 the energy-only M* understates the true cost by a factor of two. That is
# what this run corrects.
#
# WHAT IT COSTS. Extra observables are only expectation values against a shared
# propagation, so the arithmetic is nearly free. The sweep is what grows: it now
# runs until the WORST observable clears the floor rather than the first, so it
# reaches larger M. Locally that was roughly 2x the sweep length.
#
# WHAT IT CHANGES IN THE PAPER. Every speedup in Result 4 will FALL, because a
# larger M* costs more. That is the point: a speedup that holds for every
# observable measured is worth more than a larger one that holds only for the
# easiest.
#
# Note on mcsolve: at dimension 64 the ntraj ladder hit MC_TIME_BUDGET_S and
# skipped 200 and 400, leaving the S^2 fit resting on ntraj=100. The skips are
# recorded in the data. S^2 is a property of the system, so fewer sampled ntraj
# costs precision rather than validity -- but check `mc_skipped` before quoting
# a trajectory count from the large dimensions.

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
BIN=/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python

# One system per invocation so a failure on the slow one does not cost the
# other two. The spin chain runs to dimension 512 and is by far the longest.
$BIN -u run_isocost_vs_dim.py --system mixed_chain --overwrite
$BIN -u run_isocost_vs_dim.py --system oscillator_bath --overwrite
$BIN -u run_isocost_vs_dim.py --system spin_chain --overwrite
