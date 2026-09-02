#!/bin/bash
#SBATCH --job-name=extreme256
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/extreme256-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/extreme256-%j.err

# No --time: roibq imposes no wall-clock limit. The thermal check integrates to
# t=60 on the benchmark step size, which is roughly 12x the work of one sweep
# point, so the whole run is a few hours rather than the one hour a naive
# estimate from the sweep alone would suggest.
#
# 16 thermal realizations, not the 4 of the first successful run (job 19592847).
# That run ended 0.0046 short of the Gibbs value, which was 1.1 of its OWN
# standard errors -- too close to call. The residual is the whole point of the
# check, so it is worth paying to halve the bar it is judged against. This costs
# about 4.5 h against 1.1 h for the thermal stage; ~5.5 h in total.

# Pin BLAS to the allocation. NumPy/OpenBLAS otherwise size their thread pools
# from the node's core count rather than the cgroup, oversubscribing 4 CPUs and
# inflating exactly the wall-clock this benchmark records.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u run_extreme_dimension.py --size 8 --m-values 8 16 32 --thermal-realizations 16
