#!/bin/bash
#SBATCH --job-name=r4-osc
#SBATCH --account=all-account
#SBATCH --partition=allq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-osc-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r4-osc-%j.err


# Pin BLAS to the allocation. NumPy/OpenBLAS otherwise size their thread pools
# from the node's core count rather than the cgroup, oversubscribing 4 CPUs and
# inflating exactly the wall-clock this benchmark measures.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u run_isocost_vs_dim.py --system oscillator_bath --overwrite
