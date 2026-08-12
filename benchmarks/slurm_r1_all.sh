#!/bin/bash
#SBATCH --job-name=r1-all
#SBATCH --account=all-account
#SBATCH --partition=allq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/r1-all-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/r1-all-%j.err


# Pin BLAS to the allocation. NumPy/OpenBLAS otherwise size their thread pools
# from the node's core count rather than the cgroup, oversubscribing 4 CPUs and
# inflating exactly the wall-clock this benchmark measures.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python run_accuracy_vs_M.py --all --overwrite
