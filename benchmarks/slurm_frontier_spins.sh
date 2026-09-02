#!/bin/bash
#SBATCH --job-name=frontier-spins
#SBATCH --account=roib-account
#SBATCH --partition=roibq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --output=/usr/people/roib/sayak/qutip-bundling-landau/logs/frontier-%j.out
#SBATCH --error=/usr/people/roib/sayak/qutip-bundling-landau/logs/frontier-%j.err

# Frontier High-Core Benchmark on Landau roibq (32 Cores, 128 GB RAM)
# Sweeps dimensions N = 128, 256, 512 (7, 8, 9 spins) on System B (Mixed Chain).

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python -u \
    run_frontier_spins.py --system mixed_chain --dims 128 256 512 --m-values 16 32 64
