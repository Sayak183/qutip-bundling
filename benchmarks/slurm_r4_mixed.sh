#!/bin/bash
#SBATCH --job-name=r4-mixed
#SBATCH --account=all-account
#SBATCH --partition=allq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=r4-mixed-%j.out
#SBATCH --error=r4-mixed-%j.err

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
/usr/people/roib/sayak/qutip-bundling-landau/.conda-env/bin/python run_isocost_vs_dim.py --system mixed_chain --overwrite
