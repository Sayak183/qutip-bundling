#!/bin/bash
#SBATCH --job-name=r4-osc
#SBATCH --account=all-account
#SBATCH --partition=allq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=r4-osc-%j.out
#SBATCH --error=r4-osc-%j.err

cd /usr/people/roib/sayak/qutip-bundling-landau/benchmarks
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate qutip-env
python run_isocost_vs_dim.py --system oscillator_bath --overwrite
