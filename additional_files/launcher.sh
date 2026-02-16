#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -G 2
#SBATCH --time=1:00:00
#SBATCH -p gpu

module purge
conda activate pgc_project

python dqn_parallel.py 500 250