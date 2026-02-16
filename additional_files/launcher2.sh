#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --time=2:00:00
#SBATCH -p gpu

module purge
conda activate pgc_project

echo "here"
python dqn.py 100 50

echo "there"
python dqn.py 200 100

echo "yhere"
python dqn.py 500 250

echo "uhere"
python dqn.py 1000 500