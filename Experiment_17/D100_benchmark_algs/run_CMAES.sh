#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --mem-per-cpu=1G      # memory; default unit is megabytes
#SBATCH --time=148:00:00
#SBATCH --job-name=d100_CMAES
#SBATCH --output=%x-%j.out
source ~/SCIPY/bin/activate
python CMAES_run.py