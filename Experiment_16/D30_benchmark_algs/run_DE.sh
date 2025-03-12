#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --mem-per-cpu=1G      # memory; default unit is megabytes
#SBATCH --time=56:00:00
#SBATCH --job-name=DE_results
#SBATCH --output=%x-%j.out
source ~/SCIPY/bin/activate
python de_run.py