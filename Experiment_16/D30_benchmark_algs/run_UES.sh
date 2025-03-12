#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --mem-per-cpu=1G      # memory; default unit is megabytes
#SBATCH --time=48:00:00
#SBATCH --job-name=UES_results
#SBATCH --output=%x-%j.out
source ~/SCIPY/bin/activate
python ues_run.py