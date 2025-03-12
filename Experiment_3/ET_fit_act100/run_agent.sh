#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --cpus-per-task=1         # CPU cores/threads
#SBATCH --mem-per-cpu=2G      # memory; default unit is megabytes
#SBATCH --time=48:00:00
#SBATCH --job-name=ET_f100_a100
#SBATCH --output=%x-%j.out
source ~/TF_RL/bin/activate
python run_exp.py