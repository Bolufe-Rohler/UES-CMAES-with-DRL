#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --time=100:00:00
#SBATCH --job-name=all_benchmark
#SBATCH --output=%x-%j.out
source ~/TF_RL/bin/activate
python run_exp.py