#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp1_fun11
#SBATCH --output=%x-%j.out
source ~/TF_RL/bin/activate
python run_exp.py