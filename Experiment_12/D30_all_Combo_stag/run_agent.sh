#!/bin/bash
#SBATCH --account=ctb-agodbout
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3     # CPU cores/threads
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --time=69:00:00       # execution time
#SBATCH --job-name=all_X_norm
#SBATCH --output=%x-%j.out
source ~/TF_RL/bin/activate
python run_exp.py