#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem-per-cpu=16G      # memory; default unit is megabytes
#SBATCH --gpus-per-node=1
#SBATCH --time=100:00:00
#SBATCH --job-name=exp19_%j.out
#SBATCH --error=exp19_%j.err
#SBATCH --output=%x-%j.out
source ~/TF_RL/bin/activate

python run_exp.py \
  --experiment "Exp19_obsAbl" \
  --seed 0 \
  --dim 30 \
  --state_size 20 \
  --include_single_run_features 1 \
  --reward_mode standard \
  --num_iterations 200000
