#!/bin/bash
#SBATCH --account=def-bolufe_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=exp19_smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd $SLURM_SUBMIT_DIR
source ~/TF_RL/bin/activate

python run_exp.py \
  --experiment "Exp19_smoketest" \
  --seed 0 \
  --dim 30 \
  --state_size 20 \
  --include_single_run_features 1 \
  --reward_mode standard \
  --num_iterations 5000
