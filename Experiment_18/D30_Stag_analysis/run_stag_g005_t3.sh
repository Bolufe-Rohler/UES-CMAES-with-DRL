#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=160:00:00
#SBATCH --job-name=stag_g005_t3
#SBATCH --output=%x-%j.out
#SBATCH --array=11,14,23 # F11, F14, F23

source ~/TF_RL/bin/activate

export GAMMA=0.05
export TAU=3
export FUNC_NUM=${SLURM_ARRAY_TASK_ID} export NUM_ITER=30000

python run_exp.py
