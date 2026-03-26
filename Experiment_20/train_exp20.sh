#!/bin/bash
#SBATCH --job-name=ITOR_Exp20a_train
#SBATCH --account=def-bolufe_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# --------- USER SETTINGS ----------
PROJECT_DIR=/scratch/bolufe/UES_CMAES_RL/NEWReview/Experiment_20a
RESULTS_DIR=${PROJECT_DIR}/results
# ----------------------------------

set -euo pipefail

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

module purge
# IMPORTANT: do NOT load python/3.10; your venv is python3.8
source ~/TF_RL/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

# ----------------- GRID -----------------
REWARD_MODES=("standard" "normalized" "stagnation")
SEEDS=(0 1 2)

NUM_REWARDS=${#REWARD_MODES[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_REWARDS * NUM_SEEDS))

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (TOTAL=${TOTAL})"
  exit 1
fi

reward_idx=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
seed_idx=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

REWARD="${REWARD_MODES[$reward_idx]}"
SEED="${SEEDS[$seed_idx]}"

echo "Running Exp20a TRAIN: reward_mode=${REWARD}, seed=${SEED}"
echo "Project: ${PROJECT_DIR}"
echo "Results: ${RESULTS_DIR}"

python run_exp.py \
  --experiment DQN_FinalCombo_Exp20aRewardAbl_clip1_lr1e4 \
  --out_dir "${RESULTS_DIR}" \
  --seed "${SEED}" \
  --dim 30 \
  --state_size 20 \
  --include_single_run_features 1 \
  --reward_mode "${REWARD}" \
  --tau 3 \
  --stagnation_penalty 0.1 \
  --penalty_lambda 0.0 \
  --num_iterations 200000 \
  --initial_collect_steps 100 \
  --collect_steps_per_iteration 1 \
  --replay_buffer_max_length 100000 \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --eval_interval 2000 \
  --log_interval 200 \
  --num_eval_episodes 10 \
  --save_policy_interval 10 \
  --save_policy_at_end 1

echo "Done: reward_mode=${REWARD}, seed=${SEED}"
