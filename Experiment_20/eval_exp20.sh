#!/bin/bash
#SBATCH --job-name=ITOR_Exp20_eval
#SBATCH --account=def-bolufe_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

PROJECT_DIR=/scratch/bolufe/UES_CMAES_RL/NEWReview/Experiment_20a
RESULTS_ROOT=${PROJECT_DIR}/results
OUT_DIR=${PROJECT_DIR}/eval_results_array

set -euo pipefail

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}"

module purge
source ~/TF_RL/bin/activate
export TF_CPP_MIN_LOG_LEVEL=2

echo "PWD=$(pwd)"
echo "PYTHON=$(which python)"
python --version

mapfile -t RUN_DIRS < <(find "${RESULTS_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'DQN_FinalCombo_Exp20aRewardAbl_clip1_lr1e4*' | sort)

TOTAL=${#RUN_DIRS[@]}

if [ "${TOTAL}" -eq 0 ]; then
  echo "No matching run directories found."
  exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (TOTAL=${TOTAL})"
  exit 1
fi

RUN_PATH="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME=$(basename "${RUN_PATH}")

echo "Evaluating run: ${RUN_NAME}"
echo "Results root: ${RESULTS_ROOT}"
echo "Output dir:   ${OUT_DIR}"

python eval_exp20.py \
  --results_root "${RESULTS_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_prefix DQN_FinalCombo_Exp20aRewardAbl_clip1_lr1e4 \
  --runs_per_function 51 \
  --base_seed 12345 \
  --only_run_name "${RUN_NAME}"

echo "Done: ${RUN_NAME}"