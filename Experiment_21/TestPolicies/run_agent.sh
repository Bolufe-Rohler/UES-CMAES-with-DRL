#!/bin/bash
#SBATCH --account=def-bolufe_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=exp21_eval
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/TF_RL/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

echo "==== JOB CONTEXT ===="
date
hostname
pwd
which python
python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "====================="

# Relative directories
ROOT_DIR="./policy"
OUTPUT_ROOT="./exp21_eval_results"

# Discover all run folders
mapfile -t RUN_DIRS < <(find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
  echo "ERROR: No run directories found under $ROOT_DIR"
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#RUN_DIRS[@]}" ]; then
  echo "ERROR: Array index $SLURM_ARRAY_TASK_ID out of range for ${#RUN_DIRS[@]} run directories."
  exit 1
fi

RUN_DIR="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="$(basename "$RUN_DIR")"
POLICY_DIR="$RUN_DIR/policy"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

echo "Selected RUN_DIR   = $RUN_DIR"
echo "Selected RUN_NAME  = $RUN_NAME"
echo "Selected POLICY_DIR= $POLICY_DIR"
echo "Selected OUTPUT_DIR= $OUTPUT_DIR"

if [ ! -d "$POLICY_DIR" ]; then
  echo "ERROR: Missing policy directory: $POLICY_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "==== POLICY CONTENTS ===="
find "$POLICY_DIR" -maxdepth 2 | sort
echo "========================="

python benchmark_exp21_single_agent.py \
  --policy_dir "$POLICY_DIR" \
  --run_name "$RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --runs 30 \
  --dim 30 \
  --state_size 20

echo "Finished evaluation for $RUN_NAME"
date