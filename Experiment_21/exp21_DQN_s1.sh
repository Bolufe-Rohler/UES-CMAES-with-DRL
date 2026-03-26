#!/bin/bash
#SBATCH --account=def-bolufe_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=exp21_DQN_s1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd $SLURM_SUBMIT_DIR
source ~/TF_RL/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Force unbuffered python output so .out isn't empty until job ends/crashes
export PYTHONUNBUFFERED=1
# Reduce TF spam (optional)
export TF_CPP_MIN_LOG_LEVEL=2

echo "==== JOB CONTEXT ===="
date
hostname
pwd
which python
python --version
echo "====================="

echo "==== FILE CHECKSUMS (current working directory) ===="
ls -la

if command -v md5sum >/dev/null 2>&1; then
  echo "md5(run_exp.py)                :" $(md5sum run_exp.py | awk '{print $1}')
  echo "md5(env_finalcombo_ablation.py):" $(md5sum env_finalcombo_ablation.py | awk '{print $1}')
  echo "md5(ues_cmaes_X.py)             :" $(md5sum ues_cmaes_X.py | awk '{print $1}')
else
  echo "md5sum not available on this system."
fi

echo "===================================================="

echo "==== IMPORT PATH VERIFICATION (what Python will actually import) ===="
python -u - << 'PY'
import inspect, pathlib, hashlib, sys

def show(modname):
    m = __import__(modname)
    p = pathlib.Path(inspect.getfile(m)).resolve()
    h = hashlib.md5(p.read_bytes()).hexdigest()
    print(f"{modname} -> {p}")
    print(f"{modname} md5 -> {h}")
    print(f"{modname} head -> {p.read_text().splitlines()[:5]}")
    print("")

print("sys.path head:", sys.path[:5], "\n")
show("ues_cmaes_X")
show("env_finalcombo_ablation")
PY
echo "==============================================================="

python run_exp.py \
  --experiment "Exp21_DQNvsDDQN" \
  --agent dqn \
  --seed 1 \
  --dim 30 \
  --state_size 20 \
  --include_single_run_features 1 \
  --reward_mode standard \
  --num_iterations 90000 \
  --eval_interval 500 \
  --num_eval_episodes 10
