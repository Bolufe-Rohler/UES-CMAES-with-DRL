#!/bin/bash
#SBATCH --account=def-bolufe_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=exp19_base_s2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd $SLURM_SUBMIT_DIR
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
echo "====================="

echo "==== FILE CHECKSUMS (submit dir) ===="
ls -la
if command -v md5sum >/dev/null 2>&1; then
  echo "md5(run_exp.py)                 :" $(md5sum run_exp.py | awk '{print $1}')
  echo "md5(env_finalcombo_ablation.py) :" $(md5sum env_finalcombo_ablation.py | awk '{print $1}')
  echo "md5(ues_cmaes_X.py)              :" $(md5sum ues_cmaes_X.py | awk '{print $1}')
else
  echo "md5sum not available."
fi
echo "===================================="

echo "==== IMPORT PATH VERIFICATION ===="
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
echo "==============================="

python -u run_exp.py \
  --experiment "Exp19_base_CONT" \
  --seed 2 \
  --dim 30 \
  --state_size 20 \
  --include_single_run_features 1 \
  --reward_mode standard \
  --num_iterations 45000
