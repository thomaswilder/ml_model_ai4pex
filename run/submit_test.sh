#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --job-name=gpu_unet
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=00:30:00
#SBATCH --mem=6G
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

export PATH="/home/users/twilder/Python/AI4PEX/tensorflow_env/bin:$PATH"
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

echo "=== GPU snapshot ==="
nvidia-smi

# Log GPU utilization every second while python runs
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv -l 1 > logs/gpu_util_${SLURM_JOB_ID}.txt &
SMI_PID=$!

# python executable
python run_model.py --config ./config_model.yml

# Stop logging GPU utilization
kill $SMI_PID