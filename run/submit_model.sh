#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --job-name=gpu_unet
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=00:10:00
#SBATCH --mem=6G
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

export PATH="/home/users/twilder/Python/python_env_test/bin:$PATH"
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

# python executable
python run_model.py --config ./config_model.yml