#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:V100:1

source ~/doodle-jump-rl/venv/bin/activate
cd ~/doodle-jump-rl

srun python deepQAgent.py --server --max_games 1000
