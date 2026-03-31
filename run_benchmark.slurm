#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -A SCALABLEML-10V5RWPBSMU-DEFAULT-GPU
#SBATCH --time=1-00:00:00
#SBATCH --job-name run_framework
#SBATCH --output /home/rida.lefdali/work/run_framework%j.log

# Load modules
source ~/.bashrc

module load Anaconda3
module load CUDA/12.6

conda activate musk_classif
# Final command
cd /home/rida.lefdali/work/Alignment_Benchmark

python run_framework.py
