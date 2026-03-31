#!/bin/bash
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*56)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --output /home/rida.lefdali/work/Alignment_Benchmark/compute_out.log
#! How much wallclock time will be required?
#SBATCH --time=36:00:00
# Load modules
source ~/.bashrc

module load Anaconda3
module load CUDA/12.6

conda activate musk_classif
# Final command
cd /home/rida.lefdali/work/Alignment_Benchmark

python run_framework.py
