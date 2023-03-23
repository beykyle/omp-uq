#!/bin/bash

#SBATCH --array=0-1
#SBATCH --job-name mppkdcf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=5:00:00
#SBATCH --account=bckiedro98
#SBATCH --partition=standard
#SBATCH --mail-type=BEGIN,END,FAIL

python3 run_post_proc.py 20 ${SLURM_ARRAY_TASK_ID}
