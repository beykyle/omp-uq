#!/bin/bash

#SBATCH --job-name post_sf252Cf_KD_UQ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=72:00:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --mail-type=BEGIN,END,FAIL

python3 run_post_proc.py
