#!/bin/bash

#SBATCH --array=0-299
#SBATCH --job-name wlh_cgmf
#SBATCH --nodes=1
#SBATCH --tasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=48:00:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# For each job in array, run CGMF in parallell to handle a single sample,
# with 36 MPI ranks per node
srun -n 36 python -OO ./run_omp_uq.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
