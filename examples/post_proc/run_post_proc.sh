#!/bin/bash

#SBATCH --job-name p3_kdcf
#SBATCH --nodes=2
#SBATCH --mem=180GB
#SBATCH --tasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

srun python3 run_post_proc.py 0 299 ~/turbo/omp_uq/run3_all/cf252/kduq/ ~/turbo/omp_uq/run3_all/cf252/kduq/kduq_post/
#srun python3 run_post_proc.py 0 299 ~/turbo/omp_uq/run3_all/cf252/chuq ~/turbo/omp_uq/run3_all/cf252/chuq/chuq_post
#srun python3 run_post_proc.py 0 299 ~/turbo/omp_uq/run3_all/cf252/wlh/ ~/turbo/omp_uq/run3_all/cf252/wlh/wlh_post/

#srun python3 run_post_proc.py 0 299 ~/turbo/omp_uq/run3_all/u235/kduq/ ~/turbo/omp_uq/run3_all/u235/kduq/kduq_post/
#srun python3 run_post_proc.py 0 299 ~/turbo/omp_uq/run3_all/u235/chuq/ ~/turbo/omp_uq/run3_all/u235/chuq/chuq_post
#srun python3 run_post_proc.py 0 299 ~/turbo/omp_uq/run3_all/u235/wlh/ ~/turbo/omp_uq/run3_all/u235/wlh/wlh_post/
