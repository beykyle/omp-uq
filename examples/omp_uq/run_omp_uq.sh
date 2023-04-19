#!/bin/bash

#SBATCH --job-name chuq_cf_process_hist
#SBATCH --nodes=1
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=12
#SBATCH --time=2:00:00
#SBATCH --account=bckiedro98
#SBATCH --partition=standard
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Run CGMF in parallell, with 24 MPI ranks per node
mpirun -n 24 python ./run_omp_uq.py 0 3
