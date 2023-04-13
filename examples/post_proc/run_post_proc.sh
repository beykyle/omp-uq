#!/bin/bash

#SBATCH --job-name mppchcf
#SBATCH --nodes=10
#SBATCH --mem=48GB
#SBATCH --sockets-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --time=2:00:00
#SBATCH --account=bckiedro98
#SBATCH --partition=standard
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Process CGMF history files in parallell, with 2 MPI ranks per node,
# each rank being bound to 1 of the node's 2 sockets
mpiexec -bind-to socket:2 python3 -m run_post_proc 0 299

# NOTE because this is memory limited (loading and processing few GB from disk per file)
# we want to bind to units that do no share memory. In the case of Armis2, we have
# Intel(R) Xeon(R) CPU E5-2680, with 2 sockets per node, and 12 cores per socket, for a
# total of 24 cores. However, the 12 cores on the socket share L3 cache and main memory,
# so binding to cores will cause each core to stomp on eachother's cache and memory, and
# dramatically slow down, hence the -bind-to socket:2 and --ntasks-per-socket=1.
