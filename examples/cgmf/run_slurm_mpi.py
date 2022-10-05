import sys
import os
from pathlib import Path

# tools to run CGMF on MPI in a SLURM job and concatenate history files
from omp_uq import run_ensemble

# where are we putting the results
res_dir = Path("/home/beykyle/scratch/aps/gook_comp/cgmf/histories")

# optical model param file
omp_fpath = Path("/home/beykyle/OM/KDOMPuq/KDGlobal.json")

# 252Cf sf
target_zaid = 98252 # 252Cf
e_inc = 0.0

# run num_hist total histories, dividing evenly across SLURM resources
num_hist = int(96E4)
cpus_per_node    = int(os.environ["SLURM_TASKS_PER_NODE"].split("(")[0])
nodes            = int(os.environ["SLURM_NNODES"])
num_hist_per_cpu = int(num_hist / (nodes * cpus_per_node))

print("Running {} histories, on {} nodes and {} tasks per node, for {} histories / cpu"
        .format(num_hist, nodes, cpus_per_node, num_hist_per_cpu))

# run on MPI, w/out calling srun - this script should be run via sbatch, with appropriate resources allocated
run_ensemble(omp_fpath, res_dir, num_hist_per_cpu, target_zaid, e_inc, "default", slurm=False)
