import sys
import os
from pathlib import Path

# tools to run CGMF on MPI in a SLURM job and concatenate history files
from cgmf_uq import run_ensemble

# where are we putting the results
res_dir = Path("/home/beykyle/turbo/omp_uq/KDUQ")

# optical model param file - use default params
omp_fpath = Path("/home/beykyle/OM/KDOMPuq/s")

# 252Cf sf
target_zaid = 98252
e_inc = 0.0

# run num_hist total histories, dividing evenly across SLURM resources
num_hist = int(192E3)
cpus_per_node    = int(os.environ["SLURM_TASKS_PER_NODE"].split("(")[0])
nodes            = int(os.environ["SLURM_NNODES"])
num_hist_per_cpu = int(num_hist / (nodes * cpus_per_node))

print("Running {} histories, on {} nodes and {} tasks per node, for {} histories / cpu"
        .format(num_hist, nodes, cpus_per_node, num_hist_per_cpu))

for i in range(0,416):
    omp_file = omp_fpath / (str(i) + ".json")
    run_ensemble(omp_file, res_dir, num_hist_per_cpu, target_zaid, e_inc, str(i), slurm=False)
