import sys
import os
from pathlib import Path

# tools to run CGMF on MPI in a SLURM job and concatenate history files
from cgmf_uq import run_ensemble

# where are we putting the results
res_dir = Path("./")
assert(res_dir.is_dir())

# optical model param file - use default params
omp_fpath = Path("/home/kbeyer/omplib/data/KDUQSamples/samples/")

# U235 (n,th)
target_zaid = 92235
e_inc_MeV = 25.0E-9

# run num_hist total histories, dividing evenly across SLURM resources
num_hist = int(192E3)
cpus_per_node    = int(os.environ["SLURM_TASKS_PER_NODE"].split("(")[0])
nodes            = int(os.environ["SLURM_NNODES"])
num_hist_per_cpu = int(num_hist / (nodes * cpus_per_node))

print("Running {} histories, on {} nodes and {} tasks per node, for {} histories / cpu"
        .format(num_hist, nodes, cpus_per_node, num_hist_per_cpu))

# run default
default_dir = res_dir / "default"
assert(default_dir.is_dir())
default_omp_file = Path("/home/kbeyer/omplib/data/KDUQSamples/KD_default.json")
run_ensemble(default_omp_file, default_dir, num_hist_per_cpu, target_zaid, e_inc_MeV, "default", slurm=False)

# run UQ
for i in range(0,416):
    omp_file = omp_fpath / ("kd_{}.json".format(i))
    run_ensemble(omp_file, res_dir, num_hist_per_cpu, target_zaid, e_inc_MeV, str(i), slurm=False)
