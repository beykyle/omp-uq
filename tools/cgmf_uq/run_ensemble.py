import os
import math
import numpy as np
from pathlib import Path
from CGMFtk import histories as fh

"""
Run CGMF on MPI with supplied options and OMPs read in from json file at path param_fname
"""
def run_ensemble(param_fname : Path, results_dir : Path,
                 num_hist : int, zaid : int , energy_MeV : float,
                 sample_name : str, slurm=False, cpus_per_node=1, nodes=1):

    if slurm:
        print("running {} histories, on {} CPUs".format(num_hist, nodes*cpus_per_node))
        num_hist = int(math.ceil(num_hist / (nodes * cpus_per_node)))
        print("running {} histories / cpu".format(num_hist))


    cgmf_options = " -i " + str(zaid)         \
                 + " -e " + str(energy_MeV)   \
                 + " -n " + str(num_hist)     \
                 + " -o " + str(param_fname)

    slurm_options = " --job-name " + sample_name                \
                  + " --nodes=" + str(nodes)                    \
                  + " --ntasks-per-node=" + str(cpus_per_node)  \
                  + " --cpus-per-task=1"                        \
                  + " --mem-per-cpu=5g"                         \
                  + " --time=72:00:00"                          \
                  + " --account=bckiedro0"                      \
                  + " --partition=standard"                     \
                  + " --mail-type=BEGIN,END,FAIL"

    mpi_options = "--bind-to core" # we want to run a proces on each core

    if slurm:
        cmd = "srun "  + slurm_options + " cgmf.mpi.x " + cgmf_options
    else:
        cmd = "mpirun " + mpi_options + " cgmf.mpi.x " + cgmf_options

    # run CGMF and aggregate histories
    result_fpath = str(results_dir / ("histories_" + sample_name + ".o"))
    os.system(cmd)
    os.system("cat histories.cgmf.* > " + result_fpath)
    os.system("rm histories.cgmf.*")


def process_ensemble(results_dir : Path, out_dir : Path, sample_name : str,  num_hist=None):
    result_fpath = str(results_dir / str("histories_" + sample_name + ".o"))
    print("Processing {}".format(result_fpath))
    if num_hist != None:
        hist = fh.Histories(result_fpath, nevents=num_hist)
    else:
        hist = fh.Histories(result_fpath)

    # extract some data from history files for immediate post-processing
    nubins, pnu = hist.Pnu()
    ebins,pfns  = hist.pfns()
    nubarA      = hist.nubarA()

    # save compressed post-processed distributions
    np.save(str(out_dir / ("nu_"     + sample_name)), nubins )
    np.save(str(out_dir / ("pnu_"    + sample_name)), pnu )
    np.save(str(out_dir / ("ebins_"  + sample_name)), ebins )
    np.save(str(out_dir / ("pfns_"   + sample_name)), pfns )
    np.save(str(out_dir / ("A_"      + sample_name)), nubarA[0] )
    np.save(str(out_dir / ("nuA_"    + sample_name)), nubarA[1] )

def process_ensemble_corr(results_dir : Path, sample_name : str, num_hist=None):
    result_fpath = str(results_dir / str("histories_" + sample_name + ".o"))
    if num_hist != None:
        hist = fh.Histories(result_fpath, nevents=num_hist)
    else:
        hist = fh.Histories(result_fpath)

    # extract some data from history files for immediate post-processing
    nug = hist.getNugtot()
    nu  = hist.getNutot()

    # save compressed post-processed distributions
    np.save(str(results_dir / ("event_nu_p_"     + sample_name)), nug )
    np.save(str(results_dir / ("event_nu_n_"     + sample_name)), nu )


def run_and_process_ensemble(param_fname : Path, results_dir : Path,
                             num_hist : int, zaid : int , energy_MeV : float,
                             sample_name : str, slurm=False, cores=None):
    run_ensemble(param_fname, results_dir, num_hist, zaid, energy_MeV, sample_name, slurm, cores)
    process_ensemble(results_dir, sample_name, num_hist=num_hist)

def calculate_ensemble_nubar(sample_name : str, results_dir : Path):
    nu   = np.load(str(results_dir / ("nu_"     + sample_name + ".npy")))
    pnu  = np.load(str(results_dir / ("pnu_"    + sample_name + ".npy")))
    return np.dot(nu,pnu)

