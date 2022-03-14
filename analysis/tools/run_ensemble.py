import os
import numpy as np
from pathlib import Path
from CGMFtk import histories as fh

"""
Run CGMF on MPI with supplied options and OMPs read in from json file at path param_fname
"""
def run_ensemble(param_fname : Path, results_dir : Path,
                 num_hist : int, zaid : int , energy_MeV : float,
                 sample_name : str, slurm=False, cores=None):

    cgmf_options = " -i " + str(zaid)         \
                 + " -e " + str(energy_MeV)   \
                 + " -n " + str(num_hist)     \
                 + " -o " + str(param_fname)

    mpi_options = ""
    if cores != None:
        mpi_options = " --np " + str(cores)

    if slurm:
        cmd = "srun "  + mpi_options + " cgmf.mpi.x " + options
    else:
        cmd = "mpirun " + mpi_options + " cgmf.mpi.x " + options

    # run CGMF and aggregate histories
    result_fpath = str(results_dir / "histories_" + sample_name + ".o")
    os.system(cmd)
    os.system("cat histories.cgmf.* > " + result_fpath)
    os.system("rm histories.cgmf.*")

    # read histories
    hist = fh.Histories(result_fpath, nevents=num_hist)

    # extract some data from history files for immediate post-processing
    nu          = hist.getNutot()
    nubins, pnu = hist.Pnu()
    ebins,pfns  = hist.pfns()
    nubarA      = hist.nubarA()

    # save compressed post-processed distributions
    np.save(str(results_dir / ("nu_"     + sample_name)), nubins )
    np.save(str(results_dir / ("pnu_"    + sample_name)), pnu )
    np.save(str(results_dir / ("ebins_"  + sample_name)), ebins )
    np.save(str(results_dir / ("pfns_"   + sample_name)), pfns )
    np.save(str(results_dir / ("A_"      + sample_name)), nubarA[0] )
    np.save(str(results_dir / ("nuA_"    + sample_name)), nubarA[1] )


def calculate_ensemble_nubar(sample_name : str, results_dir : Path):
    nu   = np.load(str(results_dir / ("nu_"     + sample_name + ".npy")))
    pnu  = np.load(str(results_dir / ("pnu_"    + sample_name + ".npy")))
    return np.dot(nu,pnu)
