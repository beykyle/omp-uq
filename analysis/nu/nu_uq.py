import numpy as np
import os
import re
import matplotlib.pyplot as plt
from CGMFtk import histories as fh

import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'Helvetica','serif'
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['axes.labelsize'] = 18.
mpl.rcParams['xtick.labelsize'] = 18.
mpl.rcParams['ytick.labelsize'] = 18.
mpl.rcParams['lines.linewidth'] = 2.
mpl.rcParams['xtick.major.pad'] = '10'
mpl.rcParams['ytick.major.pad'] = '10'
mpl.rcParams['image.cmap'] = 'BuPu'

data_dir  = "/home/beykyle/db/projects/OM/KDOMPuq/KDUQSamples"
workdir   = './'
outdir    = "./run4/"
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'

nevents = int(1E4)
nubars   = []
nubars2 = []

for i in range(1,99):
    # run CGMF
    sample_number = str(i)
    filename = data_dir + "/" + sample_number + ".json"
    cmd = "time mpirun -np 8 --use-hwthread-cpus cgmf.mpi.x -t -1 -i 98252 -e 0.0 -n " + str(nevents) +  " -o " + str(filename)
    os.system(cmd)
    os.system("cat histories.cgmf.* > histories.out")
    os.system("rm histories.cgmf.*")

    # read histories
    hist = fh.Histories(workdir + histFile, nevents=nevents)

    # save histories for sample
    os.system("mv histories.out " + outdir + "/histories_" + sample_number + ".out" )

    # extract some data from history files for immediate post-processing
    nu          = hist.getNutot()
    nubins, pnu = hist.Pnu()
    ebins,pfns  = hist.pfns()
    nubarA      = hist.nubarA()

    nubar  = np.mean(nu)
    nubar2 = np.sqrt(np.var(nu))
    nubars.append(nubar)
    nubars2.append(nubar2)

    # save compressed post-processed distributions
    np.save(outdir + "nu_"     + sample_number, nubins )
    np.save(outdir + "pnu_"    + sample_number, pnu )
    np.save(outdir + "ebins_"  + sample_number, ebins )
    np.save(outdir + "pfns_"   + sample_number, pfns )
    np.save(outdir + "A_"      + sample_number, nubarA[0] )
    np.save(outdir + "nuA_"    + sample_number, nubarA[1] )

# final saves and cleanup
np.save("nubars"  , np.array(nubars))
np.save("nubars2" , np.array(nubars2))
os.system("rm histories.out")
