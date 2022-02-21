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
outdir    = "./run2/"
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'

nevents = int(1E2)
nubars   = []
nubars2 = []

for filename in os.scandir(data_dir):
    if filename.is_file():
        print(filename.path)

        # run CGMF
        cmd = "mpirun -np 8 --use-hwthread-cpus cgmf.mpi.x -t -1 -i 98252 -e 0.0 -n " + str(nevents) +  " -o " + str(filename.path)
        os.system(cmd)
        os.system("cat histories.cgmf.* > histories.out")
        os.system("rm histories.cgmf.*")

        # read histories
        sample_number = str(re.findall(r'\d+', filename.name)[-1])
        hist = fh.Histories(workdir + histFile, nevents=nevents)
        nu, pnu    = hist.Pnu()
        ebins,pfns = hist.pfns()
        nubarA     = hist.nubarA()

        nubar  = np.dot(nu,pnu)
        nubar2 = np.sqrt(np.dot(pnu, (nu - nubar)**2))
        nubars.append(nubar)
        nubars2.append(nubar2)

        np.save(outdir + "nu_"     + sample_number, nu )
        np.save(outdir + "pnu_"    + sample_number, pnu )
        np.save(outdir + "ebins_"  + sample_number, ebins )
        np.save(outdir + "pfns_"   + sample_number, pfns )
        np.save(outdir + "A_"      + sample_number, nubarA[0] )
        np.save(outdir + "nuA_"    + sample_number, nubarA[1] )

np.save("nubars"  , np.array(nubars))
np.save("nubars2" , np.array(nubars2))
os.system("rm histories.out")
