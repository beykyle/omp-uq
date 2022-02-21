import numpy as np
import os
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
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'

nevents = int(1E4)

nu  = []
pnu = []

for filename in os.scandir(data_dir):
    #if filename.is_file() and "42" in filename.path:
    if filename.is_file():
        print(filename.path)

        # run CGMF
        cmd = "mpirun -np 8 --use-hwthread-cpus cgmf.mpi.x -t -1 -i 98252 -e 0.0 -n " + str(nevents) +  " -o " + str(filename.path)
        os.system(cmd)
        os.system("cat histories.cgmf.* > histories.out")
        os.system("rm histories.cgmf.*")

        # read histories
        hist = fh.Histories(workdir + histFile, nevents=nevents)
        n, p= hist.Pnu()
        np.save("nu_" + filename.path[-7:-5] , n )
        np.save("pnu_" + filename.path[-7:-5], p )



