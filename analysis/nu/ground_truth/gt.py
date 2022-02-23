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

data_dir  = "/home/beykyle/db/projects/OM/KDOMPuq/"
workdir   = './'
outdir    = "./"
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'

nevents = int(1E4)
nubars   = []
nubars2 = []

filename = data_dir + "KDGlobal.json"
cmd = "mpirun -np 8 --use-hwthread-cpus cgmf.mpi.x -t -1 -i 98252 -e 0.0 -n " + str(nevents) +  " -o " + str(filename)
os.system(cmd)
os.system("cat histories.cgmf.* > histories.out")
os.system("rm histories.cgmf.*")

# read histories
hist = fh.Histories(workdir + histFile, nevents=nevents)

nu          = hist.getNutot()
nubins, pnu = hist.Pnu()
ebins,pfns  = hist.pfns()
nubarA      = hist.nubarA()

nubar  = np.mean(nu)
nubar2 = np.sqrt(np.var(nu))
nubars.append(nubar)
nubars2.append(nubar2)

np.save(outdir + "nu"     , nubins )
np.save(outdir + "pnu"    , pnu )
np.save(outdir + "ebins"  , ebins )
np.save(outdir + "pfns"   , pfns )
np.save(outdir + "A"      , nubarA[0] )
np.save(outdir + "nuA"    , nubarA[1] )

np.save("nubars"  , np.array(nubars))
np.save("nubars2" , np.array(nubars2))
os.system("rm histories.out")
