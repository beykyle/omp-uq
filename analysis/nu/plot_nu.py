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

data_dir  = "/home/beykyle/umich/omp-uq/analysis/nu/data"
workdir   = './'
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'
nevents = int(1E4)

nu = []


for i in [11,13,15,25,26,30,31,40,52,54,55,60,61,63,71,75,82,84,88,94]:
    nu_fname  = data_dir +  "/nu_" + str(i) + ".npy"
    pnu_fname = data_dir + "/pnu_" + str(i) + ".npy"
    n = np.load(nu_fname)
    p = np.load(pnu_fname)
    plt.semilogy(n,p)
    nubar = np.dot(n,p)
    nu.append(nubar)

plt.xlabel(r"$\nu$")
plt.ylabel(r"$P(\nu)$")
plt.show()
#plt.savefig("pnu.pdf")


plt.hist(100*(np.array(nu)-3.7676)/3.7676,bins=6)
plt.xlabel(r"Percent relative deviation $\bar{\nu}$")
plt.ylabel(r"frequency")
plt.tight_layout()
plt.savefig("pnubar.pdf")


print(np.mean(nu))
print(np.sqrt(np.var(nu)))
