import numpy as np
import os
import re
import matplotlib.pyplot as plt
from CGMFtk import histories as fh

import matplotlib as mpl

nubar_0 = 3.7506911764705886

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

data_dir  = "/home/beykyle/umich/omp-uq/analysis/nu"
workdir   = './'
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'
nevents = int(1E4)

nu_fname  = data_dir +  "/nubars.npy"
nu2_fname  = data_dir +  "/nubars2.npy"
n = np.load(nu_fname)
n2 = np.load(nu2_fname)

#plt.plot([(3.7676-nubar_0)/nubar_0,(3.7676-nubar_0)/nubar_0],[0,100], "--", label="ENDF/B-VI.8")
plt.hist(100*(np.array(n)-nubar_0)/nubar_0)
plt.xlim(-1.2,1.2)
plt.xlabel(r"$\Delta \bar{\nu} / \bar{\nu}$ [%]")
plt.ylabel(r"frequency")
plt.legend()
plt.tight_layout()
plt.savefig("pnubar.pdf")
plt.close()

plt.tight_layout()
plt.hist(100*n2/nubar_0)
plt.xlim(-1.2,1.2)
plt.xlabel(r"$ \sigma_{\nu}} /\bar{\nu}_{ENDF}$ [%]")
plt.ylabel(r"frequency")
plt.tight_layout()
plt.savefig("pnubar2.pdf")
plt.close()

print(np.mean(n))
print(np.mean(n2))
print(np.sqrt(np.var(n)))
