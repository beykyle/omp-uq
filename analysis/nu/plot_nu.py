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

data_dir  = "/home/beykyle/umich/omp-uq/analysis/pfns/data"
workdir   = './'
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'
nevents = int(1E4)

nu  = []
pnu = []


for filename in os.scandir(data_dir):
    if filename.is_file():
        n = np.load(filename.path)
        p = np.load(filename.path)
        nubar = np.dot(n,p)
        print(nubar)
