import numpy as np
import os
import re
import sys
import argparse
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

cf252_nufile         = "/home/beykyle/scratch/omp_uq/cf_252_sf/out_384kh_415s/nubars.npy"
cf252_nufile_default = "/home/beykyle/scratch/omp_uq/cf_252_sf/default/out_960kh_98s/nubars.npy"
u235_nufile         = "/home/beykyle/scratch/omp_uq/u235_nth_f/out_384kh_415s/nubars.npy"
u235_nufile_default = "/home/beykyle/scratch/omp_uq/u235_nth_f/default/nu.npy"

def add_plot(nu_fname, def_nu_fname, lbl):
    def_n = np.mean(np.load(def_nu_fname))
    nubar_0 = np.mean(def_n)
    n = np.load(nu_fname)
    n, bins, _ = plt.hist(np.array(n), align="mid", label=lbl)
    plt.plot([def_n, def_n], [0, max(n)], "--", label=(lbl + " default"))

add_plot(cf252_nufile, cf252_nufile_default, "Cf-252")
add_plot(u235_nufile, u235_nufile_default, "U-235")

plt.xlabel(r"$\bar{\nu}$ [neutrons]")
plt.ylabel(r"frequency")
plt.legend()
plt.tight_layout()
plt.savefig("pnubar_all.pdf")
