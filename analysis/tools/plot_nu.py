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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action="store_true", default=False, help="If the -s flag is present, the plots are saved rather than shown")
    parser.add_argument('-r', action="store_true", default=False, help="If the -r flag is present, plot relative deviation from default")
    parser.add_argument('-n', '--nu-file', type=str, help="nubars file for ensembles", dest="nu_file")
    parser.add_argument('-d', '--default-nu-file', type=str, help="default nubars ensemble",  dest="default_nu_file")
    parser.add_argument('-o', '--out-file', type=str, help="save figure to this file name",  dest="out_file", default="pnubar.pdf")


    args = parser.parse_args()
    nu_fname     = args.nu_file
    def_nu_fname = args.default_nu_file

    def_n = np.load(def_nu_fname)
    nubar_0 = np.mean(def_n)

    print("nubar default {:1.6f}".format( nubar_0))

    n = np.load(nu_fname)

    if args.r:
        plt.hist(100*(np.array(n)-nubar_0)/nubar_0)
        plt.xlabel(r"$\Delta \bar{\nu} / \bar{\nu}$ [%]")
        plt.xlim([-1.2,1.2])
    else:
        plt.hist(np.array(n))
        plt.xlabel(r"$\bar{\nu}$ [neutrons]")

    plt.ylim([0,120])
    plt.ylabel(r"frequency")
    plt.legend()
    plt.tight_layout()

    if args.s:
        plt.savefig(args.out_file)
    else:
        plt.show()
