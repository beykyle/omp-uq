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


def normalize(arr : np.array):
    return arr / np.sum(arr)

from .dataset import DataSetUQUncorr

def plot_pfns(d : DataSetUQUncorr, rel=True, save=True, outfile=""):

    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':  [3,1]}, figsize=(8,6))

    #ax1.set_yscale("log")

    pfns_stdev  = np.sqrt(np.var(d.pfns, axis=0))
    pfns_mean   = np.mean(d.pfns, axis=0)
    pfns_0      = d.pfns_default
    ebins       = d.ebins[0,:]
    ebins_0     = d.ebins[0,:]

    pct_dev = 100*(pfns_mean - pfns_0)/pfns_0

    ax1.errorbar(ebins , pfns_mean, yerr=2*pfns_stdev, marker=".", linestyle="none", label=r"$\langle \chi(E) \rangle \pm 2 \sigma$")
    ax1.step(ebins_0, pfns_0, where="mid", label="default")
    ax1.set_ylabel(r'$\chi(E)$')
    #ax1.set_yscale("log")
    #ax1.set_ylim([1E-9,1])
    ax1.legend()

    ax2.step(ebins, pct_dev, where="mid" )
    ax2.fill_between(ebins, pct_dev + 100*pfns_stdev/pfns_0, pct_dev - 100*pfns_stdev/pfns_0,
            facecolor='green', alpha=0.2, edgecolor='none')
    ax2.fill_between(ebins, pct_dev + 200*pfns_stdev/pfns_0, pct_dev - 200*pfns_stdev/pfns_0,
            facecolor='green', alpha=0.2, edgecolor='none')
    ax2.set_ylim([-25,25])
    ax2.set_ylabel('$\Delta \chi(E) / \chi(E)$ [%]')
    ax2.set_xlabel('Outgoing neutron energy (MeV)')

    plt.tight_layout()
    plt.xscale('log')
    plt.xlim([10E-2,20])

    if save:
        plt.savefig(outfile)
    else:
        plt.show()
