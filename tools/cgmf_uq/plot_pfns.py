import numpy as np
import os
import re
import matplotlib.pyplot as plt
from CGMFtk import histories as fh

import matplotlib as mpl

def normalize(arr : np.array):
    return arr / np.sum(arr)

from .dataset import DataSetUQUncorr

def maxwellian(ebins : np.array, Eavg : float):
    return 2*np.sqrt(ebins / np.pi) * (1 / Eavg)**(3./2.) * np.exp(- ebins / Eavg )

def plot_pfns(dataset : list, rel=True, save=True, outfile=""):

    if len(dataset) == 1:
        d  =dataset[0]
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
    else:
        for d in dataset:
            pfns_stdev  = np.sqrt(np.var(d.pfns, axis=0))
            pfns_mean   = np.mean(d.pfns, axis=0)
            pfns_0      = d.pfns_default
            ebins       = d.ebins[0,:]
            ebins_0     = d.ebins[0,:]
            maxx = maxwellian(ebins,1.42)
            pfns_mean /= maxx
            pfns_stdev /= maxx
            plt.step(ebins, pfns_mean, label=d.label)
            #plt.step(ebins, pfns_0, label=d.label + " default")
            plt.fill_between(ebins, pfns_mean, pfns_mean - pfns_stdev,
                    pfns_mean+pfns_stdev, alpha=0.4)

        plt.ylabel(r"$P(E)/M(E,kT=1.42 MeV)$ [a. u.]")
        plt.xlabel(r"$E$ [MeV]")
        plt.legend()
        plt.tight_layout()
        plt.xlim([10E-2,17])
        plt.ylim([1E-2,1.3])
        if save:
            plt.savefig(outfile)
        else:
            plt.show()
