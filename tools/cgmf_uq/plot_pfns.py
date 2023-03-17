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

def plot_pfns(dataset : list, rel=True, outfile=""):

    plots  = []
    labels = []
    for d in dataset:
        pfns_stdev  = np.sqrt(np.var(d.pfns, axis=0))
        pfns_mean   = np.mean(d.pfns, axis=0)
        pfns_0      = d.pfns_default
        ebins       = d.ebins[0,:]
        ebins_0     = d.ebins[0,:]
        maxx = maxwellian(ebins,1.32)
        pfns_mean /= maxx
        pfns_stdev /= maxx
        p1 = plt.step(ebins, pfns_mean)
        plots.append(p1)
        labels.append(d.label)
        #plt.step(ebins, pfns_0, label=d.label + " default")
        plt.fill_between(ebins, pfns_mean, pfns_mean - pfns_stdev,
                pfns_mean+pfns_stdev, alpha=0.4)

    plt.ylabel(r"PFNS ($kT = 1.32$ [MeV])")
    plt.xlabel(r"$E_{lab}$ [MeV]")
    plt.legend(plots, labels)
