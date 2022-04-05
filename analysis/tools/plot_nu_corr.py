import numpy as np
import os
import re
import sys
import argparse
import matplotlib.pyplot as plt
from CGMFtk import histories as fh

import matplotlib as mpl


from dataset import DataSet

def plot_nu_corr_dev(d, save=False, rel=False, outfile=""):
    mpl.rcParams['font.size'] = 22
    mpl.rcParams['font.family'] = 'Helvetica','serif'
    mpl.rcParams['font.weight'] = 'normal'
    mpl.rcParams['axes.labelsize'] = 24.
    mpl.rcParams['xtick.labelsize'] = 22.
    mpl.rcParams['ytick.labelsize'] = 22.
    mpl.rcParams['lines.linewidth'] = 2.
    mpl.rcParams['xtick.major.pad'] = '10'
    mpl.rcParams['ytick.major.pad'] = '10'
    mpl.rcParams['image.cmap'] = 'BuPu'

    hist_shape = (21,8)
    bins = [np.arange(0,hist_shape[0],step=1,dtype=int),
            np.arange(0,hist_shape[1],step=1,dtype=int)]

    histogram_layers = np.zeros((253,hist_shape[0]-1,hist_shape[1]-1))
    for i in range(0,253):
        h,_,_ = np.histogram2d(d.event_nu_p[i,:], d.event_nu_n[i,:], bins=bins)
        histogram_layers[i,:,:] = h/d.num_hist


    np_default,_,_ = np.histogram2d(d.event_nu_p_default, d.event_nu_n_default, bins=bins)
    np_default = np_default / d.num_hist
    np_mean  = np.mean(histogram_layers, axis=0)
    np_stdev = np.sqrt(np.var(histogram_layers, axis=0))

    pct_dev = 100*(np_default - np_mean)/np_mean
    pct_sig = 100*np_stdev/np_mean
    pct_sig = np.where( pct_sig < 100, pct_sig, 0)

    fig = plt.figure(figsize=(21,8))
    ax = fig.gca()
    ax.set_xticks([0,5,20,10])
    ax.set_xlim([0,hist_shape[0]])
    ax.set_ylim([0,hist_shape[1]])
    ax.set_xlabel(r"$\nu_p$")
    ax.set_ylabel(r"$\nu_n$")
    im = ax.imshow(np.rot90(np_stdev), cmap="coolwarm", extent=[0,hist_shape[0],0,hist_shape[1]])
    cbar = plt.colorbar(im)
    cbar.set_label(r"1 standard deviation in $P(\nu_p,\nu_n)$")

    if save:
        plt.savefig(outfile)
    else:
        plt.show()
