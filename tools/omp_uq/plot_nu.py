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

from .dataset import DataSetUQUncorr

def get_fact_moments(moments : np.array, nu : np.array, pnu : np.array):
    assert(nu.shape == pnu.shape)
    assert(np.sum(pnu) - 1.0 < 1E-10)
    fact_moments = np.zeros(moments.shape)
    fall_fact = np.zeros(nu.shape)
    for i in range(0,len(moments)):
        for j in range(0,len(nu)):
            if moments[i] <= nu[j]:
                fall_fact[j] = np.math.factorial(nu[j])/np.math.factorial(nu[j] - moments[i])
        fact_moments[i] = np.dot(pnu,fall_fact)

    return fact_moments


def normalize(arr : np.array):
    return arr / np.sum(arr)


def plot_nubar(data_sets, rel=True, save=False, outfile=""):
    max_n = 0
    num_plots = len(data_sets)
    alphas = np.linspace(0.9,0.5,num=num_plots)
    orders = np.arange(0,num_plots*100,100)
    for i,d in enumerate(data_sets):
        if rel:
            n,b,_ = plt.hist(
                    100*(d.nubar-d.nubar_default)/d.nubar_default, label=d.label, alpha=alphas[i], zorder=orders[i])
            if np.max(n) > max_n:
                max_n = np.max(n)
            plt.xlabel(r"$\Delta \bar{\nu} / \bar{\nu}$ [%]")
            plt.xlim(-1,1)
        else:
            plt.hist(d.nubar)
            plt.xlabel(r"$\bar{\nu}$ [neutrons]", label=d.label)

    plt.plot([0,0], [0,max_n], ":")
    plt.ylabel(r"frequency")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(outfile)
    else:
        plt.show()

def plot_pnu(data_sets, save=False, rel=False, outfile=""):

    num_plots = len(data_sets)
    alphas = np.linspace(0.9,0.5,num=num_plots)
    zo = 1
    rel_max = 7
    for i,d in enumerate(data_sets):
        pnu_mean  = np.mean(d.pnu, axis=0)
        pnu_stdev = np.sqrt(np.var(d.pnu, axis=0))
        pnu_0     = d.pnu_default
        bins      = np.arange(0,pnu_mean.shape[0],step=1)
        if rel:
            plt.errorbar(bins[:rel_max], 100*(pnu_mean[:rel_max] - pnu_0[:rel_max])/pnu_0[:rel_max],
                    yerr=(100*pnu_stdev[:rel_max]/pnu_0[:rel_max]),
                    marker="*" , label=d.label, alpha=alphas[i])
        else:
            p1 = plt.errorbar(bins, pnu_mean,
                         yerr=pnu_stdev, label=d.label, zorder=zo)
            zo = zo * 10
            plt.plot(bins, pnu_0, marker="." , linestyle="none", markersize=8,
                    label=d.label + " default", zorder=zo, color=p1[0].get_color())
        zo = zo* 10

    #plt.ylabel(r'$\frac{\Delta P(\nu)}{P(\nu)}$ [%]')
    if rel:
        plt.ylabel(r"$\frac{\Delta P(\nu)}{P(\nu)}$ [%]")
    else:
        plt.ylabel(r"$P(\nu)$")
    plt.xlabel(r"$\nu$")
    plt.legend(loc=2)
    plt.tight_layout()

    if save:
        plt.savefig(outfile)
    else:
        plt.show()

def plot_fm(data_sets, save=False, rel=False, outfile=""):

    num_plots = len(data_sets)
    alphas = np.linspace(0.9,0.5,num=num_plots)

    nmoments = 7
    moments = np.array(range(0,nmoments))
    zo = 1

    for i, d in enumerate(data_sets):
        bins  = d.nu[0,:]
        pnu_0 = d.pnu_default
        fm = np.zeros((d.num_samples, nmoments))
        fm_0  = get_fact_moments(moments, bins, pnu_0)

        # calculate factorial moments
        for j in range(0,d.num_samples):
            fm[j,:] = get_fact_moments(moments, bins, d.pnu[j,:])

        fm_mean   = np.mean(fm, axis=0)
        fm_stdev  = np.sqrt(np.var(fm, axis=0))

        if rel:
            plt.errorbar(moments, 100*(fm_mean - fm_0)/fm_0 , yerr=100*fm_stdev/fm_0,
                    marker="*", label=d.label, alpha=alphas[i])
        else:
            scaling = np.array([np.math.factorial(m) for m in moments])
            p1 = plt.errorbar(moments, fm_mean/scaling,
                              yerr=fm_stdev/scaling, label=d.label, zorder=zo)
            zo = zo * 10
            plt.plot(moments, fm_0/scaling, marker="." , linestyle="none", markersize=8,
                    label=d.label + " default", zorder=zo, color=p1[0].get_color())
            zo = zo * 10

    #plt.ylabel(r'$\frac{\Delta P(\nu)}{P(\nu)}$ [%]')
    plt.xlim([0,nmoments])
    if rel:
        plt.ylabel(r"$\frac{\Delta M_m}{M_m}$ [%]")
    else :
        plt.ylabel(r"E$\left[ \frac{\nu !}{(\nu -m)!}\right] \frac{1}{m!}$")
    plt.xlabel(r"$m$")
    plt.legend(loc=2)
    plt.tight_layout()

    if save:
        plt.savefig(outfile)
    else:
        plt.show()
