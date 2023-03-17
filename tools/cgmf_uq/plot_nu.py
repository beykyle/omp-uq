import numpy as np
import os
import re
import sys
import argparse
import matplotlib.pyplot as plt
from CGMFtk import histories as fh

from .dataset import DataSetUQUncorr

def get_fact_moments(moments : np.array, nu : np.array, pnu : np.array):
    assert(nu.shape == pnu.shape)
    assert(np.sum(pnu) - 1.0 < 1E-10)
    nu = np.array(nu, dtype=int)
    fact_moments = np.zeros(moments.shape)
    fall_fact = np.zeros(nu.shape)
    for i in range(0,len(moments)):
        for j in range(0,len(nu)):
            if moments[i] <= nu[j]:
                fall_fact[j] = np.math.factorial(int(nu[j]))/np.math.factorial(int(nu[j]) - int(moments[i]))
        fact_moments[i] = np.dot(pnu,fall_fact)

    return fact_moments

def normalize(arr : np.array):
    return arr / np.sum(arr)

def plot_nua(data_sets,  outfile=""):
    plots = []
    labels = []
    for d in data_sets:
        labels.append(d.label)
        a = d.a[0,:]
        pnu_mean  = np.mean(d.nua, axis=0)
        pnu_stdev = np.sqrt(np.var(d.nua, axis=0))
        plots.append(plt.errorbar(a, pnu_mean, yerr=pnu_stdev, label=d.label))
    plt.xlabel(r"A [u]")
    plt.ylabel(r"$\bar{\nu} | A $ [neutrons]")
    plt.legend(plots, labels)

def plot_nubar(data_sets, rel=True,  outfile="", endf=None):
    plots = []
    labels = []
    max_n = 0
    num_plots = len(data_sets)
    alphas = np.linspace(0.9,0.4,num=num_plots)
    orders = np.arange(0,num_plots*100,100)
    ma = 0
    for i,d in enumerate(data_sets):
        labels.append(d.label)
        if rel:
            n,b,_ = plt.hist(
                    100*(d.nubar-d.nubar_default)/d.nubar_default,
                    label=d.label, alpha=alphas[i], zorder=orders[i], density=True)
            plots.append(n)
            if np.max(n) > max_n:
                max_n = np.max(n)
            plt.xlabel(r"$\Delta \bar{\nu} / \bar{\nu}$ [%]")
            plt.xlim(-1,1)
            plt.plot([0,0], [0,max_n], ":")
        else:
            h,e = np.histogram(d.nubar, density=True)
            de = e[1:] - e[:-1]
            h = h / np.sum(h)
            plots.append(plt.fill_between(0.5*(e[:-1] + e[1:]) , 0, 100*h, \
                                          label=d.label, alpha=alphas[i], zorder=orders[i], step="pre"))
            if np.max(h) > ma:
                ma = np.max(h)
            plt.xlabel(r"$\bar{\nu}$ [neutrons]")

    if endf is not None:
        plt.plot([endf, endf], [0,ma], label="ENDF/B-VI.8", linestyle="--")
    plt.ylabel(r"$P(\bar{\nu})$ [%]")
    plt.legend(plots, labels)

def plot_pnu(data_sets,  rel=False, outfile=""):

    plots = []
    labels = []
    num_plots = len(data_sets)
    alphas = np.linspace(0.9,0.5,num=num_plots)
    rel_max = 7
    for i,d in enumerate(data_sets):
        pnu_mean  = np.mean(d.pnu, axis=0)
        pnu_stdev = np.sqrt(np.var(d.pnu, axis=0))
        print(pnu_stdev/pnu_mean)
        pnu_0     = d.pnu_default
        bins      = np.arange(0,pnu_mean.shape[0],step=1)
        if rel:
            p1 = plt.errorbar(bins[:rel_max], 100*(pnu_mean[:rel_max] - pnu_0[:rel_max])/pnu_0[:rel_max],
                    yerr=(100*pnu_stdev[:rel_max]/pnu_0[:rel_max]),
                    marker="*" , label=d.label, alpha=alphas[i])
        else:
            p1 = plt.errorbar(bins, pnu_mean, marker="x",
                         yerr=pnu_stdev, label=d.label)
        plots.append(p1)
        labels.append(d.label)

    #plt.ylabel(r'$\frac{\Delta P(\nu)}{P(\nu)}$ [%]')
    if rel:
        plt.ylabel(r"$\frac{\Delta P(\nu)}{P(\nu)}$ [%]")
    else:
        plt.ylabel(r"$P(\nu)$")
        #plt.yscale("log")
    plt.xlabel(r"$\nu$ [neutrons]")
    plt.legend(plots, labels)

def plot_fm(data_sets,  rel=False, outfile=""):

    plots = []
    labels = []
    num_plots = len(data_sets)
    alphas = np.linspace(0.9,0.5,num=num_plots)

    nmoments = 7
    moments = np.array(range(0,nmoments), dtype=int)

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
            p1 = plt.errorbar(moments, 100*(fm_mean - fm_0)/fm_0 , yerr=100*fm_stdev/fm_0,
                    marker="*", label=d.label, alpha=alphas[i])
        else:
            scaling = np.array([np.math.factorial(m) for m in moments])
            p1 = plt.errorbar(moments, fm_mean/scaling, marker=".",
                              yerr=fm_stdev/scaling, label=d.label)
        plots.append(p1)
        labels.append(d.label)

    #plt.ylabel(r'$\frac{\Delta P(\nu)}{P(\nu)}$ [%]')
    plt.xlim([0,nmoments])
    if rel:
        plt.ylabel(r"$\frac{\Delta M_m}{M_m}$ [%]")
    else :
        plt.ylabel(r"E$\left[ \frac{\nu !}{(\nu -m)!}\right] \frac{1}{m!}$")
    plt.xlabel(r"$m$")
    plt.legend(plots, labels)
