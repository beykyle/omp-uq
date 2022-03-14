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

def get_fact_moments(moments : np.array, nu : np.array, pnu : np.array):
    assert(nu.shape == pnu.shape)
    assert(np.sum(pnu) - 1.0 < 1E-10)
    fact_moments = np.zeros(moments.shape)
    for i in range(0,len(moments)):
        fact_moments[i] = np.dot(pnu,np.math.factorial(nu[i])/np.math.factorial(nu[i] - moments[i]))

    return fact_moments


def normalize(arr : np.array):
    return arr / np.sum(arr)

#data_dir  = "/home/beykyle/scratch/omp_uq/cf_252_sf/default/out_960kh_98s"
labels = ["Cf-252 (sf)", "U-235 (nth,f)"]
data_dir =  ["/home/beykyle/scratch/omp_uq/cf_252_sf/out_384kh_415s", "/home/beykyle/scratch/omp_uq/u235_nth_f/out_384kh_415s"]

default_data_files =  ["/home/beykyle/scratch/omp_uq/cf_252_sf/default/out_960kh_98s/pnu_8.npy", "/home/beykyle/scratch/omp_uq/u235_nth_f/default/pnu.npy"]

nplots = len(labels)
nsamples = 415
nbins    = 8
nmoments = 6
bins    = np.array(range(0,nbins))
moments = np.array(range(0,nmoments))

fm        = np.zeros((nplots,nsamples,nmoments))
pnu       = np.zeros((nplots,nsamples,nbins))
pnu_mean  = np.zeros((nplots,nbins))
pnu_stdev = np.zeros((nplots,nbins))

for j in range(0,nplots):
    for i in range(1,nsamples):
        fname     = data_dir[j]  + "/pnu_" + str(i) + ".npy"
        pnu_d     = np.load(fname)
        fm[j,i,:] = get_fact_moments(moments, bins, pnu_d)
        pnu[j,i,0:pnu_d.shape[0]] = pnu_d


    for i in range(0,nbins):
        pnu_mean[j,i]  = np.mean(pnu[j,:,i])
        #pnu_stdev[i] = np.sqrt(np.var(p[:,i]) * 960/380 * 99/415)
        pnu_stdev[j,i] = np.sqrt(np.var(pnu[j,:,i]) )

    pnu_0  = np.zeros(nbins)
    pnu_0_tmp = np.load(default_data_files[j])
    pnu_0[0:pnu_0_tmp.shape[0]] = pnu_0_tmp
    print(pnu_0)
    print(pnu_mean[j])

    #plt.errorbar(bins, 100*(pnu_mean[j] - pnu_0)/pnu_0,
    #             yerr=(100*pnu_stdev[j]/pnu_0), marker="*" , label=labels[j])
    p1 = plt.errorbar(bins, pnu_mean[j],
                 yerr=pnu_stdev[j], label=labels[j], zorder=0)
    plt.plot(bins, pnu_0, marker="." , linestyle="none", markersize=12,
            label=labels[j] + " default", zorder=99, color=p1[0].get_color())

#plt.ylabel(r'$\frac{\Delta P(\nu)}{P(\nu)}$ [%]')
plt.xlim([0,9])
plt.ylabel(r"$P(\nu)$")
plt.xlabel(r"$\nu$")
plt.legend()
plt.tight_layout()
out_dir = "/home/beykyle/scratch/omp_uq/"
plt.savefig("pnu.pdf")
plt.savefig("pnu.png")
