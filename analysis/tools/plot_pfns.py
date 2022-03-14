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

#data_dir  = "/home/beykyle/scratch/omp_uq/cf_252_sf/default/out_960kh_98s"
#data_dir  = "/home/beykyle/scratch/omp_uq/cf_252_sf/out_384kh_415s"
data_dir  = "/home/beykyle/scratch/omp_uq/u235_nth_f/out_384kh_415s"
workdir   = './'
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'

nsamples = 415
nebins   = 99

p = np.zeros((nsamples,nebins))

f_e = data_dir + "/ebins_1.npy"
ebins = np.load(f_e)

for i in range(1,nsamples):
    f_p = data_dir + "/pfns_" + str(i) + ".npy"
    pfns   = np.load(f_p)
    p[i,:] = pfns

pfns_mean  = np.zeros(nebins)
pfns_stdev = np.zeros(nebins)

for i in range(0,nebins):
    pfns_mean[i]  = np.mean(p[:,i])
    #pfns_stdev[i] = np.sqrt(np.var(p[:,i]) * 960/380 * 99/415)
    pfns_stdev[i] = np.sqrt(np.var(p[:,i]) )

scaling_factor = 1/np.sum(pfns_mean)
pfns_mean *= scaling_factor
pfns_stdev *= scaling_factor

# default scaling
#ebins_0 = ebins
#pfns_0  = pfns_mean

# U235
ebins_0 = np.load(data_dir + "/../default/ebins.npy")
pfns_0  = normalize(np.load(data_dir + "/../default/pfns.npy"))

# cf252
#ebins_0 = np.load(data_dir + "/../default/out_960kh_98s/ebins_0.npy")
#pfns_0  = normalize(np.load(data_dir + "/../default/out_960kh_98s/pfns_0.npy"))

pct_dev = 100*(pfns_mean - pfns_0)/pfns_0

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':  [3,1]}, figsize=(8,6))

ax1.set_yscale("log")
ax1.step(ebins, pfns_mean, where="mid" , label="mean")
ax1.fill_between(ebins, pfns_mean - pfns_stdev, pfns_mean + pfns_stdev,
       facecolor='green', alpha=0.5, edgecolor='none', label="1 standard deviation")
ax1.fill_between(ebins, pfns_mean - 2*pfns_stdev, pfns_mean + 2*pfns_stdev,
     facecolor='green', alpha=0.2, edgecolor='none', label="2 standard deviations")
ax1.step(ebins_0, pfns_0, where="mid", label="default")
ax1.set_xlim(0.08,20)
ax1.set_ylim(10E-6 * scaling_factor , scaling_factor)
ax1.set_ylabel(r'$\chi(E)$')
#ax1.set_yticks([])
ax1.legend()

ax2.step(ebins, pct_dev, where="mid" )
ax2.fill_between(ebins, pct_dev + 100*pfns_stdev/pfns_0, pct_dev - 100*pfns_stdev/pfns_0,
        facecolor='green', alpha=0.2, edgecolor='none')
ax2.fill_between(ebins, pct_dev + 200*pfns_stdev/pfns_0, pct_dev - 200*pfns_stdev/pfns_0,
        facecolor='green', alpha=0.2, edgecolor='none')
ax2.set_ylim(-150,150)
ax2.set_ylabel('$\Delta \chi(E) / \chi(E)$ [%]')
#ax2.set_yticks([])
ax2.set_xlabel('Outgoing neutron energy (MeV)')

plt.tight_layout()
plt.xscale('log')


out_dir = "/home/beykyle/scratch/omp_uq/"
#plt.savefig("pfns_cf252.pdf")
#plt.savefig("pfns_cf252.png")
plt.savefig(out_dir + "pfns_u235.pdf")
plt.savefig(out_dir + "pfns_u235.png")
#plt.savefig("pfns_scaling.pdf")
#plt.savefig("pfns_scaling.png")
