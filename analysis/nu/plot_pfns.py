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

data_dir  = "/home/beykyle/umich/omp-uq/analysis/nu/run4"
workdir   = './'
histFile  = 'histories.out'
timeFile  = 'histories.out'
yeildFile = 'yeilds.cgmf.0'

p = np.zeros((47,99))

f_e = data_dir + "/ebins_1.npy"
ebins = np.load(f_e)

for i in range(1,47):
    f_p = data_dir + "/pfns_" + str(i) + ".npy"
    pfns   = np.load(f_p)
    p[i,:] = pfns

pfns_mean  = np.zeros(99)
pfns_stdev = np.zeros(99)

for i in range(0,98):
    pfns_mean[i]  = np.mean(p[:,i])
    pfns_stdev[i] = np.sqrt(np.var(p[:,i]))

ebins_0 = np.load(data_dir + "/../ground_truth/ebins.npy")
pfns_0  = np.load(data_dir + "/../ground_truth/pfns.npy")

pct_dev = 100*(pfns_mean - pfns_0)/pfns_0

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':  [3,1]}, figsize=(8,6))

ax1.set_yscale("log")
ax1.step(ebins, pfns_mean, where="mid" , label="mean")
ax1.fill_between(ebins, pfns_mean - pfns_stdev, pfns_mean + pfns_stdev,
       facecolor='green', alpha=0.5, edgecolor='none', label="1 standard deviation")
ax1.fill_between(ebins, pfns_mean - 2*pfns_stdev, pfns_mean + 2*pfns_stdev,
     facecolor='green', alpha=0.2, edgecolor='none', label="2 standard deviations")
ax1.step(ebins_0, pfns_0, where="mid", label="default")
ax1.set_xlim(0.08,15)
ax1.set_ylabel('PFNS')
ax1.legend()

ax2.step(ebins, pct_dev, where="mid" )
ax2.fill_between(ebins, pct_dev + 100*pfns_stdev/pfns_0, pct_dev - 100*pfns_stdev/pfns_0,
        facecolor='green', alpha=0.2, edgecolor='none')
ax2.fill_between(ebins, pct_dev + 200*pfns_stdev/pfns_0, pct_dev - 200*pfns_stdev/pfns_0,
        facecolor='green', alpha=0.2, edgecolor='none')
ax2.set_ylim(-80,80)
ax2.set_ylabel('$\Delta P(E) / P(E)$ [%]')
ax2.set_xlabel('Outgoing neutron energy (MeV)')

plt.tight_layout()
plt.xscale('log')
plt.savefig("pfns_all.pdf")
