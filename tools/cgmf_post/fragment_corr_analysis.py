import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from CGMFtk import histories as fh

def to_zaid(a,z):
    return 1000*z + a

def from_zaid(zaid):
    z = int(zaid//1000)
    a = int(zaid%1000)
    return a,z

class NucHistories:
    def __init__(self):
        self.nu  = []
        self.nug = []
        self.ne  = []
        self.ge  = []

def sortHistoriesByFragment( hist ):
    nuc_histories = {}

    # sort histories by isotope
    for a, z, nu, nug, ne, ge in zip(hist.getA(), hist.getZ(), hist.getNu(), hist.getNug(), hist.getNeutronEcm(), hist.getGammaEcm()):
        zaid = to_zaid(a,z)
        if zaid in nuc_histories:
            nuc_histories[zaid].nu.append(nu)
            nuc_histories[zaid].nug.append(nug)
            nuc_histories[zaid].ne.append(ne)
            nuc_histories[zaid].ge.append(ge)
        else:
            nuc_histories[zaid] = NucHistories()

    return nuc_histories

def sortHistoriesByFragmentMass( hist , post_emission=False):
    nuc_histories = {}

    # sort histories by isotope
    for a, nu, nug, ne, ge in zip(hist.getA(), hist.getNu(), hist.getNug(), hist.getNeutronEcm(), hist.getGammaEcm()):
        if post_emission:
            a = a - nu
        if a in nuc_histories:
            nuc_histories[a].nu.append(nu)
            nuc_histories[a].nug.append(nug)
            nuc_histories[a].ne.append(ne)
            nuc_histories[a].ge.append(ge)
        else:
            nuc_histories[a] = NucHistories()

    return nuc_histories

def sortHistoriesByFragmentCharge( hist ):
    nuc_histories = {}

    # sort histories by isotope
    for z, nu, nug, ne, ge in zip(hist.getZ(), hist.getNu(), hist.getNug(), hist.getNeutronEcm(),  hist.getGammaEcm()):
        if z in nuc_histories:
            nuc_histories[z].nu.append(nu)
            nuc_histories[z].nug.append(nug)
            nuc_histories[z].ne.append(ne)
            nuc_histories[z].ge.append(ge)
        else:
            nuc_histories[z] = NucHistories()

    return nuc_histories

def selectElement(Z, nuc_histories):
    A = []
    for zaid in nuc_histories.keys():
        a,z  = from_zaid(zaid)
        if z == Z:
            A.append(a)

    return A, [ nuc_histories[to_zaid(a,Z)] for a in A ]

def extract( history_list: list , analysis ):
    result = []
    for h in history_list:
        result.append(analysis(h))
    return result

if __name__ == "__main__":
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

    hist = fh.Histories(sys.argv[1])
    nhist = len(hist.getFissionHistories())
    print("Fragment histories: {}".format(nhist))


    # look at 1st neutron energy isotope by isotope
    hist_by_frag = sortHistoriesByFragment(hist)
    A, xe_hists_by_A = selectElement(54, hist_by_frag) # Xe isotopes

    print(A)
    mean_nu_frag = np.array(extract( xe_hists_by_A, lambda hists : np.mean(np.array(hists.nu)), dtype=float)
    mean_nug_frag = np.array(extract( xe_hists_by_A, lambda hists : np.mean(np.array(hists.nug)), dtype=float)


    plt.plot(A, mean_n1_energy_xe)
    plt.xlabel("A [u]")
    plt.ylabel(r"$\nu$ [particles]")
    plt.show()

    hist_by_frag_mass = sortHistoriesByFragmentMass(hist, post_emission=True)
    A, hists_by_A = zip(*hist_by_frag_mass.items())