import sys
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
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
    def __init__(self, zaid):
        self.a, self.z = from_zaid(zaid)
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
            nuc_histories[zaid] = NucHistories(zaid)
            nuc_histories[zaid].nu.append(nu)
            nuc_histories[zaid].nug.append(nug)
            nuc_histories[zaid].ne.append(ne)
            nuc_histories[zaid].ge.append(ge)

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
            nuc_histories[a].nu.append(nu)
            nuc_histories[a].nug.append(nug)
            nuc_histories[a].ne.append(ne)
            nuc_histories[a].ge.append(ge)

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
            nuc_histories[a].nu.append(nu)
            nuc_histories[a].nug.append(nug)
            nuc_histories[a].ne.append(ne)
            nuc_histories[a].ge.append(ge)

    return nuc_histories

def selectElement(Z, nuc_histories, min_hists=0):
    A = []
    for zaid in nuc_histories.keys():
        a,z  = from_zaid(zaid)
        if z == Z and len(nuc_histories[zaid].nu) > min_hists:
            A.append(a)

    hists = [ nuc_histories[to_zaid(a,Z)] for a in A ]
    zipped = zip(A, hists)
    sz = sorted(zipped)
    return zip(*sz)

def extract( history_list: list , analysis ):
    result = []
    for h in history_list:
        result.append(analysis(h))
    return result

def plot_element_multiplicity(Z, hist_by_frag, label=None, save=False, min_hists=0):
    # look at 1st neutron energy isotope by isotope
    A, Z_hists_by_A = selectElement(Z, hist_by_frag, min_hists=min_hists) # Z isotopes
    print(A)
    if len(A) == 0:
        return

    print(label)

    mean_nu_frag = np.array(extract( Z_hists_by_A,
        lambda hists : np.mean(np.array(hists.nu))), dtype=float)
    stdev_nu_frag = np.array(extract( Z_hists_by_A,
        lambda hists : np.sqrt(np.var(np.array(hists.nu)))), dtype=float)

    alpha = [ (a-2*Z)/a for a in A ]

    plt.errorbar(A, mean_nu_frag, yerr=stdev_nu_frag, linestyle="None")
    plt.scatter(A, mean_nu_frag, marker=".", label=label)


def plot_sf_mult_dist(hist, save=False, title=None):
    nu_bins = range(0,np.max(hist.nu))
    nug_bins = range(0,np.max(hist.nug))
    n, b , _ = plt.hist(hist.nu, bins=nu_bins, label="n")

def compare_fragment_mult(hist_by_nuc):
    elements = ["Zr", "Nb", "Mo", "Tc"]
    for i, Z in enumerate([40, 41, 42, 43]):
        plot_element_multiplicity(Z, hist_by_nuc, save=False, min_hists=1000, label=elements[i])

    plt.xlabel("A [u]")
    plt.ylabel(r"Single Fragment Multiplicity")
    plt.tight_layout()
    plt.legend()
    plt.savefig("AL_mult_sf.png")
    plt.close()

    elements = ["I", "Xe", "Ce" , "Ba"]
    for i, Z in enumerate([53, 54, 55, 56]):
        plot_element_multiplicity(Z, hist_by_nuc, save=False, min_hists=1000, label=elements[i])

    plt.xlabel("A [u]")
    plt.ylabel(r"Single Fragment Multiplicity")
    plt.tight_layout()
    plt.legend()
    plt.savefig("AH_mult_sf.png")
    plt.close()

    elements = ["Pd", "Ag", "Sn", "Sb"]
    for i, Z in enumerate([46, 47, 50, 51]):
        plot_element_multiplicity(Z, hist_by_nuc, save=False, min_hists=1000, label=elements[i])

    plt.xlabel("A [u]")
    plt.ylabel(r"Single Fragment Multiplicity")
    plt.tight_layout()
    plt.legend()
    plt.savefig("AMiddle_mult_sf.png")
    plt.close()

if __name__ == "__main__":

    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['font.family'] = 'Helvetica','serif'
    matplotlib.rcParams['font.weight'] = 'normal'
    matplotlib.rcParams['axes.labelsize'] = 18.
    matplotlib.rcParams['xtick.labelsize'] = 18.
    matplotlib.rcParams['ytick.labelsize'] = 18.
    matplotlib.rcParams['lines.linewidth'] = 2.
    matplotlib.rcParams['xtick.major.pad'] = '10'
    matplotlib.rcParams['ytick.major.pad'] = '10'
    matplotlib.rcParams['image.cmap'] = 'BuPu'

    hist = fh.Histories(sys.argv[1])
    nhist = len(hist.getFissionHistories())
    print("Fragment histories: {}".format(nhist))

    hist_by_nuc = sortHistoriesByFragment(hist)

    #xe142 = 54142
    #xe142_hists = hist_by_nuc[xe142]
    #plot_sf_mult_dist(xe142_hists, save=True)
    #print("142Xe Histories: {}".format(len(xe142_hists.nu)))

    #for Z in range(33,63):
    #   plot_element_multiplicity(Z, hist_by_nuc, save=True, min_hists=500)

    #hist_by_frag_mass = sortHistoriesByFragmentMass(hist, post_emission=True)
    A, hists_by_A = zip(*hist_by_frag_mass.items())
