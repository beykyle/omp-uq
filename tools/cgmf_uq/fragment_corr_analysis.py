import sys
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
from pathlib import Path
from CGMFtk import histories as fh
from exp import maxwellian
import pandas as pd

class AMETable:
    def __init__(self):
        self.data_path = Path(__file__).parent.parent.parent / "data"
        self.df = pd.read_csv(  self.data_path / "AME" / "mass_nuc.txt", delim_whitespace=True, index_col=False)
        self.df['EL'] = self.df['EL'].astype('string')

    def select_row(self, zaid):
        a,z = from_zaid(zaid)
        mask = (self.df['A'] == a) & (self.df['Z'] == z)
        return self.df.loc[ mask ]

    def element_name(self, zaid):
        row = self.select_row(zaid)
        return row["EL"].to_string().split()[1]

def to_zaid(a,z):
    return 1000*z + a

def from_zaid(zaid):
    z = int(zaid//1000)
    a = int(zaid%1000)
    return a,z

class NucHistories:
    def __init__(self, a=None, z=None, zaid=None):
        self.zaid = None
        self.a = None
        self.z = None
        if zaid:
            self.a, self.z = from_zaid(zaid)
        else:
            if a:
                self.a = a
            if z:
                self.z = z

        self.nu  = []
        self.nug = []
        self.ne  = []
        self.ge  = []
        self.num_hists    = 0
        self.num_neutrons = 0
        self.num_gammas   = 0

    def appendHists(self, nu, nug, ne, ge):
        self.nu.append(nu)
        self.nug.append(nug)
        self.ne.append(ne)
        self.ge.append(ge)
        self.num_hists    += nu.size
        self.num_neutrons += np.sum(nu)
        self.num_gammas   += np.sum(nug)

    def getNeutronEnergies(self, order = None ):
        if not order:
            energies = np.zeros((self.num_neutrons))
            i = 0
            for hist in self.ne:
                for energy in hist:
                    energies[i] = energy
                    i += 1
        else:
            energies = []
            for hist in self.ne:
                if len(hist) > order:
                    energies.append(hist[order])
            energies = np.array(energies)
        return energies

    def getGammaEnergies(self, order = None ):
        if not order:
            energies = np.zeros((self.num_gammas))
            i = 0
            for hist in self.ge:
                for energy in hist:
                    energies[i] = energy
                    i += 1
        else:
            energies = []
            for hist in self.ge:
                if len(hist) > order:
                    energies.append(hist[order])
            energies = np.array(energies)
        return energies

    def getPFNS(self):
        return np.histogram(getNeutronEnergies())

    def getSingleNeutronEnergy(self, order : int):
        return np.histogram(getNeutronEnergies(order))

def sortHistoriesByFragment( hist ):
    nuc_histories = {}

    # sort histories by isotope
    for a, z, nu, nug, ne, ge in zip(hist.getA(), hist.getZ(), hist.getNu(), hist.getNug(), hist.getNeutronEcm(), hist.getGammaEcm()):
        zaid = to_zaid(a,z)

        if zaid not in nuc_histories:
            nuc_histories[zaid] = NucHistories(zaid=zaid)

        nuc_histories[zaid].appendHists(nu, nug, ne, ge)

    return nuc_histories

def sortHistoriesByFragmentMass( hist , post_emission=False):
    nuc_histories = {}

    # sort histories by isotope
    for a, nu, nug, ne, ge in zip(hist.getA(), hist.getNu(), hist.getNug(), hist.getNeutronEcm(), hist.getGammaEcm()):
        if post_emission:
            a = a - nu
        if a not in nuc_histories:
            nuc_histories[a] = NucHistories(a=a)

        nuc_histories[a].appendHists(nu, nug, ne, ge)

    return nuc_histories

def sortHistoriesByFragmentCharge( hist ):
    nuc_histories = {}

    # sort histories by isotope
    for z, nu, nug, ne, ge in zip(hist.getZ(), hist.getNu(), hist.getNug(), hist.getNeutronEcm(),  hist.getGammaEcm()):
        if z not in nuc_histories:
            nuc_histories[z] = NucHistories(z=z)

        nuc_histories[z].appendHists(nu, nug, ne, ge)

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

def plot_element_multiplicity(Z, hists_by_frag, label=None, save=False, min_hists=0):
    # look at 1st neutron energy isotope by isotope
    A, Z_hists_by_A = selectElement(Z, hists_by_frag, min_hists=min_hists) # Z isotopes
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

def compare_fragment_mult(hists_by_nuc):
    elements = ["Zr", "Nb", "Mo", "Tc"]
    for i, Z in enumerate([40, 41, 42, 43]):
        plot_element_multiplicity(Z, hists_by_nuc, save=False, min_hists=1000, label=elements[i])

    plt.xlabel("A [u]")
    plt.ylabel(r"Single Fragment Multiplicity")
    plt.tight_layout()
    plt.legend()
    plt.savefig("AL_mult_sf.png")
    plt.close()

    elements = ["I", "Xe", "Ce" , "Ba"]
    for i, Z in enumerate([53, 54, 55, 56]):
        plot_element_multiplicity(Z, hists_by_nuc, save=False, min_hists=1000, label=elements[i])

    plt.xlabel("A [u]")
    plt.ylabel(r"Single Fragment Multiplicity")
    plt.tight_layout()
    plt.legend()
    plt.savefig("AH_mult_sf.png")
    plt.close()

    elements = ["Pd", "Ag", "Sn", "Sb"]
    for i, Z in enumerate([46, 47, 50, 51]):
        plot_element_multiplicity(Z, hists_by_nuc, save=False, min_hists=1000, label=elements[i])

    plt.xlabel("A [u]")
    plt.ylabel(r"Single Fragment Multiplicity")
    plt.tight_layout()
    plt.legend()
    plt.savefig("AMiddle_mult_sf.png")
    plt.close()

def writePFNSA(out_fname, A, hists_by_A, num_ebins=20):
    # write pfns as npy array
    # A A A ... mass
    # E E E ... energy
    # d d d ... counts [a. u.]
    # e e e ... err    [a. u.]
    data = np.zeros((4,num_ebins*len(A)))
    i = 0
    for mass, hist in zip(A, hists_by_A):
        energies = hist.getNeutronEnergies()
        counts, edges = np.histogram(energies, bins=num_ebins)
        centers  = (edges[1:] + edges[:1])*0.5
        data[0,i:i+num_ebins] = int(mass)
        data[1,i:i+num_ebins] = centers
        data[2,i:i+num_ebins] = counts

        i+= (num_ebins -1)

    np.save(out_fname, data)

def plotEn1ByNuc(hist, zaids, ame_table, zaid_normalize=None):
    hists_by_nuc = sortHistoriesByFragment(hist)
    edges   = np.arange(8.0, step=0.4)
    centers = (edges[:1] + edges[1:])*0.5

    if zaid_normalize:
        el_name = str(ame_table.element_name(zaid_normalize))
        an,zn = from_zaid(zaid_normalize)
        label_norm=f'$^{ {an} }${el_name}'

        hist_norm, _ = np.histogram(
                hists_by_nuc[zaid_normalize].getNeutronEnergies(order=0),
                bins=edges
                )

    for zaid in zaids:
        el_name = str(ame_table.element_name(zaid))
        a,z = from_zaid(zaid)
        label=f'$^{ {a} }${el_name}'
        hist, _  = np.histogram(
                hists_by_nuc[zaid].getNeutronEnergies(order=0),
                bins=edges
                )
        if zaid_normalize:
            plt.step(centers, hist / hist_norm, label=label)
        else:
            hist = hist/np.trapz(hist, x=centers)
            plt.step(centers, hist, label=zaid)

    y_lab_str = r'$p(E_1 | A,Z)$'
    if zaid_normalize:
        y_lab_str = r'$p(E_1 | A,Z)/ p(E_1 | {}, {})$ '.format(an,zn)
    plt.ylabel(y_lab_str)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    hists_by_frag_mass = sortHistoriesByFragmentMass(hist, post_emission=True)
    A, hists_by_A = zip(*hists_by_frag_mass.items())
    writePFNSA("cgmf_252cf_kddef_pfns_a.npy", A, hists_by_A)

    # read atomic mass and binding energies
    ame_table = AMETable()

    # 1st energy from Xenon isotopes
    plotEn1ByNuc(hist, [56143, 56144, 56145, 56146], ame_table, zaid_normalize=None)

