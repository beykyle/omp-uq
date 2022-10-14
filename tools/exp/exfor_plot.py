from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt

def maxwellian(ebins : np.array, Eavg : float):
    return 2*np.sqrt(ebins / np.pi) * (1 / Eavg)**(3./2.) * np.exp(- ebins / Eavg )


def read_exfor_2npy(fname, out):
    df = pd.read_csv(fname, delim_whitespace=True)

    mass = np.array(pd.to_numeric(df['MASS']).to_numpy(), dtype=int)
    data = np.zeros((4,mass.shape[0]))
    data[0,:] = mass
    data[1,:] = np.array(pd.to_numeric(df['E']).to_numpy(), dtype=float)
    data[2,:] = np.array(pd.to_numeric(df['DATA']).to_numpy(), dtype=float)
    data[3,:] = np.array(pd.to_numeric(df['ERR-S']).to_numpy(), dtype=float)
    np.save(out, data, allow_pickle=True)


def read_exfor_alt_2npy(fname, out):
    with open(fname, "r") as f:
        lines = f.readlines()
        newlines = [lines[4]]
        for i, l in enumerate(lines[5:]):
            if l.__contains__("A_pre"):
                A = l.split("=")[1].strip()
            elif l[0].isnumeric():
                newlines.append(A + " " + l )


    with open(fname + ".e" , "w") as f:
        f.writelines(newlines)
        f.close()
    read_exfor_2npy(fname +".e" , out)
    os.remove(fname + ".e")


class PFNS_A:
    def __init__(self, npy_fname):
        arr = np.load(npy_fname)
        self.Amsk  = arr[0,:]
        self.E     = arr[1,:]
        self.cnts  = arr[2,:]
        self.sterr = arr[3,:]
        self.mass  = np.unique(self.Amsk)

    def getEbins(self):
        a = self.Amsk[0]
        return self.E[np.where(self.Amsk == a)]

    def getPFNS(self, A : int):
        mask = np.where(self.Amsk == A)
        return self.cnts[mask], self.sterr[mask]

def normalize_spec_err(ebins : np.array, spec : np.array, err : np.array, maxwell_norm=False, kT=None):
    norm = np.trapz(spec, x=ebins)
    mean_e = np.trapz(spec/norm * ebins, x=ebins)
    if not kT:
        kT = mean_e

    if not maxwell_norm:
        return spec / norm, err / norm
    else:
        spec = spec/norm
        err  = err/norm
        maxw_spec = maxwellian(ebins, mean_e)
        maxw_spec = maxw_spec / np.trapz(maxw_spec, x=ebins)
        return spec / maxw_spec , err / maxw_spec


def plot_comp(A : int, maxwell_norm=False):
    cf252_fragments_pfns = PFNS_A("252_Cf_PFNS_A.npy")
    u235_fragments_pfns = PFNS_A("235_U_PFNS_A.npy")
    pu239_fragments_pfns = PFNS_A("239_Pu_PFNS_A.npy")

    ebins_cf = cf252_fragments_pfns.getEbins()
    ebins_u  = pu239_fragments_pfns.getEbins()
    ebins_pu = u235_fragments_pfns.getEbins()

    pfns_cf, err_cf = normalize_spec_err(
            ebins_cf, *(cf252_fragments_pfns.getPFNS(A)), maxwell_norm=maxwell_norm, kT=1.42)
    pfns_u , err_u  = normalize_spec_err(
            ebins_u, *(u235_fragments_pfns.getPFNS(A)), maxwell_norm=maxwell_norm, kT=1.42)
    pfns_pu, err_pu = normalize_spec_err(
            ebins_pu, *(pu239_fragments_pfns.getPFNS(A)), maxwell_norm=maxwell_norm, kT=1.42)

    plt.errorbar(ebins_cf, pfns_cf, yerr=err_cf, label=r"$^{252}$Cf (sf)")
    plt.errorbar(ebins_u, pfns_u, yerr=err_u, label=r"$^{235}$U (nth,f)")
    #plt.errorbar(ebins_pu, pfns_pu, yerr=err_pu, label=r"$^{239}$Pu (nth,f)")

    plt.xlim([0,7])
    #plt.ylim([0,2.25])
    plt.yscale("log")
    plt.xlabel("E [MeV]")
    plt.legend()
    if not maxwell_norm:
        plt.ylabel(r"$P(E | A = {})$ [A.U.]".format(A))
    else:
        plt.ylabel(r"$P(E | A = {}) / M(E, kT)$".format(A))
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

    read_exfor_alt_2npy("CMspectra_vs_mass_U235.txt", "235_U_PFNS_A.npy")
    read_exfor_alt_2npy("CMspectra_vs_mass_Pu239.txt", "239_Pu_PFNS_A.npy")
    read_exfor_2npy("CMspectra_vs_mass_Cf252.txt", "252_Cf_PFNS_A.npy")

    for A in range(87,150):
        plot_comp(A, maxwell_norm=True)
