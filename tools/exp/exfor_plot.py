from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import skew


data_path = (Path(__file__).parent.parent.parent / "data").absolute()

class PFNS_A:
    def __init__(self, npy_fname):
        arr = np.load(npy_fname)
        self.Amsk  = np.array(arr[0,:], dtype=int)
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

    def getSpecs(self, masses):
        specs = []
        ebins = self.getEbins()
        for mass in masses:
            spec, err  = self.getPFNS(mass)
            specs.append(Spec(spec, err, ebins))

        return specs

class Spec :
    def __init__(self, spec, err, bins):
        self.spec = spec
        self.err = err
        self.bins = bins
        self.centers = 0.5*(self.bins[1:] + self.bins[:1])

    def interp(self, bins):
        spec = np.interp(bins, self.bins, self.spec)
        err = np.interp(bins, self.bins, self.err)

        return Spec(spec, err, bins)

    def norm(self):
        return np.trapz(self.spec, x=self.bins)

    def normalize(self):
        norm = self.norm()
        return Spec(self.spec / norm, self.err / norm, self.bins)

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


    with open(str(fname) + ".e" , "w") as f:
        f.writelines(newlines)
        f.close()
    read_exfor_2npy( str(fname) +".e" , out)
    os.remove(str(fname) + ".e")



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


def hardness(spec, ref_spec, ebins, rel=False):
    spec = spec.interp(ebins)
    ref_spec = ref_spec.interp(ebins)

    spec = spec.normalize()
    ref_spec = ref_spec.normalize()

    mean = np.trapz(spec.spec * ebins, x=ebins)
    ref_mean = np.trapz(ref_spec.spec * ebins, x=ebins)

    if not rel:
        mean_ratio = mean / ref_mean
    else:
        mean_ratio = 100* (mean - ref_mean) / ref_mean

    return mean_ratio

def plotCompPFNS(A : int, datasets , labels, maxwell_norm=False):

    ebins = [ d.getEbins() for d in datasets]
    pfns = []
    err = []

    for i , d in enumerate(datasets):
        pfns_d, err_d = normalize_spec_err(
            ebins[i], *(d.getPFNS(A)), maxwell_norm=maxwell_norm, kT=1.42)
        err.append(err_d)
        pfns.append(pfns_d)

        plt.errorbar(ebins[i], pfns_d, yerr=err_d, label=labels[i])

    plt.xlim([0,7])
    plt.yscale("log")
    plt.xlabel("E [MeV]")
    plt.legend()
    if not maxwell_norm:
        plt.ylabel(r"$P(E | A = {})$ [A.U.]".format(A))
    else:
        plt.ylabel(r"$P(E | A = {}) / M(E, kT)$".format(A))
    plt.tight_layout()
    plt.show()

    return ebins, pfns, err

def plotSpecRatio( numerators, denominators, ebins, label_num, label_denom, labels, rel_diff=False):
    pfns = []

    for i , numer in enumerate(numerators):
        denom = denominators[i].interp(ebins)
        numer = numer.interp(ebins)

        # normalize on common grid
        denom = denom.normalize()
        numer = numer.normalize()

        # plot ratio
        plt.plot(ebins, numer.spec / denom.spec, label=labels[i])

    #plt.yscale("log")
    plt.xlabel("E [MeV]")
    plt.legend()
    if rel_diff:
        plt.ylabel(r"$ \frac{P_{%s}(E|A) - P_{%s}(E|A)}{P_{%s}(E|A)}$"
                %(label_num, label_num, labels_denom))
    else:
        plt.ylabel(r"$ \frac{P_{%s}(E|A)}{P_{%s}(E|A)}$" %(label_num, label_denom))
    plt.tight_layout()
    plt.show()

    return pfns

def plotSpecRatio3D(
        numerators, denominators, ebins, label_num, label_denom, y_vals, y_label, rel_diff=False):

    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i , numer in enumerate(numerators):
        denom = denominators[i].interp(ebins)
        numer = numer.interp(ebins)

        # normalize on common grid
        denom = denom.normalize()
        numer = numer.normalize()
        y = np.ones(ebins.shape) * y_vals[i]

        # plot ratio
        if rel_diff:
            plt.plot(ebins, y, 100*(numer.spec - denom.spec )/ denom.spec, color="k")
        else:
            plt.plot(ebins, y, numer.spec / denom.spec, color="k")

    # add a plane at z=1
    xx,yy = np.meshgrid(ebins, y_vals)
    z=np.ones((len(y_vals), ebins.size ))

    ax.plot_surface(xx,yy,z, alpha=0.2)

    ax.set_xlabel("E [MeV]", labelpad=14)
    ax.set_ylabel(y_label, labelpad=14)
    if rel_diff:
        ax.set_zlabel(r"$ \frac{P_{%s}(E|A) - P_{%s}(E|A)}{P_{%s}(E|A)}$"
                %(label_num, label_denom, label_denom) + "[%]", labelpad=18)
    else:
        ax.set_zlabel(r"$ \frac{P_{%s}(E|A)}{P_{%s}(E|A)}$" %(label_num, label_denom), labelpad=18)

    return fig, ax

def plotSpecRatioColor(
        numerators, denominators, ebins, label_num, label_denom, y_vals, y_label, rel_diff=False):

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xx,yy = np.meshgrid(ebins, y_vals)
    z=np.zeros((len(y_vals), ebins.size ))

    for i , numer in enumerate(numerators):
        denom = denominators[i].interp(ebins)
        numer = numer.interp(ebins)

        # normalize on common grid
        denom = denom.normalize()
        numer = numer.normalize()
        y = np.ones(ebins.shape) * y_vals[i]

        # plot ratio
        if rel_diff:
            z[i,:] = 100*(numer.spec - denom.spec )/ denom.spec
        else:
            z[i,:] = (numer.spec / denom.spec)


    surf = ax.plot_surface(xx, yy, z, alpha=1., cmap=matplotlib.colormaps['viridis'])

    #ax.contourf(xx, yy, z, zdir='x', offset=-40, cmap=cm.coolwarm)
    #ax.contourf(xx, yy, z, zdir='y', offset=40, cmap=cm.coolwarm)

    ax.set_xlabel("E [MeV]", labelpad=14)
    ax.set_ylabel(y_label, labelpad=20)
    if rel_diff:
        ax.set_zlabel(r"$ \frac{P_{%s}(E|A) - P_{%s}(E|A)}{P_{%s}(E|A)}$"
                %(label_num, label_denom, label_denom) + "[%]", labelpad=28)
    else:
        ax.set_zlabel(r"$ \frac{P_{%s}(E|A)}{P_{%s}(E|A)}$" %(label_num, label_denom), labelpad=24)

    return xx, yy, z, ax

def compareExforExample():
    read_exfor_alt_2npy(data_path / "exfor/CMspectra_vs_mass_U235.txt", "235_U_PFNS_A.npy")
    read_exfor_alt_2npy(data_path / "exfor/CMspectra_vs_mass_Pu239.txt", "239_Pu_PFNS_A.npy")
    read_exfor_2npy(data_path /  "exfor/CMspectra_vs_mass_Cf252.txt", "252_Cf_PFNS_A.npy")

    datasets =  [
            PFNS_A("252_Cf_PFNS_A.npy") ,
            PFNS_A("235_U_PFNS_A.npy")  ,
            PFNS_A("239_Pu_PFNS_A.npy") ,
    ]

    labels = [
            r"$^{252}$Cf (sf)",
            r"$^{235}$U (nth,f)",
            r"$^{239}$Pu (nth,f)",
            ]

    # light
    for A in [105, 107, 110 , 113]:
        plotCompPFNS(A, datasets, labels, maxwell_norm=False)

    # heavy
    for A in [134, 137, 140 , 142, 145]:
        plotCompPFNS(A, datasets, labels, maxwell_norm=False)
