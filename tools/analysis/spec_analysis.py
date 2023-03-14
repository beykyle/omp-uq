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
    def __init__(self, arr : np.array):
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

class Spec:
    def __init__(self, spec, err, bins, xerr=None):
        self.spec = spec
        self.err = err
        self.bins = bins
        self.xerr=xerr

        assert(spec.shape == err.shape)

    def interp(self, bins):
        spec = np.interp(bins, self.bins, self.spec)
        err = np.interp(bins, self.bins, self.err)

        return Spec(spec, err, bins)

    def norm(self):
        return np.trapz(self.spec, x=self.bins)

    def normalize(self, norm=None):
        if not norm:
            norm = self.norm()
        return Spec(self.spec / norm, self.err / norm, self.bins, self.xerr)

    def sum_counts(self):
        return np.sum(self.spec)

    def dX(self):
        return self.bins[1:] - self.bins[:-1]

    def normalizePxdx(self):
        """
        Probability density conserving normalization. For spectra in xy form, simply normalizes
        to trapz integral. For spectra in xmin,xmax,y form (e.g. len(self.bins) == len(self.spec) + 1 ),
        converts to normalized xy form by interpolating to bin centers
        """
        if self.bins.shape != self.spec.shape:
            # interpolate to centers
            centers = 0.5*(self.bins[:-1] + self.bins[1:])
            sp = self.interp(centers)

            # get dX
            dx    = self.dX()
            total = sp.sum_counts()

            # normalize to bin widths
            sp.spec = sp.spec / dx / total
            sp.err  = sp.err / dx / total
        else:
            # simple trapz integration
            sp = self.normalize()

        return sp

    def mean(self):
        m0 = np.trapz(self.spec, x=self.bins)
        m1 = np.trapz(self.spec * self.bins, x=self.bins)
        e1 = np.trapz(self.spec * self.bins, x=self.bins)

        return m1/m0

    def variance(self):
        mean   = self.mean()
        var_un = np.trapz(self.spec * (self.bins - mean)**2, x=self.bins)
        return var_un

    def var_in_mean(self):
        sig2    = self.variance()
        num     = np.trapz( 1/(self.err**2 + sig2) , x=self.bins)
        denom   = np.trapz( 1/(self.err**2 + sig2) , x=self.bins)
        return num/denom**2

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
    norm = np.sum(spec, x=ebins)
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


def getMeanShiftErr(masses, pfns, pfns_ref, label, label_ref):
    ebins = np.arange(0.3, 6., step=0.2)

    # common interpolate and normalize
    specs  = [ (s.interp(ebins)).normalize() for s in  pfns.getSpecs(masses)     ]
    specsr = [ (s.interp(ebins)).normalize() for s in  pfns_ref.getSpecs(masses) ]


    mean   = np.array([ s.mean() for s in  specs  ])
    meanr  = np.array([ s.mean() for s in  specsr ])

    var    = np.array([ np.sqrt(s.variance()) for s in  specs  ])
    varr   = np.array([ np.sqrt(s.variance()) for s in  specsr ])

    dmean  = np.array([ np.sqrt(s.var_in_mean()) for s in  specs])
    dmeanr = np.array([ np.sqrt(s.var_in_mean()) for s in  specsr ])

    fig = plt.figure()

    #plt.errorbar(masses, mean , yerr=dmean, label=label)
    #plt.errorbar(masses, meanr, yerr=dmeanr, label=label_ref)

    p1 = plt.plot(masses, mean, label=label)
    #plt.fill_between( masses, mean - dmean/2, mean + dmean/2, color=p1[0].get_color(), alpha=0.4)
    p2 = plt.plot(masses, meanr, label=label_ref)
    #plt.fill_between( masses, meanr - dmeanr/2, meanr + dmeanr/2, color=p2[0].get_color(), alpha=0.4)

    plt.fill_between( masses, mean - var/2, mean + var/2, color=p1[0].get_color(), alpha=0.4)
    plt.fill_between( masses, meanr - varr/2, meanr + varr/2, color=p2[0].get_color(), alpha=0.4)

    plt.legend()
    plt.xlabel(r"$A$ [u]")
    plt.ylabel(r"$\langle E \rangle$ [MeV]")
    plt.tight_layout()

    return fig, mean, meanr, dmean, dmeanr

def integrateOverMaxwell(spec, kT):
    spec = spec.normalize()
    ebins = spec.bins
    maxwell = maxwellian(ebins, kT)
    nm = np.trapz(maxwell,x=ebins)
    maxwell = maxwell/nm
    norm = np.trapz(np.ones(maxwell.shape),x=ebins)
    metric     = np.trapz( spec.spec/maxwell ,x=ebins)/norm
    var_metric = np.trapz( spec.err**2/maxwell**2 , x=ebins) / norm**2

    return metric, np.sqrt(var_metric)


def getHardnessAboveMaxwell(masses, pfns, ebins):

    # common interpolate and normalize
    specs  = [ s.interp(ebins).normalize() for s in  pfns.getSpecs(masses)]
    hard = np.zeros((len(masses),2))
    for i in range(0,len(masses)):
        h,e = integrateOverMaxwell(specs[i], 1.42)
        hard[i,0] = h
        hard[i,1] = e

    return hard[:,0], hard[:,1]


def plotCompPFNS(A : int, datasets , labels, st=None):

    fig = plt.figure()

    ebins = [ d.bins for d in datasets]
    pfns = []
    err = []

    for i , d in enumerate(datasets):
        err_d  = d.err
        pfns_d = d.spec
        err.append(err_d)
        pfns.append(pfns_d)
        if st:
            if st[i] == "l":
                plt.errorbar(d.bins, pfns_d, yerr=err_d, label=labels[i])
            else:
                plt.errorbar(d.bins, pfns_d, yerr=err_d, label=labels[i], linestyle="None", marker=".")

    plt.yscale("log")
    plt.xlabel("E [MeV]")
    plt.legend()
    plt.ylabel(r"$P(E | A = {})$ [A.U.]".format(A))

    return fig, ebins, pfns, err

def plotSpecRatio( numerators, denominators, ebins, label_num, label_denom, labels, rel_diff=False):

    fig = plt.figure()

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
                %(label_num, label_num, label_denom))
    else:
        plt.ylabel(r"$ \frac{P_{%s}(E|A)}{P_{%s}(E|A)}$" %(label_num, label_denom))

    return fig

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
        plotCompPFNS(A, datasets, labels)
        plt.show()

    # heavy
    for A in [134, 137, 140 , 142, 145]:
        plotCompPFNS(A, datasets, labels)
        plt.show()
