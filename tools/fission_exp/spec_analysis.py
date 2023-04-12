import numpy as np
import pandas as pd
import os


class Spec:
    def __init__(self, spec, err, bins, xerr=None):
        self.spec = spec
        self.err = err
        self.bins = bins
        self.xerr = xerr

        assert spec.shape == err.shape

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
            centers = 0.5 * (self.bins[:-1] + self.bins[1:])
            sp = self.interp(centers)

            # get dX
            dx = self.dX()
            total = sp.sum_counts()

            # normalize to bin widths
            sp.spec = sp.spec / dx / total
            sp.err = sp.err / dx / total
        else:
            # simple trapz integration
            sp = self.normalize()


class PFNSA:
    def __init__(self, arr: np.array):
        self.Amsk = np.array(arr[0, :], dtype=int)
        self.E = arr[1, :]
        self.cnts = arr[2, :]
        self.sterr = arr[3, :]
        self.mass = np.unique(self.Amsk)

    def getEbins(self):
        a = self.Amsk[0]
        return self.E[np.where(self.Amsk == a)]

    def getPFNS(self, A: int):
        mask = np.where(self.Amsk == A)
        return self.cnts[mask], self.sterr[mask]

    def getSpecs(self, masses):
        specs = []
        ebins = self.getEbins()
        for mass in masses:
            spec, err = self.getPFNS(mass)
            specs.append(Spec(spec, err, ebins))

        return specs


def maxwellian(ebins: np.array, Eavg: float):
    return (
        2 * np.sqrt(ebins / np.pi) * (1 / Eavg) ** (3.0 / 2.0) * np.exp(-ebins / Eavg)
    )


def read_exfor_2npy(fname, out):
    df = pd.read_csv(fname, delim_whitespace=True)

    mass = np.array(pd.to_numeric(df["MASS"]).to_numpy(), dtype=int)
    data = np.zeros((4, mass.shape[0]))
    data[0, :] = mass
    data[1, :] = np.array(pd.to_numeric(df["E"]).to_numpy(), dtype=float)
    data[2, :] = np.array(pd.to_numeric(df["DATA"]).to_numpy(), dtype=float)
    data[3, :] = np.array(pd.to_numeric(df["ERR-S"]).to_numpy(), dtype=float)

    np.save(out, data, allow_pickle=True)


def read_exfor_alt_2npy(fname, out):
    with open(fname, "r") as f:
        lines = f.readlines()
        newlines = [lines[4]]
        for i, l in enumerate(lines[5:]):
            if l.__contains__("A_pre"):
                A = l.split("=")[1].strip()
            elif l[0].isnumeric():
                newlines.append(A + " " + l)

    with open(str(fname) + ".e", "w") as f:
        f.writelines(newlines)
        f.close()
    read_exfor_2npy(str(fname) + ".e", out)
    os.remove(str(fname) + ".e")
