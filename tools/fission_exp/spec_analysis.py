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

        return sp

    def moment(self, n: int):
        return np.trapz(self.spec * (self.bins) ** n, x=self.bins)

    def mean(self):
        m0 = self.moment(0)
        m1 = self.moment(1)

        return m1 / m0

    def variance(self):
        mean = self.mean()
        var_un = np.trapz(self.spec * (self.bins - mean) ** 2, x=self.bins)
        return var_un


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
        return self.E[mask], self.cnts[mask], self.sterr[mask]

    def getSpecs(self, masses):
        specs = []
        for mass in masses:
            ebins, spec, err = self.getPFNS(mass)
            specs.append(Spec(spec, err, ebins))

        return specs


def maxwellian(ebins: np.array, Eavg: float):
    return (
        2 * np.sqrt(ebins / np.pi) * (1 / Eavg) ** (3.0 / 2.0) * np.exp(-ebins / Eavg)
    )
