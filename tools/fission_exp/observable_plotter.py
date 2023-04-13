import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.labelsize"] = 18.0
matplotlib.rcParams["xtick.labelsize"] = 18.0
matplotlib.rcParams["ytick.labelsize"] = 18.0
matplotlib.rcParams["lines.linewidth"] = 2.0
matplotlib.rcParams["xtick.major.pad"] = "10"
matplotlib.rcParams["ytick.major.pad"] = "10"
matplotlib.rcParams["image.cmap"] = "BuPu"

from fission_exp import Quantity, maxwellian, PFNSA, read


class Plotter:
    def __init__(self, exp_data_path: Path):
        self.exp_data_path = exp_data_path

    def get_fact_moments(self, moments: np.array, nu: np.array, pnu: np.array):
        assert nu.shape == pnu.shape
        assert np.sum(pnu) - 1.0 < 1e-10
        nu = np.array(nu, dtype=int)
        fact_moments = np.zeros(moments.shape)
        fall_fact = np.zeros(nu.shape)
        for i in range(0, len(moments)):
            for j in range(0, len(nu)):
                if moments[i] <= nu[j]:
                    fall_fact[j] = np.math.factorial(int(nu[j])) / np.math.factorial(
                        int(nu[j]) - int(moments[i])
                    )
            fact_moments[i] = np.dot(pnu, fall_fact)

        return fact_moments

    def normalize(self, arr: np.array):
        return arr / np.sum(arr)

    def pfns(self, cgmf_datasets=[]):
        # exp
        plts_sim = []
        for d in cgmf_datasets:

            pfns_all = d.vector_qs["pfns"]
            pfns_stddev_all = d.vector_qs["pfns_stddev"]

            x = d.ecenters
            m = maxwellian(x, 1.32)
            k = np.trapz(m, x)

            pfns_err = np.sqrt(np.var(pfns_all, axis=0)) * k / m
            mean_mc_err = np.mean(pfns_stddev_all, axis=0) * k / m
            pfns = np.mean(pfns_all, axis=0) * k / m

            p1 = plt.step(x, pfns, label=d.label, zorder=100, linewidth=2, where="mid")
            plts_sim.append(p1[0])
            plt.fill_between(
                x, pfns, pfns - pfns_err, pfns + pfns_err, alpha=0.5, zorder=100, step="mid"
            )
            plt.fill_between(
                x,
                pfns,
                pfns - pfns_err - mean_mc_err,
                pfns + pfns_err + mean_mc_err,
                alpha=0.5,
                zorder=100,
                step="mid",
                color="k"
            )

        def plt_spec(s, l):
            x = s.bins
            m = maxwellian(x, 1.32)
            k = np.trapz(m, x)
            y = s.spec * k / m
            yerr = s.err * k / m
            return plt.errorbar(
                x, y, yerr=yerr, xerr=s.xerr, alpha=0.7, label=l, linestyle="none"
            )

        pfns = read(self.exp_data_path, "pfns")
        specs = pfns[0].get_specs()
        labels = [m["label"] for m in pfns[0].meta]
        units = pfns[0].units

        la = {
            "Nefedov et al., 1983",
            "Maerten et al., 1990",
            "Knitter et al., 1973",
            "Lajtai et al., 1990",
            "Boytsov et al., 1983",
            "H.R.Bowman et al., 1958",
            "N.V.Kornilov2015",
            "H.Conde et al., 1965",
            "H.Werle et al., 1972",
            "A.Chalupka et al., 1990",
            "A.Chalupka et al., 1990",
            "L.Jeki et al., 1971",
            "L.Jeki et al., 1971",
            "Li Anli et al., 1982",
            "H.Maerten et al., 1990",
            "P.R.P.Coelho et al., 1989",
            "Z.A.Alexandrova et al., 1974",
            "M.V.Blinov et al., 1973",
            "M.V.Blinov et al., 1973",
            "M.V.Blinov et al., 1973",
            "B.I.Starostov et al., 1979",
            "B.I.Starostov et al., 1983",
            "P.P.Dyachenko et al., 1989",
        }

        plts = []

        for s, l, u in zip(specs, labels, units):
            if l in la:
                p = plt_spec(s.normalizePxdx(), l)
                plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=10, ncol=3)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3, loc="lower left")

        plt.xscale("log")
        plt.ylim([0.5, 2.0])
        plt.xlim([2e-2, 21])
        plt.xlabel(r"$E_{lab}$ [MeV]")
        plt.ylabel(r"PFNS ratio to Maxwellian ($kT = 1.32$ MeV)")

    def pfgs(self):
        def plt_spec(s, l):
            x = s.bins
            y = s.spec
            yerr = s.err
            return plt.errorbar(
                x, y, yerr=yerr, xerr=s.xerr, alpha=0.7, label=l, linestyle="none"
            )

        pfgs = read(self.exp_data_path, "pfgs")

        specs = pfgs.get_specs()
        labels = [m["label"] for m in pfgs.meta]
        units = pfgs.units

        plts = []

        for s, l, u in zip(specs, labels, units):
            p = plt_spec(s.normalizePxdx(), l)
            plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=9, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)

        plt.xscale("log")
        plt.ylim([0.0, 2.0])
        plt.xlim([1e-1, 10])
        plt.xlabel(r"$E_{lab}$ [MeV]")
        plt.ylabel(r"PFGS [MeV$^{-1}$]")

    def nugbarA(self, cgmf_datasets=[]):
        # sim

        # experiment
        nugbarA = read(self.exp_data_path, "nugbarA")

        labels = [m["label"] for m in nugbarA.meta]
        plts = []

        for d, l in zip(nugbarA.data, labels):
            p = plt.errorbar(
                d[0, :],
                d[2, :],
                d[3, :],
                d[1, :],
                label=l,
                linestyle="none",
                marker=".",
                zorder=0,
            )
            plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="upper left")
        plt.gca().add_artist(lexp)

        plt.xlim([70, 180])
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$\bar{\nu_\gamma}$ [gammas]")

    def nugbarTKE(self):
        nugbarTKE = read(self.exp_data_path, "nugbarTKE")

        plts = []
        labels = [m["label"] for m in nugbarTKE.meta]

        for d, l in zip(nugbarTKE.data, labels):
            plts.append(
                plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                )
            )

        plt.legend(plts, labels, fontsize=10)
        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$\bar{\nu_\gamma}$ [gammas]")

    def nubarZ(self, cgmf_datasets=[]):
        # sim

        # experiment
        nubarZ = read(self.exp_data_path, "nubarZ")

        labels = [m["label"] for m in nubarZ.meta]
        plts = []

        for d, l in zip(nubarZ.data, labels):
            p = plt.errorbar(
                d[0, :],
                d[2, :],
                d[3, :],
                d[1, :],
                label=l,
                linestyle="none",
                marker=".",
                zorder=0,
            )
            plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="upper left")
        plt.gca().add_artist(lexp)
        # plt.legend( handles=plts_sim, fontsize=10 , ncol=1, loc='lower right')

        plt.xlabel(r"$Z$ [protons]")
        plt.ylabel(r"$\bar{\nu}$ [neutrons]")

    def nubarA(self, cgmf_datasets=[]):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            a = d.abins
            nubarA = d.vector_qs["nubarA"]
            nubarA_mean = np.mean(nubarA, axis=0)
            nubarA_stdev = np.sqrt(np.var(nubarA, axis=0))
            plts_sim.append(
                plt.plot(a, nubarA_mean, label=d.label, zorder=1)
            )
            plt.fill_between(a, nubarA_mean + nubarA_stdev, nubarA_mean - nubarA_stdev, alpha=0.5, zorder=1)

        # experiment
        nubarA = read(self.exp_data_path, "nubarA")

        labels = [m["label"] for m in nubarA.meta]
        plts = []

        for d, l in zip(nubarA.data, labels):
            if l != "B.G.Basova et al., 1979":
                p = plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                    zorder=0,
                )
                plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="upper left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc="lower right")

        plt.xlim([70, 180])
        plt.ylim([0, 5])
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$\bar{\nu}$ [neutrons]")

    def nubarTKE(self):
        nubarTKE = read(self.exp_data_path, "nubarTKE")

        plts = []
        labels = [m["label"] for m in nubarTKE.meta]

        for d, l in zip(nubarTKE.data, labels):
            plts.append(
                plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                )
            )

        plt.legend(plts, labels, fontsize=10)
        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$\bar{\nu}$ [neutrons]")

    def pnug(self, cgmf_datasets=[]):
        # exp
        pnug = read(self.exp_data_path, "pnug")
        labels = [m["label"] for m in pnug.meta]
        plts = []

        for d, l in zip(pnug.data, labels):
            y = d[2, :]
            yerr = d[3, :]
            n = np.sum(y)
            y /= n
            yerr /= n

            plts.append(
                plt.errorbar(
                    d[0, :], y, yerr, d[1, :], label=l, linestyle="none", marker="."
                )
            )

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)

        plt.xlabel(r"$\nu_\gamma$ [gammas]")
        plt.ylabel(r"$p(\nu_\gamma)$ ")
        plt.tight_layout()

    def pnu(self, cgmf_datasets=[]):
        # exp
        plts_sim = []
        for d in cgmf_datasets:
            nu = np.arange(0, d.num_nu_bins)
            pnu_mean = np.mean(d.pnu, axis=0)
            pnu_stdev = np.sqrt(np.var(d.pnu, axis=0))
            plts_sim.append(
                plt.errorbar(nu, pnu_mean, yerr=pnu_stdev, label=d.label, zorder=1)
            )
            plt.fill_between(nu, pnu_mean - pnu_stdev, pnu_mean + pnu_stdev, alpha=0.4)

        pnu = read(self.exp_data_path, "pnu")

        labels = [m["label"] for m in pnu.meta]
        plts = []

        for d, l in zip(pnu.data, labels):
            y = d[2, :]
            yerr = d[3, :]
            n = np.sum(y)
            y /= n
            yerr /= n

            plts.append(
                plt.errorbar(
                    d[0, :], y, yerr, d[1, :], label=l, linestyle="none", marker="."
                )
            )

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc="upper left")

        plt.xlabel(r"$\nu$ [neutrons]")
        plt.ylabel(r"$p(\nu)$ ")
        plt.tight_layout()

    def nugbar(self, cgmf_datasets=[], endf=None):
        # simulation
        ma = 0.3

        # experiment
        nugbar = read(self.exp_data_path, "nugbar")

        labels = [m["label"] for m in nugbar.meta]
        plts = []

        y = 0.8 * ma * 100
        i = 0
        for d, l in zip(nugbar.data, labels):
            p = plt.errorbar(
                [d[0]], [y], xerr=[d[1] / 2], label=l, linestyle="none", marker="."
            )
            plts.append(p)
            y += 5 * ma
            i += 1

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)

        plt.grid(visible=True, axis="x", which="major")
        plt.xlabel(r"$\bar{\nu_\gamma}$ [gammas]")
        plt.ylabel(r"$p(\bar{\nu_\gamma})$ [%]")

    def nubar(self, cgmf_datasets=[], endf=None):
        # simulation
        plts_sim = []
        max_n = 0
        num_plots = len(cgmf_datasets)
        alphas = np.linspace(0.9, 0.4, num=num_plots)
        orders = np.arange(0, num_plots * 100, 100)
        ma = 0
        for i, d in enumerate(cgmf_datasets):
            nubar = d.scalar_qs["nubar"]
            h, e = np.histogram(nubar, density=True)
            de = e[1:] - e[:-1]
            h = h / np.sum(h)
            p = plt.fill_between(
                0.5 * (e[:-1] + e[1:]),
                0,
                100 * h,
                label=d.label,
                alpha=alphas[i],
                zorder=orders[i],
                step="pre",
            )
            plts_sim.append(p)
            if np.max(h) > ma:
                ma = np.max(h)

        if endf is not None:
            plt.plot([endf, endf], [0, ma], label="ENDF/B-VI.8", linestyle="--")

        # experiment
        nubar = read(self.exp_data_path, "nubar")

        labels = [m["label"] for m in nubar.meta]
        plts = []

        y = 0.8 * ma * 100
        i = 0
        for d, l in zip(nubar.data, labels):
            if d[1] < 0.1 and d[0] > 3.7 and d[0] < 3.85:
                p = plt.errorbar(
                    [d[0]], [y], xerr=[d[1] / 2], label=l, linestyle="none", marker="."
                )
                plts.append(p)
                y += 5 * ma
                i += 1

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc=2)

        plt.xlim([3.7, 3.88])
        plt.ylim([0, y * 1.2])
        plt.xticks(np.arange(3.7, 3.84, 0.02))
        plt.grid(visible=True, axis="x", which="major")
        plt.xlabel(r"$\bar{\nu}$ [neutrons]")
        plt.ylabel(r"$p(\bar{\nu})$ [%]")

    def pfns_A_moments(self, n: int):
        pfnsa = read(self.exp_data_path, "pfnsA")

        labels = [m["label"] for m in pfnsa.meta]
        plts = []

        for d, l in zip(pfnsa.data, labels):
            data = PFNSA(np.vstack([d[4, :], d[0, :], d[2, :], d[3, :]]))
            specs = data.getSpecs(data.mass)
            means = [s.moment(n) / s.moment(0) for s in specs]
            plt.errorbar(data.mass, means, label=l)

        plt.legend(fontsize=10, ncol=1)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \bar{E^{%d}} $ [MeV]" % n)

    def multratA(self):
        mr = read(self.exp_data_path, "multiplicityRatioA")

        labels = [m["label"] for m in mr.meta]
        plts = []

        for d, l in zip(mr.data, labels):
            plts.append(
                plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                )
            )

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        # plt.legend( handles=plts_sim, fontsize=10 , ncol=1, loc=2)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \frac{ \nu_\gamma }{ \nu_n }$")

    def enbarA(self):
        enbar = read(self.exp_data_path, "enbarA")

        labels = [m["label"] for m in enbar.meta]
        plts = []

        for d, l in zip(enbar.data, labels):
            plts.append(
                plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                )
            )

        plt.ylim([1, 3.0])
        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        # plt.legend( handles=plts_sim, fontsize=10 , ncol=1, loc=2)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \bar{E}_n $ [MeV]")

    def enbarTKE(self):
        enbar = read(self.exp_data_path, "enbarTKE")

        labels = [m["label"] for m in enbar.meta]
        plts = []

        for d, l in zip(enbar.data, labels):
            plts.append(
                plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                )
            )

        plt.ylim([1, 2.0])
        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        # plt.legend( handles=plts_sim, fontsize=10 , ncol=1, loc=2)
        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$ \bar{E}_n $ [MeV]")

    def egbarTKE(self, cgmf_datasets=[]):
        egbar = read(self.exp_data_path, "egbarTKE")

        labels = [m["label"] for m in egbar.meta]
        plts = []

        for d, l in zip(egbar.data, labels):
            plts.append(
                plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                )
            )

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        # plt.legend( handles=plts_sim, fontsize=10 , ncol=1, loc=2)
        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$ \bar{E_\gamma} $ [MeV]")

    def egbarA(self, cgmf_datasets=[]):
        egbar = read(self.exp_data_path, "egbarA")

        labels = [m["label"] for m in egbar.meta]
        plts = []

        for d, l in zip(egbar.data, labels):
            plts.append(
                plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                )
            )

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        # plt.legend( handles=plts_sim, fontsize=10 , ncol=1, loc=2)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \bar{E_\gamma} $ [MeV]")

    def egbarnubar(self, cgmf_datasets=[]):
        egbar = read(self.exp_data_path, "egbarnubar")

        labels = [m["label"] for m in egbar.meta]
        plts = []

        for d, l in zip(egbar.data, labels):
            plts.append(
                plt.errorbar(
                    d[0, :],
                    d[2, :],
                    d[3, :],
                    d[1, :],
                    label=l,
                    linestyle="none",
                    marker=".",
                )
            )

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        # plt.legend( handles=plts_sim, fontsize=10 , ncol=1, loc=2)
        plt.xlabel(r"$\nu$ [neutrons]")
        plt.ylabel(r"$ \bar{E_\gamma} $ [MeV]")
