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


def normalize_to_maxwell(x, y, dy, temp_MeV):
    m = maxwellian(x, temp_MeV)
    k = np.trapz(m, x)
    y = y * k / m
    dy = dy * k / m
    return y, dy


class Plotter:
    def __init__(self, exp_data_path: Path, energy_range=None):
        self.exp_data_path = exp_data_path
        self.energy_range = energy_range

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

    def plot_cgmf_vec(self, d, quantity, x, mc=False):
        vec_all = d.vector_qs[quantity]
        vec_stddev = d.vector_qs[quantity + "_stddev"]
        return self.plot_vec(vec_all, vec_stddev, x, d.label, mc)

    def plot_vec(self, vec_all, vec_stddev, x, label, mc=False, plot_type="fill"):
        vec_err = np.sqrt(np.var(vec_all, axis=0))
        mean_mc_err = np.mean(vec_stddev, axis=0)
        vec = np.mean(vec_all, axis=0)

        p1 = plt.plot(x, vec, label=label, zorder=100, linewidth=2)

        if plot_type == "overlapping":
            for i in range(vec_all.shape[0]):
                plt.fill_between(
                    x,
                    vec_all[i,...] + vec_stddev[i,...],
                    vec_all[i,...] - vec_stddev[i,...],
                    alpha=1./vec_all.shape[0],
                    zorder=100,
                    step="mid",
                    color=p1[0].get_color()
                )
        elif plot_type == "fill":
            plt.fill_between(x, vec + vec_err, vec - vec_err, alpha=0.6, zorder=100)
            if mc:
                plt.fill_between(
                    x,
                    vec + vec_err,
                    vec + vec_err + mean_mc_err,
                    alpha=0.3,
                    zorder=100,
                    color="k",
                )
                plt.fill_between(
                    x,
                    vec - vec_err,
                    vec - vec_err - mean_mc_err,
                    alpha=0.3,
                    zorder=100,
                    color="k",
                )
        return p1[0]

    def plot_spec(self, spec_all, spec_stddev_all, x, label, mc=False, plot_type="fill"):
        spec_err = np.sqrt(np.var(spec_all, axis=0))
        mean_mc_err = np.mean(spec_stddev_all, axis=0)
        spec = np.mean(spec_all, axis=0)

        p1 = plt.step(x, spec, label=label, zorder=100, linewidth=2, where="mid")

        if plot_type == "overlapping":
            for i in range(spec_all.shape[0]):
                plt.fill_between(
                    x,
                    spec_all[i,...] + spec_stddev_all[i,...],
                    spec_all[i,...] - spec_stddev_all[i,...],
                    alpha=1./spec_all.shape[0],
                    zorder=100,
                    step="mid",
                    color=p1[0].get_color()
                )
        elif plot_type == "fill":
            plt.fill_between(
                x, spec + spec_err, spec - spec_err, alpha=0.6, zorder=100, step="mid"
            )
            if mc:
                plt.fill_between(
                    x,
                    spec + spec_err,
                    spec + spec_err + mean_mc_err,
                    alpha=0.3,
                    zorder=100,
                    step="mid",
                    color="k",
                )
                plt.fill_between(
                    x,
                    spec - spec_err,
                    spec - spec_err - mean_mc_err,
                    alpha=0.3,
                    zorder=100,
                    step="mid",
                    color="k",
                )

        return p1[0]

    def plot_cgmf_spec(self, d, quantity, x, mc=False):
        spec_all = d.vector_qs[quantity]
        spec_stddev_all = d.vector_qs[quantity + "_stddev"]
        return self.plot_spec(spec_all, spec_stddev_all, x, d.label, mc)

    def plot_cgmf_spec_from_tensor(self, d, quantity, x, index, mc=False):
        spec_all = d.tensor_qs[quantity][:, index, :]
        spec_stddev_all = d.tensor_qs[quantity + "_stddev"][:, index, :]
        return self.plot_spec(spec_all, spec_stddev_all, x, d.label, mc)

    def plot_cgmf_vec_from_tensor(self, d, quantity, x, index, mc=False):
        vec_all = d.tensor_qs[quantity][:, index, :]
        vec_stddev_all = d.tensor_qs[quantity + "_stddev"][:, index, :]
        return self.plot_vec(vec_all, vec_stddev_all, x, d.label, mc)

    def pfns(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            # normalization
            x = d.bins["pfns"]
            pfns = d.vector_qs["pfns"]
            pfns_err = d.vector_qs["pfns_stddev"]

            pfns, pfns_err = normalize_to_maxwell(x, pfns, pfns_err, 1.32)

            plts_sim.append(self.plot_spec(pfns, pfns_err, x, d.label))

        def plt_exp_spec(s, l):
            x = s.bins
            pfns = s.spec
            pfns_err = s.err
            pfns, pfns_err= normalize_to_maxwell(x, pfns, pfns_err, 1.32)

            return plt.errorbar(
                x,
                pfns,
                yerr=pfns_err,
                xerr=s.xerr,
                alpha=0.7,
                label=l,
                linestyle="none",
            )

        pfns = read(self.exp_data_path, "pfns", self.energy_range)
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
                s = s.normalizePxdx()
                p = plt_exp_spec(s, l)
                plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=10, ncol=3)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3, loc="lower left")

        plt.xscale("log")
        plt.ylim([0.5, 2.0])
        plt.xlim([2e-2, 21])
        plt.xlabel(r"$E_{lab}$ [MeV]")
        plt.ylabel(r"$p(E) / M(kT = 1.32$ MeV$)$")

    def pfgs(self, cgmf_datasets=None):
        plts_sim = []

        for d in cgmf_datasets:
            x = d.bins["pfgs"]
            plts_sim.append(self.plot_cgmf_spec(d, "pfgs", x))

        # experimental data
        pfgs = read(self.exp_data_path, "pfgs", self.energy_range)

        specs = pfgs.get_specs()
        labels = [m["label"] for m in pfgs.meta]
        units = pfgs.units

        plts = []

        def plt_exp_spec(s, l):
            x = s.bins
            y = s.spec
            yerr = s.err
            return plt.errorbar(
                x, y, yerr=yerr, xerr=s.xerr, alpha=0.7, label=l, linestyle="none"
            )

        for s, l, u in zip(specs, labels, units):
            p = plt_exp_spec(s.normalizePxdx(), l)
            plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=9, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3, loc="lower left")

        plt.xscale("log")
        plt.ylim([0.0, 2.0])
        plt.xlim([1e-1, 10])
        plt.xlabel(r"$E_{lab}$ [MeV]")
        plt.ylabel(r"PFGS [MeV$^{-1}$]")

    def nugbarA(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_vec(d, "nugbarA", d.bins["nugbarA"]))

        # experiment
        nugbarA = read(self.exp_data_path, "nugbarA", self.energy_range)

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
        plt.legend(handles=plts_sim, fontsize=10, ncol=3, loc="lower left")

        plt.xlim([76, 174])
        plt.ylim([1, 8])
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$\bar{\nu}_\gamma$ [gammas]")

    def nubarTKE(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_spec(d, "nubarTKE", d.bins["nubarTKE"]))

        nubarTKE = read(self.exp_data_path, "nubarTKE", self.energy_range)

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

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3, loc="lower left")

        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$\bar{\nu}$ [neutrons]")

    def nugbarTKE(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_spec(d, "nugbarTKE", d.TKEcenters))

        nugbarTKE = read(self.exp_data_path, "nugbarTKE", self.energy_range)

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

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3, loc="lower right")

        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$\bar{\nu}_\gamma$ [gammas]")

    def nubarZ(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_vec(d, "nubarZ", d.bins["nubarZ"]))

        # experiment
        nubarZ = read(self.exp_data_path, "nubarZ", self.energy_range)

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
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc="lower right")

        plt.xlabel(r"$Z$ [protons]")
        plt.ylabel(r"$\bar{\nu}$ [neutrons]")

    def nubarA(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_vec(d, "nubarA", d.bins["nubarA"]))

        # experiment
        nubarA = read(self.exp_data_path, "nubarA", self.energy_range)

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

        lexp = plt.legend(handles=plts, fontsize=10, ncol=2, loc="upper left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc="lower right")

        plt.xlim([70, 180])
        plt.ylim([0, 6])
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$\bar{\nu}$ [neutrons]")

    def pnug(self, cgmf_datasets=None):
        plts_sim = []
        for d in cgmf_datasets:
            nu = d.bins["pnug"]
            plts_sim.append(self.plot_cgmf_vec(d, "pnug", nu))

        # exp
        pnug = read(self.exp_data_path, "pnug", self.energy_range)
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
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc="upper left")

        plt.xlabel(r"$\nu_\gamma$ [gammas]")
        plt.ylabel(r"$p(\nu_\gamma)$ [\%]")
        plt.tight_layout()

    def pnu(self, cgmf_datasets=None):
        # exp
        plts_sim = []
        for d in cgmf_datasets:
            nu = d.bins["pnu"]
            plts_sim.append(self.plot_cgmf_vec(d, "pnu", nu))

        pnu = read(self.exp_data_path, "pnu", self.energy_range)

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
        plt.ylabel(r"$p(\nu)$  ")
        plt.tight_layout()

    def nugbar(self, cgmf_datasets=None, endf=None):
        # simulation
        plts_sim = []
        max_n = 0
        num_plots = len(cgmf_datasets)
        alphas = np.linspace(0.9, 0.4, num=num_plots)
        orders = np.arange(0, num_plots * 100, 100)
        ma = 0
        num_bins = None
        for i, d in enumerate(cgmf_datasets):
            nugbar = d.scalar_qs["nugbar"]
            if num_bins == None:
                h, e = np.histogram(nugbar, density=True)
                num_bins = h.size
            else:
                h, e = np.histogram(nugbar, density=True, bins=num_bins)

            h = h / np.sum(h)
            de = e[1:] - e[:-1]
            p = plt.fill_between(
                0.5 * (e[:-1] + e[1:]),
                h,
                0.0,
                label=d.label,
                alpha=alphas[i],
                zorder=orders[i],
                step="pre",
            )
            plts_sim.append(p)
            if np.max(h) > ma:
                ma = np.max(h)

        # experiment
        nugbar = read(self.exp_data_path, "nugbar", self.energy_range)

        labels = [m["label"] for m in nugbar.meta]
        plts = []

        y = 0.8 * ma
        i = 0
        for d, l in zip(nugbar.data, labels):
            p = plt.errorbar(
                [d[0]], [y], xerr=[d[1] / 2], label=l, linestyle="none", marker="."
            )
            plts.append(p)
            y += 0.05 * ma
            i += 1

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3, loc="lower right")

        plt.xlim([7.5, 11])
        plt.ylim([0, y * 1.2])
        plt.xticks(np.arange(7.5, 10, 0.5))
        plt.grid(visible=True, axis="x", which="major")

        plt.grid(visible=True, axis="x", which="major")
        plt.xlabel(r"$\bar{\nu}_\gamma$ [gammas]")
        plt.ylabel(r"$p(\bar{\nu}_\gamma)$")

    def nubar(self, cgmf_datasets=None, endf=None):
        # simulation
        plts_sim = []
        max_n = 0
        num_plots = len(cgmf_datasets)
        alphas = np.linspace(0.9, 0.4, num=num_plots)
        orders = np.arange(0, num_plots * 100, 100)
        ma = 0
        num_bins = None
        for i, d in enumerate(cgmf_datasets):
            nubar = d.scalar_qs["nubar"]
            if num_bins == None:
                h, e = np.histogram(nubar, density=True)
                num_bins = h.size
            else:
                h, e = np.histogram(nubar, density=True, bins=num_bins)

            h = h / np.sum(h)
            de = e[1:] - e[:-1]
            p = plt.fill_between(
                0.5 * (e[:-1] + e[1:]),
                h,
                0.0,
                label=d.label,
                alpha=alphas[i],
                zorder=orders[i],
                step="mid",
            )
            plts_sim.append(p)
            if np.max(h) > ma:
                ma = np.max(h)

        if endf is not None:
            plt.plot([endf, endf], [0, ma], label="ENDF/B-VI.8", linestyle="--")

        # experiment
        nubar = read(self.exp_data_path, "nubar", self.energy_range)

        labels = [m["label"] for m in nubar.meta]
        plts = []

        y = 0.8 * ma
        i = 0
        for d, l in zip(nubar.data, labels):
            if d[1] < 0.1 and d[0] > 3.7 and d[0] < 3.85:
                p = plt.errorbar(
                    [d[0]], [y], xerr=[d[1] / 2], label=l, linestyle="none", marker="."
                )
                plts.append(p)
                y += 0.05 * ma
                i += 1

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc=2)

        plt.xlim([3.7, 3.88])
        plt.ylim([0, y * 1.2])
        plt.xticks(np.arange(3.7, 3.84, 0.02))
        plt.grid(visible=True, axis="x", which="major")
        plt.xlabel(r"$\bar{\nu}$ [neutrons]")
        plt.ylabel(r"$p(\bar{\nu})$")

    def pfns_A_moments(self, n: int):
        pfnsa = read(self.exp_data_path, "pfnsA", self.energy_range)

        labels = [m["label"] for m in pfnsa.meta]
        plts = []

        for d, l in zip(pfnsa.data, labels):
            data = PFNSA(np.vstack([d[4, :], d[0, :], d[2, :], d[3, :]]))
            specs = data.getSpecs(data.mass)
            means = [s.moment(n) / s.moment(0) for s in specs]
            plt.errorbar(data.mass, means, label=l)

        plt.legend(fontsize=10, ncol=1)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \langle{E}^{%d}\rangle $ [MeV]" % n)

    def pfnsA(self, a: int, pfnsa, cgmf_datasets=None):
        plts_sim = []
        for d in cgmf_datasets:
            index = np.nonzero(a == d.abins)[0][0]
            x = d.bins["pfnscomA"][1]
            pfns = d.tensor_qs["pfnscomA"][:, index, :]
            pfns_err = d.tensor_qs["pfnscomA_stddev"][:, index, :]
            pfns, pfns_err = normalize_to_maxwell(x, pfns, pfns_err, 1.32)
            plts_sim.append(self.plot_spec(pfns, pfns_err, x, d.label, mc=False))

        labels = [m["label"] for m in pfnsa.meta]

        plts = []
        for d, l in zip(pfnsa.data, labels):
            data = PFNSA(np.vstack([d[4, :], d[0, :], d[2, :], d[3, :]]))
            x, pfns, pfns_err = data.getPFNS(a)
            pfns, pfns_err = normalize_to_maxwell(x, pfns, pfns_err, 1.32)
            plts.append(plt.errorbar(x, pfns, pfns_err, label=l))

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="lower left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3)

        plt.xscale("log")
        plt.xlabel(r"$E_{cm}$ [MeV]")
        plt.ylabel(r"$p(E | A = {}) / M(E, kT = 1.32$ MeV$)$".format(a), fontsize=16)

    def nubarATKE(self, a: int, nubaratke, cgmf_datasets=None):
        plts_sim = []

        for d in cgmf_datasets:
            (abins, TKEbins) = d.bins["nuATKE"]
            index = np.nonzero(a == abins)[0][0]
            plts_sim.append(
                self.plot_cgmf_vec_from_tensor(
                    d, "nuATKE", TKEbins, index, mc=False
                )
            )

        labels = [m["label"] for m in nubaratke.meta]
        plts = []

        for d, l in zip(nubaratke.data, labels):
            amin = d[4,:]
            mask = a == amin
            tke = d[0, :][mask]
            dtke = d[1, :][mask]
            nu = d[2, :][mask]
            dnu = d[3, :][mask]
            plts.append(plt.errorbar(tke, nu, dnu, dtke, label=l, linestyle='none'))

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="lower left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3)

        plt.xlabel(r"$TKE$ [MeV]")
        plt.ylabel(r"$ \langle \nu | A = {} \rangle $ [neutrons]".format(a))


    def nubartATKE(self, a: int, nubaratke, cgmf_datasets=None):
        plts_sim = []

        for d in cgmf_datasets:
            (abins, TKEbins) = d.bins["nutATKE"]
            index = np.nonzero(a == abins)[0][0]
            plts_sim.append(
                self.plot_cgmf_vec_from_tensor(
                    d, "nutATKE", TKEbins, index, mc=False
                )
            )

        labels = [m["label"] for m in nubaratke.meta]
        plts = []

        for d, l in zip(nubaratke.data, labels):
            amin = d[4,:]
            mask = a == amin
            tke = d[0, :][mask]
            dtke = d[1, :][mask]
            nu = d[2, :][mask]
            dnu = d[3, :][mask]
            plts.append(plt.errorbar(
                tke, nu, dnu, dtke, label=l, linestyle='none'))

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="lower left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3)

        plt.xlabel(r"$TKE$ [MeV]")
        plt.ylabel(r"$ \langle \nu_t | A = {},{} \rangle $ [neutrons]"\
                   .format(a, 252-a)
                   )

    def encomATKE(self, a: int, encomatke, cgmf_datasets=None):
        plts_sim = []
        for d in cgmf_datasets:
            (abins, TKEbins) = d.bins["nutATKE"]
            index = np.nonzero(a == abins)[0][0]
            plts_sim.append(
                self.plot_cgmf_vec_from_tensor(
                    d,
                    "encomATKE",
                    TKEbins,
                    index,
                    mc = False
                )
            )

        labels = [m["label"] for m in encomatke.meta]
        plts = []

        for d, l in zip(encomatke.data, labels):
            mask = a == d[4, :]
            tke = d[0, :][mask]
            dtke = d[1, :][mask]
            encom = d[2, :][mask]
            dencom = d[3, :][mask]
            plts.append(plt.errorbar(tke, dtke, encom, dencom, label=l))

        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc="lower left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=3)

        plt.xlabel(r"$TKE$ [MeV]")
        plt.ylabel(r"$ \langle E | A = {} \rangle $ [neutrons]".format(a))

    def multratA(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_vec(d, "multratioA", d.bins["multratioA"], mc=False))

        mr = read(self.exp_data_path, "multiplicityRatioA", self.energy_range)

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
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc=2)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \frac{ \nu_\gamma }{ \nu_n }$")

    def encomA(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_vec(d, "encomA", d.bins["encomA"], mc=False))

        enbar = read(self.exp_data_path, "encomA", self.energy_range)

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

        plt.ylim([0, 3.0])
        lexp = plt.legend(handles=plts, fontsize=10, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc=2)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \langle {E}_n \rangle$ [MeV]")

    def encomTKE(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_spec(d, "encomTKE", d.bins["encomTKE"]))

        enbar = read(self.exp_data_path, "encomTKE", self.energy_range)

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
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc=2)
        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$ \langle{E}_n\rangle $ [MeV]")

    def egtbarTKE(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_spec(d, "egtbarTKE", d.bins["egtbarTKE"]))

        egtbar = read(self.exp_data_path, "egtbarTKE", self.energy_range)

        labels = [m["label"] for m in egtbar.meta]
        plts = []

        for d, l in zip(egtbar.data, labels):
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
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc=2)
        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$ \langle{E}^T_\gamma\rangle $ [MeV]")

    def egtbarA(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_vec(d, "egtbarA", d.bins["egtbarA"]))

        egtbar = read(self.exp_data_path, "egtbarA", self.energy_range)

        labels = [m["label"] for m in egtbar.meta]
        plts = []

        for d, l in zip(egtbar.data, labels):
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
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc=2)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \langle{E}^T_\gamma\rangle $ [MeV]")

    def egtbarnu(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_spec(d, "egtbarnu", d.bins["egtbarnu"]))

        egtbar = read(self.exp_data_path, "egtbarnu", self.energy_range)

        labels = [m["label"] for m in egtbar.meta]
        plts = []

        for d, l in zip(egtbar.data, labels):
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
        plt.legend(handles=plts_sim, fontsize=10, ncol=1, loc=2)
        plt.xlabel(r"$\nu$ [neutrons]")
        plt.ylabel(r"$ \langle{E}^T_\gamma\rangle $ [MeV]")
