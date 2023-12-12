import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib.ticker import StrMethodFormatter
from matplotlib.lines import Line2D


import statsmodels.stats.api as sms

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.labelsize"] = 18.0
matplotlib.rcParams["xtick.labelsize"] = 18.0
matplotlib.rcParams["ytick.labelsize"] = 18.0
matplotlib.rcParams["lines.linewidth"] = 2.0
matplotlib.rcParams["xtick.major.pad"] = "10"
matplotlib.rcParams["ytick.major.pad"] = "10"
matplotlib.rcParams["image.cmap"] = "BuPu"

from fission_exp import Quantity, maxwellian, PFNSA, read

def normalize_to_maxwell(x, y, dy, temp_MeV, ratio=None):
    m = maxwellian(x, temp_MeV)
    if x.shape == y.shape:
        k = np.trapz(m, x) / np.trapz(y, x, axis=0)
    else:
        k = np.trapz(m, x) / np.trapz(y, x, axis=1)

    if ratio:
        if x.shape == y.shape:
            y = y * k / m
            dy = dy * k / m
        else:
            y = y * k[:, np.newaxis] / m
            dy = dy * k[:, np.newaxis] / m
    else:
        if x.shape == y.shape:
            y = y * k
            dy = dy * k
        else:
            y = y * k[:, np.newaxis]
            dy = dy * k[:, np.newaxis]

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

    def plot_cgmf_vec(self, d, quantity, x, mc=True):
        vec_all = d.vector_qs[quantity]
        vec_stddev = d.vector_qs[quantity + "_stddev"]
        return self.plot_vec(vec_all, vec_stddev, x, d.label, mc)

    def plot_vec(self, vec_all, vec_stddev, x, label, mc=True, plot_type="fill"):
        vec_err = np.sqrt(np.var(vec_all, axis=0))
        mean_mc_err = np.mean(vec_stddev, axis=0)
        vec = np.mean(vec_all, axis=0)

        p1 = plt.plot(x, vec, label=label, zorder=100, linewidth=2)

        if plot_type == "overlapping":
            for i in range(vec_all.shape[0]):
                plt.fill_between(
                    x,
                    vec_all[i, ...] + vec_stddev[i, ...],
                    vec_all[i, ...] - vec_stddev[i, ...],
                    alpha=1.0 / vec_all.shape[0],
                    zorder=100,
                    step="mid",
                    color=p1[0].get_color(),
                )
        elif plot_type == "fill":
            plt.fill_between(x, vec + vec_err, vec - vec_err, alpha=0.6, zorder=100)
            if mc:
                plt.fill_between(
                    x,
                    vec + vec_err,
                    vec + vec_err + mean_mc_err,
                    alpha=0.2,
                    zorder=100,
                    color="k",
                )
                plt.fill_between(
                    x,
                    vec - vec_err,
                    vec - vec_err - mean_mc_err,
                    alpha=0.2,
                    zorder=100,
                    color="k",
                )
        return p1[0]

    def plot_spec(self, spec_all, spec_stddev_all, x, label, mc=True, plot_type="fill"):
        spec_err = np.sqrt(np.var(spec_all, axis=0))
        mean_mc_err = np.mean(spec_stddev_all, axis=0)
        spec = np.mean(spec_all, axis=0)

        # p1 = plt.step(x, spec, label=label, zorder=100, linewidth=2, where="mid")
        p1 = plt.plot(x, spec, label=label, zorder=100, linewidth=2)

        if plot_type == "overlapping":
            for i in range(spec_all.shape[0]):
                plt.fill_between(
                    x,
                    spec_all[i, ...] + spec_stddev_all[i, ...],
                    spec_all[i, ...] - spec_stddev_all[i, ...],
                    alpha=1.0 / spec_all.shape[0],
                    zorder=100,
                    color=p1[0].get_color(),
                )
        elif plot_type == "fill":
            if mc:
                err = np.where(
                    spec_err > mean_mc_err, np.sqrt(spec_err**2 - mean_mc_err**2), 0
                )
            else:
                confint = sms.DescrStatsW(spec_all).tconfint_mean()
                err = confint
            plt.fill_between(
                x,
                spec + err,
                spec - err,
                alpha=0.6,
                zorder=100,
                # step="mid"
            )
            if mc:
                plt.fill_between(
                    x,
                    spec + err,
                    spec + err + mean_mc_err,
                    alpha=0.2,
                    zorder=100,
                    # step="mid",
                    color="k",
                )
                plt.fill_between(
                    x,
                    spec - err,
                    spec - err - mean_mc_err,
                    alpha=0.2,
                    zorder=100,
                    # step="mid",
                    color="k",
                )

        return p1[0]

    def plot_cgmf_spec(self, d, quantity, x, mc=True):
        spec_all = d.vector_qs[quantity]
        spec_stddev_all = d.vector_qs[quantity + "_stddev"]
        return self.plot_spec(spec_all, spec_stddev_all, x, d.label, mc)

    def plot_cgmf_spec_from_tensor(self, d, quantity, x, index, mc=True):
        spec_all = d.tensor_qs[quantity][:, index, :]
        spec_stddev_all = d.tensor_qs[quantity + "_stddev"][:, index, :]
        return self.plot_spec(spec_all, spec_stddev_all, x, d.label, mc)

    def plot_cgmf_vec_from_tensor(self, d, quantity, x, index, mc=True):
        vec_all = d.tensor_qs[quantity][:, index, :]
        vec_stddev_all = d.tensor_qs[quantity + "_stddev"][:, index, :]
        return self.plot_vec(vec_all, vec_stddev_all, x, d.label, mc)

    def pfns(self, cgmf_datasets=None, temp=None, allowed_labels=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            # normalization
            x = d.bins["pfns"]
            pfns = d.vector_qs["pfns"]
            pfns_err = d.vector_qs["pfns_stddev"]

            if temp is not None:
                pfns, pfns_err = normalize_to_maxwell(
                    x, pfns, pfns_err, temp, ratio=True
                )
            else:
                pfns, pfns_err = normalize_to_maxwell(x, pfns, pfns_err, temp_MeV=1.21)

            plts_sim.append(self.plot_spec(pfns, pfns_err, x, d.label))

        def plt_exp_spec(s, l):
            if allowed_labels is None or (allowed_labels is not None and l in allowed_labels):
                x = s.bins
                pfns = s.spec
                pfns_err = s.err
                if temp is not None:
                    pfns, pfns_err = normalize_to_maxwell(
                        x, pfns, pfns_err, temp, ratio=True
                    )
                else:
                    pfns, pfns_err = normalize_to_maxwell(
                        x, pfns, pfns_err, temp_MeV=1.21
                    )

                return plt.errorbar(
                    x,
                    pfns,
                    yerr=pfns_err,
                    xerr=s.xerr,
                    alpha=0.7,
                    label=l,
                    linestyle="none",
                )

        pfns = read(
            self.exp_data_path, "pfns", self.energy_range, allowed_labels=allowed_labels
        )
        specs = pfns[0].get_specs()
        labels = [m["label"] for m in pfns[0].meta]
        units = pfns[0].units

        plts = []

        for s, l, u in zip(specs, labels, units):
            s = s.normalizePxdx()
            p = plt_exp_spec(s, l)
            plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=12, ncol=3)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=3, loc="lower left")

        plt.xlabel(r"$E_{lab}$ [MeV]")
        plt.ylabel(r"$p(E_{\rm{lab}}) / " + r"M(E, kT = {}$ MeV$)$".format(temp), fontsize=16)

    def pfgs(self, cgmf_datasets=None, allowed_labels=None):
        plts_sim = []

        for d in cgmf_datasets:
            x = d.bins["pfgs"]
            plts_sim.append(self.plot_cgmf_spec(d, "pfgs", x))

        # experimental data
        pfgs = read(
            self.exp_data_path, "pfgs", self.energy_range, allowed_labels=allowed_labels
        )

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
            if allowed_labels is None or (allowed_labels is not None and l in allowed_labels):
                p = plt_exp_spec(s.normalizePxdx(), l)
                plts.append(p)

        lexp = plt.legend(handles=plts, fontsize=9, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=3, loc="lower left")

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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="upper left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=3, loc="lower left")

        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$\bar{\nu}_\gamma$ [photons]")

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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=3, loc="lower left")

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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=3, loc="lower right")

        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$\bar{\nu}_\gamma$ [photons]")

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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="upper left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc="lower right")

        plt.xlabel(r"$Z$ [protons]")
        plt.ylabel(r"$\bar{\nu}$ [neutrons]")

    def nubarA(self, cgmf_datasets=None, allowed_labels=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_vec(d, "nubarA", d.bins["nubarA"]))

        # experiment
        nubarA = read(self.exp_data_path, "nubarA", self.energy_range)

        labels = [m["label"] for m in nubarA.meta]
        plts = []

        for d, l in zip(nubarA.data, labels):
            if allowed_labels is None or (allowed_labels is not None and l in allowed_labels):
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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=2, loc="upper left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc="lower right")

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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc="upper left")

        plt.xlabel(r"$\nu_\gamma$ [photons]")
        plt.ylabel(r"$p(\nu_\gamma)$")
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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc="upper left")

        plt.xlabel(r"$\nu$ [neutrons]")
        plt.ylabel(r"$p(\nu)$  ")
        plt.tight_layout()

    def nugbar(self, cgmf_datasets=None, allowed_labels=None, bins=None):
        # simulation
        plts_sim = []
        max_n = 0
        num_plots = len(cgmf_datasets)
        alphas = np.linspace(0.9, 0.4, num=num_plots)
        orders = np.arange(0, num_plots * 100, 100)
        ma = 0
        num_bins = 16
        for i, d in enumerate(cgmf_datasets):
            nugbar = d.scalar_qs["nugbar"]
            if bins is not None:
                h, e = np.histogram(nugbar, density=False, bins=bins)
                h = h / np.sum(h)
            else:
                h, e = np.histogram(nugbar, density=False)
                delta = e[1:] - e[:-1]
                h = h / delta / np.sum(h)

            num_bins = h.size

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
            if allowed_labels is None or (allowed_labels is not None and l in allowed_labels):
                p = plt.errorbar(
                    [d[0]], [y], xerr=[d[1] / 2], label=l, linestyle="none", marker="."
                )
                plts.append(p)
                y += 0.05 * ma
                i += 1

        lexp = plt.legend(handles=plts, fontsize=12, ncol=2, loc="upper right")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc="lower right")

        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
        #plt.grid(visible=True, axis="x", which="major")
        plt.xlabel(r"$\bar{\nu}_\gamma$ [photons]")
        plt.ylabel(r"$p(\bar{\nu}_\gamma)$")

        return y

    def nubar(self, cgmf_datasets=None, allowed_labels=None, evaluated=None, bins=None):
        # simulation
        plts_sim = []
        max_n = 0
        num_plots = len(cgmf_datasets)
        alphas = np.linspace(0.9, 0.4, num=num_plots)
        orders = np.arange(0, num_plots * 100, 100)
        ma = 0
        for i, d in enumerate(cgmf_datasets):
            nubar = d.scalar_qs["nubar"]

            if bins is not None:
                h, e = np.histogram(nubar, density=False, bins=bins)
                h = h / np.sum(h)
            else:
                h, e = np.histogram(nubar, density=False)
                delta = e[1:] - e[:-1]
                h = h / delta / np.sum(h)

            num_bins = h.size

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

        # experiment
        # nubar = read(self.exp_data_path, "nubar", self.energy_range, allowed_labels=allowed_labels)
        nubar = read(self.exp_data_path, "nubar", self.energy_range)

        labels = [m["label"] for m in nubar.meta]
        plts = []

        y = 0.8 * ma
        i = 0
        for d, l in zip(nubar.data, labels):
            if allowed_labels is None or (allowed_labels is not None and l in allowed_labels):
                p = plt.errorbar(
                    [d[0]], [y], xerr=[d[1] / 2], label=l, linestyle="none", marker="."
                )
                plts.append(p)
                y += 0.05 * ma
                i += 1

        def plot_eval(val):
            (nu, dnu, label) = val
            nu = np.array([nu, nu])
            dnu = np.array([dnu, dnu])
            plts.append(
                plt.fill_betweenx([0, y], nu - dnu, nu + dnu, label=label, alpha=0.2)
            )


        if evaluated is not None:
            if isinstance(evaluated, list):
                for val in evaluated:
                    plot_eval(val)
            else:
                plot_eval(evaluated)


        lexp = plt.legend(handles=plts, fontsize=12, ncol=2, loc='upper right')
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc='lower right')
        #plt.grid(visible=True, axis="x", which="major")
        plt.xlabel(r"$\bar{\nu}$ [neutrons]")
        plt.ylabel(r"$p(\bar{\nu})$")

        return y

    def pfns_A_moments(self, n: int):
        pfnsa = read(self.exp_data_path, "pfnsA", self.energy_range)

        labels = [m["label"] for m in pfnsa.meta]
        plts = []

        for d, l in zip(pfnsa.data, labels):
            data = PFNSA(np.vstack([d[4, :], d[0, :], d[2, :], d[3, :]]))
            specs = data.getSpecs(data.mass)
            means = [s.moment(n) / s.moment(0) for s in specs]
            plt.errorbar(data.mass, means, label=l)

        plt.legend(fontsize=12, ncol=1)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \langle{E}^{%d}\rangle $ [MeV]" % n)

    def plot_exp_data_over(self, x, y, dx=None, dy=None, label=None, num_sets=1, remove_zeros=True):

        if remove_zeros:
            mask = y > 0
            y = y[mask]
            x = x[mask]
            if dx is not None:
                dx = dx[mask]
            if dy is not None:
                dy = dy[mask]

        if num_sets == 1:
            return plt.errorbar(
                x,
                y,
                yerr=dy,
                xerr=dx,
                marker=".",
                linestyle="none",
                color="k",
                markersize=12,
                label=label,
                zorder=999,
            )
        else:
            return plt.errorbar(
                x,
                y,
                yerr=dy,
                xerr=dx,
                marker=".",
                linestyle="none",
                markersize=12,
                label=label,
                zorder=999,
            )

    def pfnsA(self, a: int, pfnsa, cgmf_datasets=None, temp=None):
        plts_sim = []
        for d in cgmf_datasets:
            index = np.nonzero(a == d.abins)[0][0]
            x = d.bins["pfnscomA"][1]
            pfns = d.tensor_qs["pfnscomA"][:, index, :]
            pfns_err = d.tensor_qs["pfnscomA_stddev"][:, index, :]
            if temp is not None:
                pfns, pfns_err = normalize_to_maxwell(
                    x, pfns, pfns_err, temp, ratio=True
                )
            else:
                pfns, pfns_err = normalize_to_maxwell(x, pfns, pfns_err)
            plts_sim.append(self.plot_spec(pfns, pfns_err, x, d.label, mc=True))

        labels = [m["label"] for m in pfnsa.meta]

        plts = []
        for d, l in zip(pfnsa.data, labels):
            data = PFNSA(np.vstack([d[4, :], d[0, :], d[2, :], d[3, :]]))
            x, pfns, pfns_err = data.getPFNS(a)
            if temp is not None:
                pfns, pfns_err = normalize_to_maxwell(
                    x, pfns, pfns_err, temp, ratio=True
                )
            else:
                pfns, pfns_err = normalize_to_maxwell(x, pfns, pfns_err)

            plts.append(
                self.plot_exp_data_over(
                    x, pfns, dy=pfns_err, label=l, num_sets=len(labels)
                )
            )

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="lower left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1)

        plt.xlabel(r"$E_{\rm cm}$ [MeV]")
        plt.ylabel(
            r"$p(E | A = {}) / M(E, kT = {}$ MeV$)$".format(a, temp), fontsize=16
        )
        # plt.ylabel(r"$p(E_{\rm cm} | A = %d)$" % a, fontsize=16)

    def nubarATKE(self, a: int, nubaratke, cgmf_datasets=None):
        plts_sim = []

        for d in cgmf_datasets:
            (abins, TKEbins) = d.bins["nuATKE"]
            index = np.nonzero(a == abins)[0][0]
            plts_sim.append(
                self.plot_cgmf_vec_from_tensor(d, "nuATKE", TKEbins, index, mc=True)
            )

        labels = [m["label"] for m in nubaratke.meta]
        plts = []

        for d, l in zip(nubaratke.data, labels):
            amin = d[4, :]
            mask = a == amin
            tke = d[0, :][mask]
            dtke = d[1, :][mask]
            nu = d[2, :][mask]
            dnu = d[3, :][mask]
            plts.append(
                self.plot_exp_data_over(
                    tke, nu, dy=dnu, dx=dtke, label=l, num_sets=len(labels)
                )
            )

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="lower left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1)

        plt.xlabel(r"$TKE$ [MeV]")
        plt.ylabel(r"$ \langle \nu | A = {} \rangle $ [neutrons]".format(a))

    def nubartATKE(self, a: int, nubaratke, cgmf_datasets=None):
        plts_sim = []

        for d in cgmf_datasets:
            (abins, TKEbins) = d.bins["nutATKE"]
            index = np.nonzero(a == abins)[0][0]
            plts_sim.append(
                self.plot_cgmf_vec_from_tensor(d, "nutATKE", TKEbins, index, mc=True)
            )

        labels = [m["label"] for m in nubaratke.meta]
        plts = []

        for d, l in zip(nubaratke.data, labels):
            amin = d[4, :]
            mask = a == amin
            tke = d[0, :][mask]
            dtke = d[1, :][mask]
            nu = d[2, :][mask]
            dnu = d[3, :][mask]
            plts.append(
                self.plot_exp_data_over(
                    tke, nu, dy=dnu, dx=dtke, label=l, num_sets=len(labels)
                )
            )

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="lower left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1)

        plt.xlabel(r"$TKE$ [MeV]")
        plt.ylabel(
            r"$ \langle \nu_t | A = {},{} \rangle $ [neutrons]".format(a, 252 - a)
        )

    def encomATKE(self, a: int, encomatke, cgmf_datasets=None):
        plts_sim = []
        for d in cgmf_datasets:
            (abins, TKEbins) = d.bins["nutATKE"]
            index = np.nonzero(a == abins)[0][0]
            plts_sim.append(
                self.plot_cgmf_vec_from_tensor(d, "encomATKE", TKEbins, index, mc=True)
            )

        labels = [m["label"] for m in encomatke.meta]
        plts = []

        for d, l in zip(encomatke.data, labels):
            mask = a == d[4, :]
            tke = d[0, :][mask]
            dtke = d[1, :][mask]
            encom = d[2, :][mask]
            dencom = d[3, :][mask]
            plts.append(
                self.plot_exp_data_over(
                    tke, encom, dy=dencom, dx=dtke, label=l, num_sets=len(labels)
                )
            )

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc="lower left")
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=3)

        plt.xlabel(r"$TKE$ [MeV]")
        plt.ylabel(r"$ \langle E_{\rm cm} | A = %d \rangle $ [neutrons]" % a)

    def multratA(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(
                self.plot_cgmf_vec(d, "multratioA", d.bins["multratioA"], mc=True)
            )

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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc=2)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \frac{ \nu_\gamma }{ \nu_n }$")

    def encomA(self, cgmf_datasets=None):
        # sim
        plts_sim = []
        for d in cgmf_datasets:
            plts_sim.append(self.plot_cgmf_vec(d, "encomA", d.bins["encomA"], mc=True))

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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc=2)
        plt.xlabel(r"$A$ [u]")
        plt.ylabel(r"$ \langle {E}_{\rm cm} \rangle$ [MeV]")

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
                self.plot_exp_data_over(
                    d[0, :],
                    d[2, :],
                    dy = d[3, :],
                    dx= d[1, :],
                    label=l,
                    num_sets=len(labels),
                    remove_zeros=False
                )
            )

        lexp = plt.legend(
            handles=plts,
            fontsize=12,
            ncol=1,
            loc="best",
            bbox_to_anchor=(0.5, 0.7, 0.5, 0.3),
        )
        plt.gca().add_artist(lexp)
        plt.legend(
            handles=plts_sim,
            fontsize=12,
            ncol=1,
            loc="best",
            bbox_to_anchor=(0.5, 0.5, 0.5, 0.3),
        )
        plt.xlabel(r"TKE [MeV]")
        plt.ylabel(r"$ \langle{E}_{\rm cm}\rangle $ [MeV]")

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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc=2)
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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc=2)
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

        lexp = plt.legend(handles=plts, fontsize=12, ncol=1, loc=1)
        plt.gca().add_artist(lexp)
        plt.legend(handles=plts_sim, fontsize=12, ncol=1, loc=2)
        plt.xlabel(r"$\nu$ [neutrons]")
        plt.ylabel(r"$ \langle{E}^T_\gamma\rangle $ [MeV]")
