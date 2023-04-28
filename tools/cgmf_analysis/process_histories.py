import sys
import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from mpi4py import MPI

from CGMFtk import histories as fh

parent_dir = Path(__file__).parent
post_dir = Path(__file__).parent / "post"

all_quantities = [
    "nubar",
    "nugbar",
    "nubarA",
    "nugbarA",
    "nubarZ",
    "nugbarZ",
    "nubarTKE",
    "nugbarTKE",
    "enbarA",
    "egtbarA",
    "enbarZ",
    "egtbarZ",
    "enbarTKE",
    "egtbarTKE",
    "pnu",
    "pnug",
    "egtbarnu",
    "pfns",
    "pfgs",
    "multratioA",
    "pfnsTKE",
    "pfgsTKE",
    "pfnsA",
    "pfgsA",
    "pfnsZ",
    "pfgsZ",
    "nuATKE",
    "pfnscomTKE",
    "pfnscomA",
    "encomA",
    "encomTKE",
    "encomATKE",
]


def balance_load(num_jobs, rank, size):
    stride, remainder = divmod(num_jobs, size)
    if stride > 0:
        ranks_wout_extra_job = size - remainder
        # divide up remainder evently among last ranks
        if rank <= ranks_wout_extra_job:
            start = rank * stride
            end = start + stride
            if rank == ranks_wout_extra_job:
                end = end + 1
        else:
            start = ranks_wout_extra_job * stride + (rank - ranks_wout_extra_job) * (
                stride + 1
            )
            end = start + stride + 1
    else:
        if rank < remainder:
            start = rank
            end = start + 1
        else:
            return 0, 0

    return start, end


def check_pairs(quantities, pairs):
    for pair in pairs:
        if pair[0] in quantities and pair[1] not in quantities:
            quantities.append(pair[1])
        if pair[1] in quantities and pair[0] not in quantities:
            quantities.append(pair[0])
    return list(dict.fromkeys(quantities))  # remove duplicates preserving order


class HistData:
    def __init__(
        self,
        ensemble_range: tuple,
        quantities: list,
        hist_dir=parent_dir,
        res_dir=post_dir,
        convert_cgmf_to_npy=False,
    ):
        # filesystem dealings
        self.hist_dir = Path(hist_dir)
        self.res_dir = Path(res_dir)
        self.hist_fname_prefix = "histories"
        self.convert_cgmf_to_npy = convert_cgmf_to_npy
        if convert_cgmf_to_npy:
            self.hist_fname_postfix = ".o"
        else:
            self.hist_fname_postfix = ".npy"

        # get ensemble range
        self.min_ensemble = ensemble_range[0]
        self.max_ensemble = ensemble_range[1]
        nensemble = self.max_ensemble - self.min_ensemble + 1

        # aset up dictionaries which hold all our data
        self.quantities = check_pairs(
            quantities,
            [
                ("enbarA", "pfnsA"),
                ("egtbarA", "pfgsA"),
                ("enbarTKE", "pfnsTKE"),
                ("egtbarTKE", "pfgsTKE"),
                ("enbarZ", "pfnsZ"),
                ("egcomTKE", "pfgscomTKE"),
                ("encomTKE", "pfnscomTKE"),
                ("encomA", "pfnscomA"),
                ("encomZ", "pfnscomZ"),
                ("egtbarZ", "pfgsZ"),
                ("multratioA", "nubarA"),
                ("multratioA", "nugbarA"),
                ("nuATKE", "nutATKE"),
            ],
        )
        self.scalar_qs = {}
        self.vector_qs = {}
        self.tensor_qs = {}

        # default processing parameters
        self.Ethn = 0.001  # neutron detection lower energy threshold
        self.Ethg = 0.10  # gamma detection lower energy threshold
        self.min_time = 0
        self.max_time = 10e9

        # set up histogram grids
        self.nubins = np.arange(0, 11)
        self.nugbins = np.arange(0, 26)

        self.zbins = np.arange(38, 61)
        self.abins = np.arange(74, 180)

        nTKE = 60
        TKEmin = 120
        TKEmax = 240
        TKEstep = (TKEmax - TKEmin) / nTKE

        self.TKEbins = np.arange(TKEmin, TKEmax, TKEstep)
        self.TKEcenters = 0.5 * (self.TKEbins[0:-1] + self.TKEbins[1:])

        self.ebins = np.logspace(-3, 2, 60)
        self.ecenters = 0.5 * (self.ebins[0:-1] + self.ebins[1:])
        self.de = self.ebins[1:] - self.ebins[:-1]

        # ebins for gammas
        self.gebins = np.logspace(-2, 1, 200)
        self.gecenters = 0.5 * (self.gebins[0:-1] + self.gebins[1:])
        self.gde = self.gebins[1:] - self.gebins[:-1]

        # ebins for tensor q's q/ lower statistics
        self.tebins = np.logspace(-3, 2, 30)
        self.tecenters = 0.5 * (self.tebins[0:-1] + self.tebins[1:])
        self.tde = self.tebins[1:] - self.tebins[:-1]

        # center of mass bins for neutron tensor q's
        self.com_tebins = np.logspace(-1.1, 1.1, 30)
        self.com_tecenters = 0.5 * (self.com_tebins[0:-1] + self.com_tebins[1:])
        self.com_tde = self.com_tebins[1:] - self.com_tebins[:-1]

        # ebins for tensor gamma q's
        self.tgebins = np.logspace(-1, 1, 60)
        self.tgecenters = 0.5 * (self.tgebins[0:-1] + self.tgebins[1:])
        self.tde = self.tgebins[1:] - self.tgebins[:-1]

        self.bins = {}
        self.bin_edges = {}

        # allocate arrays for histogram values
        # all arrays have as their first axis the ensemble
        for q in self.quantities:
            if q == "nubar":
                self.scalar_qs["nubar"] = np.zeros((nensemble))
            elif q == "nugbar":
                self.scalar_qs["nugbar"] = np.zeros((nensemble))
            elif q == "nubarA":
                self.vector_qs["nubarA"] = np.zeros((nensemble, self.abins.size))
                self.bins["nubarA"] = self.abins
            elif q == "nugbarA":
                self.vector_qs["nugbarA"] = np.zeros((nensemble, self.abins.size))
                self.bins["nugbarA"] = self.abins
            elif q == "enbarA":
                self.vector_qs["enbarA"] = np.zeros((nensemble, self.abins.size))
                self.bins["enbarA"] = self.abins
            elif q == "encomA":
                self.vector_qs["encomA"] = np.zeros((nensemble, self.abins.size))
                self.bins["encomA"] = self.abins
            elif q == "egtbarA":
                self.vector_qs["egtbarA"] = np.zeros((nensemble, self.abins.size))
                self.bins["egtbarA"] = self.abins
            elif q == "multratioA":
                self.vector_qs["multratioA"] = np.zeros((nensemble, self.abins.size))
                self.bins["multratioA"] = self.abins
            elif q == "enbarZ":
                self.vector_qs["enbarZ"] = np.zeros((nensemble, self.zbins.size))
                self.bins["enbarZ"] = self.zbins
            elif q == "egtbarZ":
                self.vector_qs["egtbarZ"] = np.zeros((nensemble, self.zbins.size))
                self.bins["egtbarZ"] = self.zbins
            elif q == "nubarZ":
                self.vector_qs["nubarZ"] = np.zeros((nensemble, self.zbins.size))
                self.bins["nubarZ"] = self.zbins
            elif q == "nugbarZ":
                self.vector_qs["nugbarZ"] = np.zeros((nensemble, self.zbins.size))
                self.bins["nugbarZ"] = self.zbins
            elif q == "nubarTKE":
                self.vector_qs["nubarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
                self.bins["nubarTKE"] = self.TKEcenters
                self.bin_edges["nubarTKE"] = self.TKEbins
            elif q == "nugbarTKE":
                self.vector_qs["nugbarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
                self.bins["nugbarTKE"] = self.TKEcenters
                self.bin_edges["nugbarTKE"] = self.TKEbins
            elif q == "enbarTKE":
                self.vector_qs["enbarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
                self.bins["enbarTKE"] = self.TKEcenters
                self.bin_edges["enbarTKE"] = self.TKEbins
            elif q == "encomTKE":
                self.vector_qs["encomTKE"] = np.zeros((nensemble, self.TKEcenters.size))
                self.bins["encomTKE"] = self.TKEcenters
                self.bin_edges["encomTKE"] = self.TKEbins
            elif q == "egtbarTKE":
                self.vector_qs["egtbarTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size)
                )
                self.bins["egtbarTKE"] = self.TKEcenters
                self.bin_edges["egtbarTKE"] = self.TKEbins
            elif q == "pnu":
                self.vector_qs["pnu"] = np.zeros((nensemble, self.nubins.size))
                self.bins["pnu"] = self.nubins
            elif q == "pnug":
                self.vector_qs["pnug"] = np.zeros((nensemble, self.nugbins.size))
                self.bins["pnug"] = self.nugbins
            elif q == "egtbarnu":
                self.vector_qs["egtbarnu"] = np.zeros((nensemble, self.nubins.size))
                self.bins["egtbarnu"] = self.nubins
            elif q == "pfns":
                self.vector_qs["pfns"] = np.zeros((nensemble, self.ecenters.size))
                self.bins["pfns"] = self.ecenters
                self.bin_edges["pfns"] = self.ebins
            elif q == "pfgs":
                self.vector_qs["pfgs"] = np.zeros((nensemble, self.gecenters.size))
                self.bins["pfgs"] = self.gecenters
                self.bin_edges["pfgs"] = self.gebins
            elif q == "pfnsTKE":
                self.tensor_qs["pfnsTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size, self.tecenters.size)
                )
                self.bins["pfnsTKE"] = (self.TKEbins, self.tecenters)
                self.bin_edges["pfnsTKE"] = (self.TKEbins, self.tebins)
            elif q == "pfgsTKE":
                self.tensor_qs["pfgsTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size, self.tgecenters.size)
                )
                self.bins["pfgsTKE"] = (self.TKEbins, self.tgecenters)
                self.bin_edges["pfgsTKE"] = (self.TKEbins, self.tgebins)
            elif q == "pfnscomTKE":
                self.tensor_qs["pfnscomTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size, self.com_tecenters.size)
                )
                self.bins["pfnscomTKE"] = (self.TKEbins, self.com_tecenters)
                self.bin_edges["pfnscomTKE"] = (self.TKEbins, self.com_tebins)
            elif q == "pfnsZ":
                self.tensor_qs["pfnsZ"] = np.zeros(
                    (nensemble, self.zbins.size, self.tecenters.size)
                )
                self.bins["pfnsZ"] = (self.zbins, self.tecenters)
                self.bin_edges["pfnsZ"] = (self.zbins, self.tebins)
            elif q == "pfgsZ":
                self.tensor_qs["pfgsZ"] = np.zeros()
                self.bins["pfgsZ"] = (self.zbins, self.tgebins)
            elif q == "pfnsA":
                self.tensor_qs["pfnsA"] = np.zeros(
                    (nensemble, self.abins.size, self.tecenters.size)
                )
                self.bins["pfnsA"] = (self.abins, self.tecenters)
                self.bin_edges["pfnsA"] = (self.abins, self.tebins)
            elif q == "pfgsA":
                self.tensor_qs["pfgsA"] = np.zeros(
                    (nensemble, self.abins.size, self.tgecenters.size)
                )
                self.bins["pfgsA"] = (self.abins, self.tgecenters)
                self.bin_edges["pfgsA"] = (self.abins, self.tgebins)
            elif q == "pfnscomA":
                self.tensor_qs["pfnscomA"] = np.zeros(
                    (nensemble, self.abins.size, self.com_tecenters.size)
                )
                self.bins["pfnscomA"] = (self.abins, self.com_tecenters)
                self.bin_edges["pfnscomA"] = (self.abins, self.com_tebins)
            elif q == "pfnscomZ":
                self.tensor_qs["pfnscomZ"] = np.zeros(
                    (nensemble, self.zbins.size, self.com_tecenters.size)
                )
                self.bins["pfnscomZ"] = (self.zbins, self.com_tecenters)
                self.bin_edges["pfnscomZ"] = (self.zbins, self.com_tebins)
            elif q == "nuATKE":
                self.tensor_qs["nuATKE"] = np.zeros(
                    (nensemble, self.abins.size, self.TKEcenters.size)
                )
                self.bins["nuATKE"] = (self.abins, self.TKEcenters)
                self.bin_edges["nuATKE"] = (self.abins, self.TKEbins)
            elif q == "nutATKE":
                self.tensor_qs["nutATKE"] = np.zeros(
                    (nensemble, self.abins.size, self.TKEcenters.size)
                )
                self.bins["nutATKE"] = (self.abins, self.TKEcenters)
                self.bin_edges["nutATKE"] = (self.abins, self.TKEbins)
            elif q == "encomATKE":
                self.tensor_qs["encomATKE"] = np.zeros(
                    (nensemble, self.abins.size, self.TKEcenters.size)
                )
                self.bins["encomATKE"] = (self.abins, self.TKEcenters)
                self.bin_edges["encomATKE"] = (self.abins, self.TKEbins)
            else:
                raise ValueError("Unknown quantity: {}".format(q))

        for k in list(self.scalar_qs):
            key = k + "_stddev"
            self.scalar_qs[key] = np.zeros_like(self.scalar_qs[k])
        for k in list(self.vector_qs):
            key = k + "_stddev"
            self.vector_qs[key] = np.zeros_like(self.vector_qs[k])
        for k in list(self.tensor_qs):
            key = k + "_stddev"
            self.tensor_qs[key] = np.zeros_like(self.tensor_qs[k])

    def plot_scalar(self, quantity):
        if quantity not in self.scalar_qs:
            print("invalid key: {}".format(quantity))

        vals = self.scalar_qs[quantity]
        x, y = np.histogram(vals)
        x = x[1:] - x[:-1]
        plt.fill_between(x, y, 0)

    def plot_vector(self, quantity, x):
        if quantity not in self.vector_qs:
            print("invalid key: {}".format(quantity))

        vals = self.vector_qs[quantity]
        vals_unc = self.vector_qs[quantity + "_stddev"]
        for i in range(self.min_ensemble, self.max_ensemble):
            plt.fill_between(
                x,
                vals[i, :] + vals_unc[i, :],
                vals - vals_unc[i, :],
                alpha=0.05,
                color="k",
            )

    def write_bins(self, with_ensemble_idx=True, mpi_comm=None):
        if with_ensemble_idx:
            f = "_samples_{}_to_{}".format(self.min_ensemble, self.max_ensemble)
        else:
            f = ""

        def pickle_dump(data, fpath: Path):
            with open(fpath, "wb") as f:
                pickle.dump(data, f)

        for k, v in self.bins.items():
            pickle_dump(v, self.res_dir / "{}_bins{}.npy".format(k, f))

    def write(self, with_ensemble_idx=True):
        if with_ensemble_idx:
            f = "_samples_{}_to_{}".format(self.min_ensemble, self.max_ensemble)
        else:
            f = ""

        for k, v in self.scalar_qs.items():
            np.save(self.res_dir / "{}{}.npy".format(k, f), v)

        for k, v in self.vector_qs.items():
            np.save(self.res_dir / "{}{}.npy".format(k, f), v)

        for k, v in self.tensor_qs.items():
            np.save(self.res_dir / "{}{}.npy".format(k, f), v)

    def read(self, with_ensemble_idx=True):
        def pickle_load(fpath: Path):
            with open(fpath, "rb") as f:
                return pickle.load(f)

        self.res_dir = Path(self.res_dir)
        if with_ensemble_idx:
            f = "_samples_{}_to_{}".format(self.min_ensemble, self.max_ensemble)
        else:
            f = ""

        for k in self.scalar_qs:
            self.scalar_qs[k] = np.load(self.res_dir / "{}{}.npy".format(k, f))

        for k in self.vector_qs:
            self.vector_qs[k] = np.load(self.res_dir / "{}{}.npy".format(k, f))

        for k in self.tensor_qs:
            self.tensor_qs[k] = np.load(self.res_dir / "{}{}.npy".format(k, f))

        for k in self.bins:
            self.bins[k] = pickle_load(self.res_dir / "{}_bins{}.npy".format(k, f))

    def hist_from_list_of_lists(
        self, num, lol, bins, mask_generator=None, totals=False, fragment=True
    ):
        """
        returns a histogram of values that may occur multiple times per history,
        e.g. neutron energies, formatted as a list of lists; outer being history,
        and inner being list of quantities associated with that history.

        num  - number of histories
        lol  - input data in list of lists format, each entry in the outer list
        corresponding to a single fragment
        bins - bins for histogram
        mask_generator - mask function that takes in a history # and returns
        a mask over the quantities in the history provided by lol
        totals - bool describing whether to also sum the quantity in lol
        over each fragment and find the mean of that
        fragment - bool describing whether to to consider data fragment-by-fragment (True, default),
        or event-by-event (False) for totals. If the latter, sums over every other entry in totals_v
        before finding the means
        """
        v = np.zeros(int(num))
        totals_v = []

        c = 0
        if mask_generator is not None:
            for i in range(lol.size):
                h = np.asarray(lol[i])
                h = h[mask_generator(i)]
                numi = h.size
                v[c : c + numi] = h
                totals_v.append(np.sum(h))
                c = c + numi
        else:
            for i in range(lol.size):
                h = np.asarray(lol[i])
                numi = h.size
                v[c : c + numi] = h
                c = c + numi
                totals_v.append(np.sum(h))

        mean, sem = self.estimate_mean(v[0:c])
        hist, stdev = self.histogram_with_binomial_uncertainty(v[0:c], bins=bins)

        if totals:
            totals_v = np.asarray(totals_v)
            if not fragment:
                # sum every 2 totals together to get totals per fission event
                totals_v = totals_v.reshape((-1, 2)).sum(1)
            meant, semt = self.estimate_mean(totals_v)
            return hist, stdev, mean, sem, meant, semt, c

        return hist, stdev, mean, sem, c

    def histogram_with_binomial_uncertainty(self, histories, bins, int_bins=False):
        """
        For a set of scalar quantities organized by history along axis 0,
        returns a histogram of values over bins, along with uncertainty
        """
        nhist = histories.shape[0]
        if nhist == 0:
            num_bins = bins.size - 1
            if int_bins:
                num_bins = bins.size
            return np.zeros(num_bins), np.zeros(num_bins)

        if int_bins:
            bins = bins - 0.5
            bins = np.append(bins, [bins[-1] + 0.5])

        h, _ = np.histogram(histories, bins=bins, density=False)
        stdev = np.sqrt(h * (1 - h / nhist))
        dbins = bins[1:] - bins[:-1]
        norm = np.sum(h) * dbins
        return h / norm, stdev / norm

    def estimate_mean(self, histories):
        """
        For a set of scalar quantities organized by history along axis 0,
        return the mean, and the standard error in the mean sqrt(stddev/N)
        """
        if histories.size == 0:
            return 0, 0
        h2 = histories**2
        hbar = np.mean(histories, axis=0)
        sem = np.sqrt(1 / histories.size * (np.mean(h2, axis=0) - hbar**2))
        return hbar, sem

    def gamma_cut(self, energy_lol: np.array, ages_lol: np.array):
        """
        Given list of lists representing gamma energies and ages for each fragment,
        returns a callable that takes in a fragment index and returns a mask selecting
        the gammas within the time and energy cuts
        """

        def cut(i: int):
            return np.logical_and(
                np.logical_and(
                    np.asarray(ages_lol[i]) >= self.min_time,
                    np.asarray(ages_lol[i]) < self.max_time,
                ),
                np.asarray(energy_lol[i]) > self.Ethg,
            )

        return cut

    def neutron_cut(self, energy_lol: np.array):
        """
        Given list of lists representing neutron energies for each fragment, returns
        a callable that takes in a fragment index and returns a mask selecting the
        neutrons  within the energy cut
        """

        def cut(i: int):
            return np.asarray(energy_lol[i]) > self.Ethn

        return cut

    def kinematic_cut(self, energy_lol: np.array, min_energy: np.array):
        """
        Given energies in list of lists format, and some min energy cutoff for each
        fragment, returns the corresponding mask for a given fragment number
        """
        assert min_energy.size == energy_lol.size

        def cut(i: int):
            return np.asarray(energy_lol[i]) >= min_energy[i]

        return cut

    def process_ensemble(self, hs: fh.Histories, n: int):
        # TODO enforce cutoffs in energy and time for gammas
        # scalar quantities
        if "nubar" in self.scalar_qs:
            (
                self.scalar_qs["nubar"][n],
                self.scalar_qs["nubar_stddev"][n],
            ) = self.estimate_mean(hs.nuLF + hs.nuHF + hs.preFissionNu)

        if "nugbar" in self.scalar_qs:
            (
                self.scalar_qs["nugbar"][n],
                self.scalar_qs["nugbar_stddev"][n],
            ) = self.estimate_mean(hs.nugLF + hs.nugHF)

        if "pnu" in self.vector_qs:
            nutot = (
                hs.getNuEnergyCut(self.Ethn).reshape((hs.numberEvents, 2)).sum(axis=1)
            )
            (
                self.vector_qs["pnu"][n],
                self.vector_qs["pnu_stddev"][n],
            ) = self.histogram_with_binomial_uncertainty(
                nutot, self.bins["pnu"], int_bins=True
            )

        if "pnug" in self.vector_qs:
            nugtot = (
                hs.getNugEnergyCut(self.Ethg).reshape((hs.numberEvents, 2)).sum(axis=1)
            )
            (
                self.vector_qs["pnug"][n],
                self.vector_qs["pnug_stddev"][n],
            ) = self.histogram_with_binomial_uncertainty(
                nugtot, self.bins["pnug"], int_bins=True
            )

        # energy dependent vector quantities
        if "pfns" in self.vector_qs:
            nelab = hs.getNeutronElab()
            num_neutrons = np.sum(hs.getNutot())
            (
                self.vector_qs["pfns"][n],
                self.vector_qs["pfns_stddev"][n],
                _,
                _,
                _,
            ) = self.hist_from_list_of_lists(
                num_neutrons, nelab, self.bin_edges["pfns"], self.neutron_cut(nelab)
            )

        if "pfgs" in self.vector_qs:
            gelab = hs.getGammaElab()
            ages = hs.getGammaAges()
            num_gammas = np.sum(hs.getNugtot())
            (
                self.vector_qs["pfgs"][n],
                self.vector_qs["pfgs_stddev"][n],
                _,
                _,
                _,
            ) = self.hist_from_list_of_lists(
                num_gammas, gelab, self.bin_edges["pfgs"], self.gamma_cut(gelab, ages)
            )

        # nu dependent
        if "egtbarnu" in self.vector_qs:
            for l, nu in enumerate(self.nubins):
                # TODO egtbar* experiments use energies and multiplcities event by event
                # not fragment bt fragment
                mask = np.where(hs.getNu() == nu)
                num_gammas = np.sum(hs.getNug()[mask])
                gelab = hs.getGammaElab()[mask]
                ages = hs.getGammaElab()[mask]

                (
                    _,
                    _,
                    _,
                    _,
                    self.vector_qs["egtbarnu"][n, l],
                    self.vector_qs["egtbarnu_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_gammas,
                    gelab,
                    bins=self.bin_edges["pfgs"],
                    mask_generator=self.gamma_cut(gelab, ages),
                    totals=True,
                )

        # Z dependent
        for l, z in enumerate(self.zbins):
            # TODO add cuts for energy and time
            mask = np.where(hs.Z == z)
            num_ns = np.sum(hs.getNu()[mask])
            num_gs = np.sum(hs.getNug()[mask])
            if "nubarZ" in self.vector_qs:
                (
                    self.vector_qs["nubarZ"][n, l],
                    self.vector_qs["nubarZ_stddev"][n, l],
                ) = self.estimate_mean(hs.getNu()[mask])

            if "nugbarZ" in self.vector_qs:
                (
                    self.vector_qs["nugbarZ"][n, l],
                    self.vector_qs["nugbarZ_stddev"][n, l],
                ) = self.estimate_mean(hs.getNug()[mask])
            # < d nu / d E_n | A >
            if "pfnsZ" in self.tensor_qs or "enbarZ" in self.vector_qs:
                nelab = hs.getNeutronElab()[mask]
                necm = hs.getNeutronEcm()[mask]

                (
                    self.tensor_qs["pfnsZ"][n, l, :],
                    self.tensor_qs["pfnsZ_stddev"][n, l, :],
                    self.vector_qs["enbarZ"][n, l],
                    self.vector_qs["enbarZ_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_ns,
                    nelab,
                    bins=self.bin_edges["pfnsZ"][1],
                    mask_generator=self.neutron_cut(nelab),
                )

            # < d nu_g / d E_g | A >
            if "pfgsZ" in self.tensor_qs or "egtbarZ" in self.vector_qs:
                gelab = hs.getGammaElab()[mask]
                ages = hs.getGammaAges()[mask]

                (
                    self.tensor_qs["pfgsZ"][n, l, :],
                    self.tensor_qs["pfgsZ_stddev"][n, l, :],
                    _,
                    _,
                    self.vector_qs["egtbarZ"][n, l],
                    self.vector_qs["egtbarZ_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_gs,
                    gelab,
                    bins=self.bin_edges["pfgsZ"][1],
                    mask_generator=self.gamma_cut(gelab, ages),
                    totals=True,
                )

        # TKE dependent
        for l in range(self.TKEcenters.size):
            # TODO add cuts for energy and time
            TKE_min = self.TKEbins[l]
            TKE_max = self.TKEbins[l + 1]
            TKE = hs.getTKEpost()
            mask = np.logical_and(TKE >= TKE_min, TKE < TKE_max)

            num_neutrons = np.sum(hs.getNutot()[mask])
            num_gammas = np.sum(hs.getNugtot()[mask])

            # < nu | TKE >
            if "nubarTKE" in self.vector_qs:
                (
                    self.vector_qs["nubarTKE"][n, l],
                    self.vector_qs["nubarTKE_stddev"][n, l],
                ) = self.estimate_mean(hs.getNutot()[mask])

            if "nugbarTKE" in self.vector_qs:
                (
                    self.vector_qs["nugbarTKE"][n, l],
                    self.vector_qs["nugbarTKE_stddev"][n, l],
                ) = self.estimate_mean(hs.getNugtot()[mask])

            # for PFNS and PFGS, data is fragment by fragment, rather than event by event
            mask = mask.repeat(2, axis=0)

            # < d nu / dE | TKE >
            if "pfnsTKE" in self.tensor_qs or "enbarTKE" in self.vector_qs:
                nelab = hs.getNeutronElab()[mask]

                (
                    self.tensor_qs["pfnsTKE"][n, l, :],
                    self.tensor_qs["pfnsTKE_stddev"][n, l, :],
                    self.vector_qs["enbarTKE"][n, l],
                    self.vector_qs["enbarTKE_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_neutrons,
                    nelab,
                    bins=self.bin_edges["pfnsTKE"][1],
                    mask_generator=self.neutron_cut(nelab),
                )

            if "pfnscomTKE" in self.tensor_qs or "encomTKE" in self.vector_qs:
                nelab = hs.getNeutronElab()[mask]
                necm = hs.getNeutronEcm()[mask]
                ke_pre = hs.getKEpre()[mask]
                A = hs.getA()[mask]
                # Bowman small angle cut (E[Mev] = m_n *(1 cm/ns)**2)
                min_energy = np.ones_like(A) * 1.04540752

                (
                    self.tensor_qs["pfnscomTKE"][n, l, :],
                    self.tensor_qs["pfnscomTKE_stddev"][n, l, :],
                    self.vector_qs["encomTKE"][n, l],
                    self.vector_qs["encomTKE_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_neutrons,
                    necm,
                    bins=self.bin_edges["pfnscomTKE"][1],
                    mask_generator=self.kinematic_cut(nelab, min_energy),
                )

            # < d nu_g / dE_g | TKE >
            if "pfgsTKE" in self.tensor_qs:
                gelab = hs.getGammaElab()[mask]
                ages = hs.getGammaAges()[mask]

                (
                    self.tensor_qs["pfgsTKE"][n, l, :],
                    self.tensor_qs["pfgsTKE_stddev"][n, l, :],
                    _,
                    _,
                    self.vector_qs["egtbarTKE"][n, l],
                    self.vector_qs["egtbarTKE_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_gammas,
                    gelab,
                    bins=self.bin_edges["pfgsTKE"][1],
                    mask_generator=self.gamma_cut(gelab, ages),
                    totals=True,
                )

        # A dependent
        for l, a in enumerate(self.abins):
            # TODO add cuts for energy and time
            mask = np.where(hs.getA() == a)
            num_ns = np.sum(hs.getNu()[mask])
            num_gs = np.sum(hs.getNug()[mask])

            # < * | A n>
            # TODO add back energy and time cutoff masks
            if "nubarA" in self.vector_qs or "multratioA" in self.vector_qs:
                (
                    self.vector_qs["nubarA"][n, l],
                    self.vector_qs["nubarA_stddev"][n, l],
                ) = self.estimate_mean(hs.nu[mask])

            if "nugbarA" in self.vector_qs or "multratioA" in self.vector_qs:
                (
                    self.vector_qs["nugbarA"][n, l],
                    self.vector_qs["nugbarA_stddev"][n, l],
                ) = self.estimate_mean(hs.nug[mask])

            if "multratioA" in self.vector_qs:
                nu, nug = (
                    self.vector_qs["nubarA"][n, l],
                    self.vector_qs["nugbarA"][n, l],
                )
                dnu, dnug = (
                    self.vector_qs["nubarA_stddev"][n, l],
                    self.vector_qs["nugbarA_stddev"][n, l],
                )
                if nu > 0 and dnu > 0:
                    self.vector_qs["multratioA"][n, l] = nug / nu
                    self.vector_qs["multratioA_stddev"][n, l] = np.sqrt(
                        dnug**2 / nu**2 + dnu**2 * (dnug / dnu) ** 2
                    )

            # < d nu / d E_n | A >
            if "pfnsA" in self.tensor_qs or "enbarA" in self.vector_qs:
                nelab = hs.getNeutronElab()[mask]
                necm = hs.getNeutronEcm()[mask]
                ke_pre = hs.getKEpre()[mask]
                min_energy = ke_pre / float(a)

                (
                    self.tensor_qs["pfnsA"][n, l, :],
                    self.tensor_qs["pfnsA_stddev"][n, l, :],
                    self.vector_qs["enbarA"][n, l],
                    self.vector_qs["enbarA_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_ns,
                    nelab,
                    bins=self.bin_edges["pfnsA"][1],
                    mask_generator=self.kinematic_cut(nelab, min_energy),
                )

            if "pfnscomA" in self.tensor_qs or "encomA" in self.vector_qs:
                nelab = hs.getNeutronElab()[mask]
                necm = hs.getNeutronEcm()[mask]
                ke_pre = hs.getKEpre()[mask]
                min_energy = ke_pre / float(a)

                (
                    self.tensor_qs["pfnscomA"][n, l, :],
                    self.tensor_qs["pfnscomA_stddev"][n, l, :],
                    self.vector_qs["encomA"][n, l],
                    self.vector_qs["encomA_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_ns,
                    necm,
                    bins=self.bin_edges["pfnscomA"][1],
                    mask_generator=self.kinematic_cut(nelab, min_energy),
                )
                # TODO use kinematic cut here?

            # < d nu_g / d E_g | A >
            if "pfgsA" in self.tensor_qs or "egtbarA" in self.vector_qs:
                gelab = hs.getGammaElab()[mask]
                ages = hs.getGammaAges()[mask]

                (
                    self.tensor_qs["pfgsA"][n, l, :],
                    self.tensor_qs["pfgsA_stddev"][n, l, :],
                    _,
                    _,
                    self.vector_qs["egtbarA"][n, l],
                    self.vector_qs["egtbarA_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_gs,
                    gelab,
                    bins=self.bin_edges["pfgsA"][1],
                    mask_generator=self.gamma_cut(gelab, ages),
                    totals=True,
                )

            # < nu | TKE, A >
            if "nuATKE" in self.tensor_qs:
                for m in range(self.TKEcenters.size):
                    TKE_min = self.TKEbins[m]
                    TKE_max = self.TKEbins[m + 1]
                    TKE = hs.getTKEpost()
                    mask_nut = np.logical_and(
                        np.logical_and(TKE >= TKE_min, TKE < TKE_max),
                        np.logical_or(hs.getAHF() == a, hs.getALF() == a),
                    )
                    TKE = TKE.repeat(2, axis=0)
                    mask_nu = np.logical_and(
                        np.logical_and(TKE >= TKE_min, TKE < TKE_max),
                        hs.getA() == a,
                    )

                    (
                        self.tensor_qs["nuATKE"][n, l, m],
                        self.tensor_qs["nuATKE_stddev"][n, l, m],
                    ) = self.estimate_mean(hs.getNu()[mask_nu])

                    (
                        self.tensor_qs["nutATKE"][n, l, m],
                        self.tensor_qs["nutATKE_stddev"][n, l, m],
                    ) = self.estimate_mean(hs.getNutot()[mask_nut])

            if "encomATKE" in self.tensor_qs:
                for m in range(self.TKEcenters.size):
                    TKE_min = self.TKEbins[m]
                    TKE_max = self.TKEbins[m + 1]
                    TKE = hs.getTKEpost()
                    TKE = TKE.repeat(2, axis=0)
                    mask = np.logical_and(
                        np.logical_and(TKE >= TKE_min, TKE < TKE_max),
                        hs.getA() == a,
                    )
                    necm = hs.getNeutronEcm()[mask]
                    num_neutrons = np.sum(hs.getNu()[mask])
                    if num_neutrons > 0:
                        # TODO kinematic cut?

                        (
                            _,
                            _,
                            self.tensor_qs["encomATKE"][n, l, m],
                            self.tensor_qs["encomATKE_stddev"][n, l, m],
                            _,
                        ) = self.hist_from_list_of_lists(num_neutrons, necm, np.array([0, 100]))

    def gather(self, mpi_comm, rank, size, rank_slice):
        if mpi_comm is None:
            return

        (s, f) = rank_slice
        for k, v in self.scalar_qs.items():
            result = mpi_comm.gather(v[s:f, ...], root=0)
            if rank == 0:
                self.scalar_qs[k] = np.concatenate(result)
        for k, v in self.vector_qs.items():
            result = mpi_comm.gather(v[s:f, ...], root=0)
            if rank == 0:
                self.vector_qs[k] = np.concatenate(result)
        for k, v in self.tensor_qs.items():
            result = mpi_comm.gather(v[s:f, ...], root=0)
            if rank == 0:
                self.tensor_qs[k] = np.concatenate(result)

    def post_process(self, mpi_comm=None):
        if mpi_comm is None or mpi_comm.Get_rank() == 0:
            print(
                "Running samples {} to {}".format(self.min_ensemble, self.max_ensemble)
            )
            sys.stdout.flush()

        nensemble = self.max_ensemble - self.min_ensemble + 1

        stride = nensemble
        start = 0
        end = start + stride

        # split up work on MPI ranks, balancing load
        if mpi_comm is not None:
            rank, size = mpi_comm.Get_rank(), mpi_comm.Get_size()
            start, end = balance_load(nensemble, rank, size)

        for n in range(start, end):
            ensemble_idx = self.min_ensemble + n

            fname = self.hist_dir / (
                "{}_{}{}".format(
                    self.hist_fname_prefix,
                    ensemble_idx,
                    self.hist_fname_postfix,
                )
            )

            if mpi_comm is None:
                print("Reading {} ...".format(fname))
            else:
                print(
                    "Rank {} on lap {} out of {}; reading {}...".format(
                        mpi_comm.Get_rank(), n - start, end - start, fname
                    )
                )
                sys.stdout.flush()

            if not self.convert_cgmf_to_npy or self.hist_fname_postfix == ".npy":
                hs = fh.Histories.load(fname)
            else:
                hs = fh.Histories(filename=fname, ang_mom_printed=True)
                if self.convert_cgmf_to_npy:
                    fname_out = self.hist_dir / (
                        "{}_{}{}".format(
                            self.hist_fname_prefix,
                            ensemble_idx,
                            ".npy",
                        )
                    )
                    hs.save(fname_out)

            if mpi_comm is None:
                print(
                    "Processing {} histories from {} ...".format(
                        hs.getNutot().size, fname
                    )
                )
            else:
                print(
                    "Processing {} histories from {} on rank {}".format(
                        hs.getNutot().size, fname, mpi_comm.Get_rank()
                    )
                )
                sys.stdout.flush()

            self.process_ensemble(hs, n)

        if mpi_comm is not None:
            return start, end
