import sys
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
    ):
        # filesystem dealings
        self.hist_dir = Path(hist_dir)
        self.res_dir = Path(res_dir)
        self.hist_fname_prefix = "histories"
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
            ],
        )
        self.scalar_qs = {}
        self.vector_qs = {}
        self.tensor_qs = {}

        # default processing parameters
        self.Ethn = 0.001  # neutron detection lower energy threshold
        self.Ethg = 0.10   # gamma detection lower energy threshold
        self.min_time = 0
        self.max_time = 10E9

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

        self.tgebins = np.logspace(-1, 1, 60)
        self.tgecenters = 0.5 * (self.tgebins[0:-1] + self.tgebins[1:])
        self.tde = self.tgebins[1:] - self.tgebins[:-1]

        # allocate arrays for histogram values
        # all arrays have as their first axis the ensemble
        for q in self.quantities:
            if q == "nubar":
                self.scalar_qs["nubar"] = np.zeros((nensemble))
            elif q == "nugbar":
                self.scalar_qs["nugbar"] = np.zeros((nensemble))
            elif q == "nubarA":
                self.vector_qs["nubarA"] = np.zeros((nensemble, self.abins.size))
            elif q == "nugbarA":
                self.vector_qs["nugbarA"] = np.zeros((nensemble, self.abins.size))
            elif q == "enbarA":
                self.vector_qs["enbarA"] = np.zeros((nensemble, self.abins.size))
            elif q == "encomA":
                self.vector_qs["encomA"] = np.zeros((nensemble, self.abins.size))
            elif q == "egtbarA":
                self.vector_qs["egtbarA"] = np.zeros((nensemble, self.abins.size))
            elif q == "enbarZ":
                self.vector_qs["enbarZ"] = np.zeros((nensemble, self.zbins.size))
            elif q == "egtbarZ":
                self.vector_qs["egtbarZ"] = np.zeros((nensemble, self.zbins.size))
            elif q == "nubarZ":
                self.vector_qs["nubarZ"] = np.zeros((nensemble, self.zbins.size))
            elif q == "nugbarZ":
                self.vector_qs["nugbarZ"] = np.zeros((nensemble, self.zbins.size))
            elif q == "nubarTKE":
                self.vector_qs["nubarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
            elif q == "nugbarTKE":
                self.vector_qs["nugbarTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size)
                )
            elif q == "enbarTKE":
                self.vector_qs["enbarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
            elif q == "encomTKE":
                self.vector_qs["encomTKE"] = np.zeros((nensemble, self.TKEcenters.size))
            elif q == "egtbarTKE":
                self.vector_qs["egtbarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
            elif q == "pnu":
                self.vector_qs["pnu"] = np.zeros((nensemble, self.nubins.size))
            elif q == "pnug":
                self.vector_qs["pnug"] = np.zeros((nensemble, self.nugbins.size))
            elif q == "egtbarnu":
                self.vector_qs["egtbarnu"] = np.zeros((nensemble, self.nubins.size))
            elif q == "pfns":
                self.vector_qs["pfns"] = np.zeros((nensemble, self.ecenters.size))
            elif q == "pfgs":
                self.vector_qs["pfgs"] = np.zeros((nensemble, self.gecenters.size))
            elif q == "multratioA":
                self.vector_qs["multratioA"] = np.zeros((nensemble, self.abins.size))
            elif q == "pfnsTKE":
                self.tensor_qs["pfnsTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size, self.tecenters.size)
                )
            elif q == "pfgsTKE":
                self.tensor_qs["pfgsTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size, self.tgecenters.size)
                )
            elif q == "pfnscomTKE":
                self.tensor_qs["pfnscomTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size, self.tecenters.size)
                )
            elif q == "pfnsZ":
                self.tensor_qs["pfnsZ"] = np.zeros(
                    (nensemble, self.zbins.size, self.tecenters.size)
                )
            elif q == "pfgsZ":
                self.tensor_qs["pfgsZ"] = np.zeros(
                    (nensemble, self.zbins.size, self.tgecenters.size)
                )
            elif q == "pfnsA":
                self.tensor_qs["pfnsA"] = np.zeros(
                    (nensemble, self.abins.size, self.tecenters.size)
                )
            elif q == "pfnscomA":
                self.tensor_qs["pfnscomA"] = np.zeros(
                    (nensemble, self.abins.size, self.tecenters.size)
                )
            elif q == "pfnscomZ":
                self.tensor_qs["pfnscomZ"] = np.zeros(
                    (nensemble, self.zbins.size, self.tecenters.size)
                )
            elif q == "pfgsA":
                self.tensor_qs["pfgsA"] = np.zeros(
                    (nensemble, self.abins.size, self.tgecenters.size)
                )
            elif q == "nuATKE":
                self.tensor_qs["nuATKE"] = np.zeros(
                    (nensemble, self.abins.size, self.TKEcenters.size)
                )
            else:
                print("Unkown quantity: {}".format(q))
                sys.stdout.flush()
                exit(1)

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

    def write_bins(self, with_ensemble_idx=False, mpi_comm=None):
        f = ""
        if with_ensemble_idx:
            f = "_ensembles_{}_to_{}".format(self.min_ensemble, self.max_ensemble)
        print('Writing bins to "{}/*{}_bins.npy" '.format(str(self.res_dir), f))
        sys.stdout.flush()
        np.save(self.res_dir / "nu_bins.npy", self.nubins)
        np.save(self.res_dir / "nug_bins.npy", self.nugbins)
        np.save(self.res_dir / "Z_bins.npy", self.zbins)
        np.save(self.res_dir / "A_bins.npy", self.abins)
        np.save(self.res_dir / "TKE_bins.npy", self.TKEbins)
        np.save(self.res_dir / "E_bins.npy", self.ebins)
        np.save(self.res_dir / "TE_bins.npy", self.tebins)
        np.save(self.res_dir / "TGE_bins.npy", self.tgebins)
        np.save(self.res_dir / "GE_bins.npy", self.gebins)

    def write(self, with_ensemble_idx=True):
        if with_ensemble_idx:
            f = "_ensembles_{}_to_{}".format(self.min_ensemble, self.max_ensemble)
        else:
            f = ""

        for k, v in self.scalar_qs.items():
            np.save(self.res_dir / "{}{}.npy".format(k, f), v)

        for k, v in self.vector_qs.items():
            np.save(self.res_dir / "{}{}.npy".format(k, f), v)

        for k, v in self.tensor_qs.items():
            np.save(self.res_dir / "{}{}.npy".format(k, f), v)

    def read(self, with_ensemble_idx=True):
        self.res_dir = Path(self.res_dir)
        if with_ensemble_idx:
            f = "_ensembles_{}_to_{}".format(self.min_ensemble, self.max_ensemble)
        else:
            f = ""

        for k in self.scalar_qs:
            self.scalar_qs[k] = np.load(self.res_dir / "{}{}.npy".format(k, f))

        for k in self.vector_qs:
            self.vector_qs[k] = np.load(self.res_dir / "{}{}.npy".format(k, f))

        for k in self.tensor_qs:
            self.tensor_qs[k] = np.load(self.res_dir / "{}{}.npy".format(k, f))

        self.nubins = np.load(self.res_dir / "nu_bins.npy")
        self.nugbins = np.load(self.res_dir / "nug_bins.npy")
        self.zbins = np.load(self.res_dir / "Z_bins.npy")
        self.abins = np.load(self.res_dir / "A_bins.npy")
        self.TKEbins = np.load(self.res_dir / "TKE_bins.npy")
        self.ebins = np.load(self.res_dir / "E_bins.npy")
        self.tebins = np.load(self.res_dir / "TE_bins.npy")
        self.tgebins = np.load(
            self.res_dir / "TGE_bins.npy",
        )
        self.gebins = np.load(self.res_dir / "GE_bins.npy")

    def hist_from_list_of_lists(self, num, lol, bins, mask_generator=None, totals=False, fragment=True):
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
        v = np.zeros(num)
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
                # sum every 2 totals together
                totals_v = totals_v.reshape((-1,2)).sum(1)
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
            return np.where(
                np.logical_and(
                    np.logical_and(
                        np.asarray(ages_lol[i]) >= self.min_time,
                        np.asarray(ages_lol[i]) < self.max_time,
                    ),
                    np.asarray(energy_lol[i]) > self.Ethg,
                )
            )

        return cut

    def neutron_cut(self, energy_lol: np.array):
        """
        Given list of lists representing neutron energies for each fragment, returns
        a callable that takes in a fragment index and returns a mask selecting the
        neutrons  within the energy cut
        """

        def cut(i: int):
            return np.where(np.asarray(energy_lol[i]) > self.Ethn)

        return cut

    def kinematic_cut(self, necm : np.array, ke_pre: np.array, a : float):
        """
        Given COM neutron energies in list of lists format, and the pre-emission
        kinetic energy for each fragment, returns a callable that takes in a fragment
        index and returns a mask selecting the neutrons with kinetic energy larger than
        the kinetic energy per nucleon of the pre-emission fragment
        """
        def cut(i : int):
            return np.where( np.asarray(necm[i]) >= ke_pre[i]/a )

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
                nutot, self.nubins, int_bins=True
            )

        if "pnug" in self.vector_qs:
            nugtot = (
                hs.getNugEnergyCut(self.Ethg).reshape((hs.numberEvents, 2)).sum(axis=1)
            )
            (
                self.vector_qs["pnug"][n],
                self.vector_qs["pnug_stddev"][n],
            ) = self.histogram_with_binomial_uncertainty(
                nugtot, self.nugbins, int_bins=True
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
                num_neutrons, nelab, self.ebins, self.neutron_cut(nelab)
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
                num_gammas, gelab, self.gebins, self.gamma_cut(gelab, ages)
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
                    bins=self.ebins,
                    mask_generator=self.gamma_cut(gelab, ages),
                    totals=True
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
                    num_ns, nelab, bins=self.tebins, mask_generator=self.neutron_cut(nelab)
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
                    bins=self.tgebins,
                    mask_generator=self.gamma_cut(gelab, ages),
                    totals=True
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
                    num_neutrons, nelab, bins=self.tebins, mask_generator=self.neutron_cut(nelab)
                )

            if "pfnscomTKE" in self.tensor_qs or "encomTKE" in self.vector_qs:
                necm = hs.getNeutronEcm()[mask]

                (
                    self.tensor_qs["pfnscomTKE"][n, l, :],
                    self.tensor_qs["pfnscomTKE_stddev"][n, l, :],
                    self.vector_qs["encomTKE"][n, l],
                    self.vector_qs["encomTKE_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_neutrons, necm, bins=self.tebins
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
                    bins=self.tgebins,
                    mask_generator=self.gamma_cut(gelab, ages),
                    totals=True
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
                necm = hs.getNeutronEcm()[mask]
                nelab = hs.getNeutronElab()[mask]
                KE_pre = hs.getKEpre()[mask]

                (
                    self.tensor_qs["pfnsA"][n, l, :],
                    self.tensor_qs["pfnsA_stddev"][n, l, :],
                    self.vector_qs["enbarA"][n, l],
                    self.vector_qs["enbarA_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_ns, nelab, bins=self.tebins, mask_generator=self.kinematic_cut(necm, KE_pre, a)
                )

            if "pfnscomA" in self.tensor_qs or "encomA" in self.vector_qs:
                necm = hs.getNeutronEcm()[mask]
                KE_pre = hs.getKEpre()[mask]

                (
                    self.tensor_qs["pfnscomA"][n, l, :],
                    self.tensor_qs["pfnscomA_stddev"][n, l, :],
                    self.vector_qs["encomA"][n, l],
                    self.vector_qs["encomA_stddev"][n, l],
                    _,
                ) = self.hist_from_list_of_lists(
                    num_ns, necm, bins=self.tebins, mask_generator=self.kinematic_cut(necm, KE_pre, a)
                )

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
                    bins=self.tgebins,
                    mask_generator=self.gamma_cut(gelab, ages),
                    totals=True
                )

            # < nu | TKE, A >
            if "nuATKE" in self.tensor_qs:
                for m in range(self.TKEcenters.size):
                    TKE_min = self.TKEbins[m]
                    TKE_max = self.TKEbins[m + 1]
                    TKE = hs.getTKEpost()
                    mask = np.logical_and(
                        np.logical_and(TKE >= TKE_min, TKE < TKE_max),
                        np.logical_or(hs.getAHF() == a, hs.getALF() == a),
                    )

                    (
                        self.tensor_qs["nuATKE"][n, l, m],
                        self.tensor_qs["nuATKE_stddev"][n, l, m],
                    ) = self.estimate_mean(hs.getNutot()[mask])

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
                "Running ensembles {} to {}".format(
                    self.min_ensemble, self.max_ensemble
                )
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
                print("On lap {}; reading {} on rank {}...".format(n % mpi_comm.Get_size(), fname, mpi_comm.Get_rank()))
                sys.stdout.flush()

            if self.hist_fname_postfix == ".npy":
                hs = fh.Histories.load(fname)
            else:
                hs = fh.Histories(filename=fname, ang_mom_printed=True)

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
