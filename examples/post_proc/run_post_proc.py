import sys
from pathlib import Path
import numpy as np

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
    "nuATKE",
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


def hist_from_list_of_lists(num, lol, bins, mask_generator=None):
    """
    returns a histogram of values that may occur multiple times per history,
    e.g. neutron energies, formatted as a list of lists; outer being history,
    and inner bein list of quantities associated with that history.

    num  - number of histories
    lol  - input data in list of lists format
    bins - bins for histogram


    mask_generator - mask function that takes in a history # and returns
    a mask over the quantities in the history provided by lol
    """

    v = np.zeros(num)

    c = 0
    if mask_generator is not None:
        for i in range(lol.size):
            h = np.asarray(lol[i])
            h = h[mask_generator(i)]
            numi = h.size
            v[c : c + numi] = h
            c = c + numi
    else:
        for i in range(lol.size):
            h = np.asarray(lol[i])
            numi = h.size
            v[c : c + numi] = h
            c = c + numi

    m = np.mean(v[0:c])
    hist, _ = np.histogram(v[0:c], bins=bins, density=True)

    return hist, m


class HistData:
    def __init__(
        self,
        ensemble_range: tuple,
        quantities: list,
        hist_dir=parent_dir,
        res_dir=post_dir,
    ):
        # filesystem dealings
        self.hist_dir = hist_dir
        self.res_dir = res_dir
        self.hist_fname_prefix = "histories"
        self.hist_fname_postfix = ".npy"

        # get ensemble range
        self.min_ensemble = ensemble_range[0]
        self.max_ensemble = ensemble_range[1]
        nensemble = self.max_ensemble - self.min_ensemble + 1

        # aset up dictionaries which hold all our data
        self.quantities = quantities
        self.scalar_qs = {}
        self.vector_qs = {}
        self.tensor_qs = {}

        # default processing parameters
        self.Ethn = 0  # neutron detection lower energy threshold
        self.Ethg = 0  # gamma detection lower energy threshold
        self.min_time = 0
        self.max_time = 1e3

        # set up histogram grids
        self.nubins = np.arange(0, 11)
        self.nugbins = np.arange(0, 26)

        self.zbins = np.arange(38, 61)
        self.abins = np.arange(70, 185)

        nTKE = 20
        TKEmin = 120
        TKEmax = 220
        TKEstep = (TKEmax - TKEmin) / nTKE

        self.TKEbins = np.arange(TKEmin, TKEmax, TKEstep)
        self.TKEcenters = 0.5 * (self.TKEbins[0:-1] + self.TKEbins[1:])

        self.ebins = np.logspace(-3, 2, 100)
        self.ecenters = 0.5 * (self.ebins[0:-1] + self.ebins[1:])
        self.de = self.ebins[1:] - self.ebins[:-1]

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
            elif q == "pnu":
                self.vector_qs["pnu"] = np.zeros((nensemble, self.nubins.size))
            elif q == "pnug":
                self.vector_qs["pnug"] = np.zeros((nensemble, self.nugbins.size))
            elif q == "egtbarnu":
                self.vector_qs["egtbarnu"] = np.zeros((nensemble, self.nubins.size))
            elif q == "pfns":
                self.vector_qs["pfns"] = np.zeros((nensemble, self.ecenters.size))
            elif q == "pfgs":
                self.vector_qs["pfgs"] = np.zeros((nensemble, self.ecenters.size))
            elif q == "multratioA":
                self.vector_qs["multratioA"] = np.zeros((nensemble, self.abins.size))
            elif q == "pfnsTKE":
                self.tensor_qs["pfnsTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size, self.ecenters.size)
                )
            elif q == "pfgsTKE":
                self.tensor_qs["pfgsTKE"] = np.zeros(
                    (nensemble, self.TKEcenters.size, self.ecenters.size)
                )
            elif q == "pfnsA":
                self.tensor_qs["pfnsA"] = np.zeros(
                    (nensemble, self.abins.size, self.ecenters.size)
                )
            elif q == "pfgsA":
                self.tensor_qs["pfgsA"] = np.zeros(
                    (nensemble, self.abins.size, self.ecenters.size)
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
            key = k + "_variance"
            self.scalar_qs[key] = np.zeros_like(self.scalar_qs[k])
        for k in list(self.vector_qs):
            key = k + "_variance"
            self.vector_qs[key] = np.zeros_like(self.vector_qs[k])
        for k in list(self.tensor_qs):
            key = k + "_variance"
            self.tensor_qs[key] = np.zeros_like(self.tensor_qs[k])

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

    def estimate(histories):
        h2 = histories**2
        hbar = np.mean(histories)
        hvar = 1 / histories.size * (np.mean(h2) - hbar**2)
        return hbar, hvar

    def process_ensemble(self, hs: fh.Histories, n: int):
        # scalar quantities
        if "nubar" in self.scalar_qs:
            self.scalar_qs["nubar"][n], self.scalar_qs["nubar_variance"][n] = estimate(
                hs.nuLF + hs.nuHF + hs.preFissionNu
            )
        if "nugbar" in self.scalar_qs:
            self.scalar_qs["nugbar"][n] = hs.nubargtot(timeWindow=None, Eth=self.Ethg)

        if "pnu" in self.vector_qs:
            _, self.vector_qs["pnu"][n] = hs.Pnu(Eth=self.Ethn, nu=self.nubins)
        if "pnug" in self.vector_qs:
            _, self.vector_qs["pnug"][n] = hs.Pnug(Eth=self.Ethg, nug=self.nugbins)

        # energy dependent vector quantities
        if "pfns" in self.vector_qs:
            _, self.vector_qs["pfns"][n] = hs.pfns(egrid=self.ebins, Eth=self.Ethn)

        if "pfgs" in self.vector_qs:
            _, self.vector_qs["pfgs"][n] = hs.pfgs(
                egrid=self.ebins,
                Eth=self.Ethg,
                minTime=self.min_time,
                maxTime=self.max_time,
            )

        # nu dependent
        if "egtbarnu" in self.vector_qs:
            for l, nu in enumerate(self.nubins):
                mask = np.where(hs.getNu() == nu)
                num_gammas = np.sum(hs.getNug()[mask])
                nglab = hs.getGammaElab()[mask]
                _, self.vector_qs["egtbarnu"][n, l] = hist_from_list_of_lists(
                    num_gammas, nglab, bins=self.ebins
                )

        # Z dependent
        for l, z in enumerate(self.zbins):
            mask = np.where(hs.Z == z)
            if "nubarZ" in self.vector_qs:
                self.vector_qs["nubarZ"][n, l] = np.mean(hs.getNu()[mask])
            if "nugbarZ" in self.vector_qs:
                self.vector_qs["nugbarZ"][n, l] = np.mean(hs.getNug()[mask])

        # TKE dependent
        for l in range(self.TKEcenters.size):
            TKE_min = self.TKEbins[l]
            TKE_max = self.TKEbins[l + 1]
            TKE = hs.getTKEpost()
            mask = np.logical_and(TKE >= TKE_min, TKE < TKE_max)

            num_neutrons = np.sum(hs.getNutot()[mask])
            num_gammas = np.sum(hs.getNugtot()[mask])

            # < nu | TKE >
            if "nubarTKE" in self.vector_qs:
                self.vector_qs["nubarTKE"][n, l] = np.mean(hs.getNutot()[mask])
            if "nugbarTKE" in self.vector_qs:
                self.vector_qs["nugbarTKE"][n, l] = np.mean(hs.getNugtot()[mask])

            # for PFNS and PFGS, data is fragment by fragment, rather than event by event
            mask = mask.repeat(2, axis=0)

            # < d nu / dE | TKE >
            if "pfnsTKE" in self.tensor_qs:
                nelab = hs.getNeutronElab()[mask]
                necm = hs.getNeutronEcm()[mask]
                KE_pre = hs.getKEpre()[mask] / hs.getA()[mask]

                def kinematic_cut(hist: int):
                    return np.where(np.asarray(necm[hist]) > KE_pre[hist])

                self.tensor_qs["pfnsTKE"][n, l, :], _ = hist_from_list_of_lists(
                    num_neutrons, nelab, bins=self.ebins, mask_generator=kinematic_cut
                )

            # < d nu_g / dE_g | TKE >
            if "pfgsTKE" in self.tensor_qs:
                nglab = hs.getGammaElab()[mask]
                self.tensor_qs["pfgsTKE"][n, l, :], _ = hist_from_list_of_lists(
                    num_gammas, nglab, bins=self.ebins
                )

        # A dependent
        for l, a in enumerate(self.abins):
            mask = np.where(hs.getA() == a)
            num_ns = np.sum(hs.getNu()[mask])
            num_gs = np.sum(hs.getNug()[mask])

            # < * | A >
            # TODO add back energy and time cutoff masks
            if "nubarA" in self.vector_qs:
                self.vector_qs["nubarA"][n, l] = np.mean(hs.nu[mask])
            if "nugbarA" in self.vector_qs:
                self.vector_qs["nugbarA"][n, l] = np.mean(hs.nug[mask])

            if "multratioA" in self.vector_qs:
                mult_ratio = np.mean(hs.getNug()[mask]) / np.mean(hs.getNu()[mask])
                self.vector_qs["multratioA"][n, l] = mult_ratio

            # < d nu / d E_n | A >
            if "pfnsA" in self.tensor_qs:
                nelab = hs.getNeutronElab()[mask]
                necm = hs.getNeutronEcm()[mask]
                KE_pre = hs.getKEpre()[mask] / float(a)

                def kinematic_cut(hist: int):
                    min_energy = KE_pre[hist] / float(a)
                    return np.where(np.asarray(necm[hist]) > min_energy)

                self.tensor_qs["pfnsA"][n, l, :], _ = hist_from_list_of_lists(
                    num_ns, nelab, bins=self.ebins, mask_generator=kinematic_cut
                )

            # < d nu_g / d E_g | A >
            if "pfgsA" in self.tensor_qs:
                nglab = hs.getGammaElab()[mask]
                self.tensor_qs["pfgsA"][n, l, :], _ = hist_from_list_of_lists(
                    num_gs, nglab, bins=self.ebins
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
                    self.tensor_qs["nuATKE"][n, l, m] = np.mean(hs.getNutot()[mask])

    def gather(self, mpi_comm, rank, size):
        if mpi_comm is None:
            return
        for k, v in self.scalar_qs.items():
            result = mpi_comm.gather(v, root=0)
            if rank == 0:
                self.scalar_qs[k] = np.concatenate(result)
        for k, v in self.vector_qs.items():
            result = mpi_comm.gather(v, root=0)
            if rank == 0:
                self.vector_qs[k] = np.concatenate(result)
        for k, v in self.tensor_qs.items():
            result = mpi_comm.gather(v, root=0)
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
                print("Reading {} on rank {}...".format(fname, mpi_comm.Get_rank()))
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


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = HistData((int(sys.argv[1]), int(sys.argv[2])), all_quantities)
    data.post_process(mpi_comm=comm)
    data.gather(comm, rank, size)
    if rank == 0:
        print('Writing output to "{}/*.npy"'.format(str(data.res_dir)))
        sys.stdout.flush()
        data.write()
        data.write_bins()


if __name__ == "__main__":
    main()
