import sys
from pathlib import Path
import numpy as np
import pandas as pd

from CGMFtk import histories as fh
from cgmf_uq import process_ensemble, calculate_ensemble_nubar

# where are we putting the results
hist_dir = Path(__file__).parent
res_dir     = hist_dir / "post"
default_dir = hist_dir / "default"

hist_fname_prefix  = "histories"
hist_fname_postfix = ".o"

total_ensembles = 300

all_quantities = [
        "nubar",
        "nugbar",
        "nubarA",
        "nugbarA",
        "nubarZ",
        "nugbarZ",
        "nubarTKE",
        "nugbarTKE",
        "pnu" ,
        "pnug",
        "egtbarnu",
        "pfns",
        "pfgs",
        "multratioA",
        "pfnsTKE",
        "pfgsTKE",
        "pfnsA",
        "pfgsA",
        "nuATKE"
        ]

def hist_from_list_of_lists(num, lol, bins, mask_generator=None, out=False):
    """
    returns a histogram of values that may occur multiple times per history, e.g.
    neutron energies, formatted as a list of lists; outer being history, and inner being
    list of quantities associated with that history.

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
            h = np.array(lol[i])
            h = h[ mask_generator(i) ]
            numi = h.size
            v[c: c + numi] = h
            c = c + numi
    else:
        for i in range(lol.size):
            h = np.array(lol[i])
            numi = h.size
            v[c: c + numi] = h
            c = c + numi

    m = np.mean(v[0:c])
    hist , _ = np.histogram(v[0:c], bins=bins, density=True)

    if out:
        out = v[0:c]

        return hist, m, out

    return hist, m

class HistData:
    def __init__(self, ensemble_range : tuple, quantities : list):
        #TODO mc uncertanties

        self.quantities = quantities

        self.scalar_qs = {}
        self.vector_qs = {}
        self.tensor_qs = {}

        # get ensemble range
        self.min_ensemble = ensemble_range[0]
        self.max_ensemble = ensemble_range[1]
        nensemble = self.max_ensemble - self.min_ensemble

        # default processing parameters
        self.Ethn = 0 # neutron detection lower energy threshold
        self.Ethg = 0 # gamma detection lower energy threshold
        self.min_time = 0
        self.max_time = 1E3

        # set up histogram grids
        self.nubins  = np.arange(0,11)
        self.nugbins = np.arange(0,26)

        self.zbins = np.arange(38, 61)
        self.abins = np.arange(70, 185)

        nTKE        = 20
        TKEmin      = 120
        TKEmax      = 220
        TKEstep     = (TKEmax - TKEmin)/nTKE

        self.TKEbins     = np.arange(TKEmin, TKEmax, TKEstep)
        self.TKEcenters  = 0.5*(self.TKEbins[0:-1] + self.TKEbins[1:])

        self.ebins    = np.logspace(-3,2,100)
        self.ecenters = 0.5*(self.ebins[0:-1] + self.ebins[1:])
        self.de       = (self.ebins[1:] - self.ebins[:-1])

        # allocate arrays for histogram values
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
                self.vector_qs["nugbarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
            elif q == "pnu" :
                self.vector_qs["pnu" ] = np.zeros((nensemble, self.nubins.size))
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
                self.tensor_qs["pfnsTKE"] = np.zeros((nensemble, self.TKEcenters.size, self.ecenters.size))
            elif q == "pfgsTKE":
                self.tensor_qs["pfgsTKE"] = np.zeros((nensemble, self.TKEcenters.size, self.ecenters.size))
            elif q == "pfnsA":
                self.tensor_qs["pfnsA"] = np.zeros((nensemble, self.abins.size, self.ecenters.size))
            elif q == "pfgsA":
                self.tensor_qs["pfgsA"] = np.zeros((nensemble, self.abins.size, self.ecenters.size))
            elif q == "nuATKE":
                self.tensor_qs["nuATKE"] = np.zeros((nensemble, self.abins.size, self.TKEcenters.size))
            else:
                print("Unkown quantity: {}".format(q))
                exit(1)

    def concat_from_array_job(self, num_jobs):
        interval = int(total_ensembles / num_jobs)
        for i in range(num_jobs):
            start = interval * i
            end = start + interval
            f = "ensembles_{}_to_{}".format(start, end)
            for k in self.scalar_qs:
                tmp = np.load(res_dir /"{}_{}.npy".format(key,f))
                self.scalar_qs[key][start:end] = tmp
            for k in vector_qs:
                tmp = np.load(res_dir /"{}_{}.npy".format(key,f))
                self.scalar_qs[key][start:end,:] = tmp
            for k in tensor_qs:
                tmp = np.load(res_dir /"{}_{}.npy".format(key,f))
                self.scalar_qs[key][start:end,:,:] = tmp


    def write_bins(self):
        #TODO
        print("Not impl")
        exit(1)

    def write(self, with_ensemble_idx=False):

        if with_ensemble_idx:
            f = "_ensembles_{}_to_{}".format(self.min_ensemble, self.max_ensemble )
        else:
            f = ""

        for k,v in self.scalar_qs.items():
            np.save(res_dir /"{}{}.npy".format(k,f), v)

        for k,v in self.vector_qs.items():
            np.save(res_dir /"{}{}.npy".format(k,f), v)

        for k,v in self.tensor_qs.items():
            np.save(res_dir /"{}{}.npy".format(k,f), v)

    def read(self):
        concat_from_array_job(1)

    def process_ensemble(self, hs : fh.Histories, n : int):

        # self.scalars
        if "nubar" in self.scalar_qs:
            self.scalar_qs[ "nubar"][n] = hs.nubar()
        if "nugbar" in self.scalar_qs:
            self.scalar_qs["nugbar"][n] = hs.nubarg(timeWindow=None, Eth=self.Ethg)

        # multiplicity dependent vector quantities
        def first_from(x, y):
            return y

        if "pnu" in self.vector_qs:
            self.vector_qs[ "pnu"][n] = first_from( *hs.Pnu(Eth=self.Ethn ,  nu=self.nubins ))
        if "pnug" in self.vector_qs:
            self.vector_qs["pnug"][n] = first_from( *hs.Pnug(Eth=self.Ethg, nug=self.nugbins))

        # energy dependent self.vector quantities
        if "pfns" in self.vector_qs:
            self.vector_qs["pfns"][n]  = first_from(*hs.pfns(egrid=self.ebins, Eth=self.Ethn))
        if "pfgs" in self.vector_qs:
            self.vector_qs["pfgs"][n]  = first_from(
                    *hs.pfgs(egrid=self.ebins, Eth=self.Ethg,
                             minTime=self.min_time, maxTime=self.max_time)
                    )

        # nu dependent
        if "egtbarnu" in self.vector_qs:
            for l, nu in enumerate(self.nubins):
                mask        = np.where(hs.getNu() == nu)
                num_gammas  = np.sum( hs.getNug() [mask] )
                nglab       = hs.getGammaElab() [mask]
                _ , self.vector_qs["egtbarnu"][n,l] = hist_from_list_of_lists(
                            num_gammas, nglab, bins=self.ebins)

        # Z dependent
        for l, z in enumerate(self.zbins):
            mask  = np.where(hs.Z == z)
            if "nubarZ" in self.vector_qs:
                self.vector_qs[ "nubarZ"][n,l] = np.mean( hs.getNu()  [mask] )
            if "nugbarZ" in self.vector_qs:
                self.vector_qs["nugbarZ"][n,l] = np.mean( hs.getNug() [mask] )

        # TKE dependent
        for l in range(self.TKEcenters.size):
            TKE_min = self.TKEbins[l]
            TKE_max = self.TKEbins[l+1]
            TKE     = hs.getTKEpost()
            mask    = np.logical_and( TKE >= TKE_min , TKE < TKE_max)

            num_neutrons  = np.sum( hs.getNutot()  [mask] )
            num_gammas    = np.sum( hs.getNugtot() [mask] )

            # < nu | TKE >
            if "nubarTKE" in self.vector_qs:
                self.vector_qs[ "nubarTKE"][n,l] = np.mean( hs.getNutot()[mask] )
            if "nugbarTKE" in self.vector_qs:
                self.vector_qs["nugbarTKE"][n,l] = np.mean( hs.getNugtot()[mask] )

            # for PFNS and PFGS, data is fragment by fragment, rather than event by event
            mask = np.hstack( zip(mask,mask) )

            # < nu | E_n, TKE >
            if "pfnsTKE" in self.tensor_qs:
                nelab  = hs.getNeutronElab()[mask]
                necm   = hs.getNeutronEcm()[mask]
                KE_pre = hs.getKEpre()[mask] / hs.getA()[mask]

                def kinematic_cut(hist : int):
                    return np.where( np.array(necm[hist]) > KE_pre[hist] )

                self.tensor_qs["pfnsTKE"][n,l,:], _ = hist_from_list_of_lists(
                        num_neutrons, nelab, bins=self.ebins, mask_generator=kinematic_cut)

            # < nu_g | E_g, TKE >
            if "pfgsTKE" in self.tensor_qs:
                nglab      = hs.getGammaElab()[mask]
                self.tensor_qs["pfgsTKE"][n,l,:], _ = hist_from_list_of_lists(
                            num_gammas, nglab, bins=self.ebins)

        # A dependent
        for l, a in enumerate(self.abins):
            mask   = np.where(hs.getA() == a)
            num_ns = np.sum( hs.getNu()  [mask] )
            num_gs = np.sum( hs.getNug() [mask] )

            # < * | A >
            # TODO add back energy and time cutoff masks
            if "nubarA" in self.vector_qs:
                self.vector_qs[ "nubarA"][n,l] = np.mean( hs.nu[mask] )
            if "nugbarA" in self.vector_qs:
                self.vector_qs["nugbarA"][n,l] = np.mean( hs.nug[mask] )

            if "multratioA" in self.vector_qs:
                mult_ratio =  hs.getNug() [mask] / hs.getNu() [mask]
                self.vector_qs["multratioA"] = np.mean( mult_ratio )

            # < nu | E_n, A >
            if "pfnsA" in self.tensor_qs:
                nelab  = hs.getNeutronElab()[mask]
                necm   = hs.getNeutronEcm()[mask]
                KE_pre = hs.getKEpre()[mask] / float(a)

                def kinematic_cut(hist : int):
                    return np.where( np.array(necm[hist]) > KE_pre[hist] / float(a) )

                self.tensor_qs["pfnsA"][n,l,:], _ = hist_from_list_of_lists(
                        num_ns, nelab, bins=self.ebins, mask_generator=kinematic_cut)

            # < nu_g | E_g, A >
            if "pfgsA" in self.tensor_qs:
                nglab      = hs.getGammaElab()[mask]
                self.tensor_qs["pfgsA"][n,l,:], _ = hist_from_list_of_lists(
                            num_gs, nglab, bins=self.ebins)

            # < nu | TKE, A >
            if "nuATKE" in self.tensor_qs:
                for m in range(self.TKEcenters.size):
                    TKE_min = self.TKEbins[m]
                    TKE_max = self.TKEbins[m+1]
                    TKE     = hs.getTKEpost()
                    mask    = np.logical_and(
                                np.logical_and( TKE >= TKE_min , TKE < TKE_max),
                                np.logical_or( hs.getAHF() == a , hs.getALF() == a )
                            )
                    self.tensor_qs["nuATKE"][n,l,m] = np.mean( hs.getNutot()[mask] )

    def post_process(self):

        print("Running ensembles {} to {}".format(self.min_ensemble, self.max_ensemble))
        nensemble = self.max_ensemble - self.min_ensemble

        for n in range(0,nensemble):

            fname = hist_dir / ("{}_{}{}".format(hist_fname_prefix, n, hist_fname_postfix))

            print("Reading {} ...".format(fname))
            hs = fh.Histories(fname, ang_mom_printed=True)
            #hs = fh.Histories(fname, ang_mom_printed=True, nevents=100)

            print("Processing {} histories from {} ...".format( hs.getNu().size, fname))
            hd.process_ensemble(hs,n)

            if n < nensemble -1:
                print("Done! onto the the next ensemble...\n")
            else:
                print("Done with all ensembles!\n")

        f = "ensembles_{}_to_{}".format(self.min_ensemble, self.max_ensemble)
        print("Writing output to *_{}.npy".format(f))
        hd.write()

if __name__ == "__main__":
    if sys.argv[1] == "--concat":
        hd = HistData((0,total_ensembles-1))
        hd.concat_from_array_job(int(sys.argv[2]))
        hd.write(with_ensemble_idx=False)
        hd.write_bins(with_ensemble_idx=False)
    else:
        num_jobs = int(sys.argv[1])
        job_num  = int(sys.argv[2])
        interval = int(total_ensembles / num_jobs)
        start = interval * job_num
        end = start + interval
        hd = HistData((start, end), all_quantities)
        hd.post_process()
        hd.write()
