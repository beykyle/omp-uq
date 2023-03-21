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
            h = lol[i]
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
        nensemble = max_ensemble - min_ensemble

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
        TKEstep     = (self.TKEmax - self.TKEmin)/self.nTKE

        self.TKEbins     = np.linspace(TKEmin, TKEmax, step=TKEstep)
        self.TKEcenters  = 0.5*(self.TKEbins[0:-1] + self.TKEbins[1:])

        self.ebins    = np.logspace(-3,2,100)
        self.ecenters = 0.5*(ebins[0:-1] + ebins[1:])
        self.de       = (ebins[1:] - ebins[:-1])

        # allocate arrays for histogram values
        self.pfnsA  = np.zeros((nensemble, self.abins.size, self.ecenters.size))
        self.pfgsA  = np.zeros((nensemble, self.abins.size, self.ecenters.size))

        self.nuATKE = np.zeros((nensemble, self.TKEbins.size, self.abins.size))

        for q in quantities:
            if q == "nubar":
                self.scalar_qs["nubar"] = np.zeros((nensemble))
            elif q == "nugbar":
                self.scalar_qs["nugbar"] = np.zeros((nensemble))
            elif q == "nubarA":
                self.scalar_qs["nubarA"] = np.zeros((nensemble, self.abins.size))
            elif q == "nugbarA":
                self.scalar_qs["nugbarA"] = np.zeros((nensemble, self.abins.size))
            elif q == "nubarZ":
                self.scalar_qs["nubarZ"] = np.zeros((nensemble, self.zbins.size))
            elif q == "nugbarZ":
                self.scalar_qs["nugbarZ"] = np.zeros((nensemble, self.zbins.size))
            elif q == "nubarTKE":
                self.scalar_qs["nubarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
            elif q == "nugbarTKE":
                self.scalar_qs["nugbarTKE"] = np.zeros((nensemble, self.TKEcenters.size))
            elif q == "pnu" :
                self.scalar_qs["pnu" ] = np.zeros((nensemble, self.nubins.size))
            elif q == "pnug":
                self.scalar_qs["pnug"] = np.zeros((nensemble, self.nugbins.size))
            elif q == "egtbarnu":
                self.scalar_qs["egtbarnu"] = np.zeros((nensemble, self.nubins.size))
            elif q == "pfns":
                self.scalar_qs["pfns"] = np.zeros((nensemble, self.ecenters.size))
            elif q == "pfgs":
                self.scalar_qs["pfgs"] = np.zeros((nensemble, self.ecenters.size))
            elif q == "multratioA":
                self.scalar_qs["multratioA"] = np.zeros((nensemble, self.abins.size))
            elif q == "pfnsTKE":
                self.scalar_qs["pfnsTKE"] = np.zeros((nensemble, self.TKEcenters.size, self.ecenters.size))
            elif q == "pfgsTKE":
                self.scalar_qs["pfgsTKE"] = np.zeros((nensemble, self.TKEcenters.size, self.ecenters.size))
            elif q == "pfnsA":
                self.scalar_qs["pfnsA"] = np.zeros((nensemble, self.abins.size, self.ecenters.size))
            elif q == "pfgsA":
                self.scalar_qs["pfgsA"] = np.zeros((nensemble, self.abins.size, self.ecenters.size))
            elif q == "nuATKE":
                self.scalar_qs["nuATKE"] = np.zeros((nensemble, self.abins.size, self.TKEcenters.size))
            else:
                print("Unkown quantity: {}".format(q))
                exit(1)

    def concat_from_array_job(num_jobs):
        interval = int(total_ensembles / num_jobs)
        for i in range(num_jobs):
            start = interval * i
            end = start + interval
            f = "ensembles_{}_to_{}".format(start, end)
            for k in scalar_qs:
                tmp = np.load(resdir /"{}_{}.npy".format(key,f))
                self.scalar_qs[key][start:end] = tmp
            for k in vector_qs:
                tmp = np.load(resdir /"{}_{}.npy".format(key,f))
                self.scalar_qs[key][start:end,:] = tmp
            for k in tensor_qs:
                tmp = np.load(resdir /"{}_{}.npy".format(key,f))
                self.scalar_qs[key][start:end,:,:] = tmp


    def write():
        f = "ensembles_{}_to_{}".format(self.min_ensemble, self.max_ensemble )

        for k,v in scalar_qs.items():
            np.save(resdir /"{}_{}.npy".format(key,f), v)

        for k,v in vector_qs.items():
            np.save(resdir /"{}_{}.npy".format(key,f), v)

        for k,v in tensor_qs.items():
            np.save(resdir /"{}_{}.npy".format(key,f), v)

    def read():
        concat_from_array_job(1)

    def process_ensemble(hs : CGMFtk.Histories, n : int):

        # scalars
        if "nubar" is in scalar_qs:
            scalar_qs[ "nubar"][n] = hs.nubar()
        if "nugbar" is in scalar_qs:
            scalar_qs["nugbar"][n] = hs.nubarg(timeWindow=None, Eth=Ethg)

        # multiplicity dependent vector quantities
        if "pnu" is in vector_qs:
            vector_qs[ "pnu"][n] = hs.Pnu(Eth=self.Ethn)
        if "pnug" is in vector_qs:
            vector_qs["pnug"][n] = hs.Pnug(Eth=self.Ethg)

        # energy dependent vector quantities
        if "pfns" is in vector_qs:
            vector_qs["pfns"][n]  = hs.pfns(egrid=self.ebins, Eth=self.Ethn)
        if "pfgs" is in vector_qs:
            vector_qs["pfgs"][n]  = hs.pfgs(
                    egrid=self.ebins, Eth=self.Ethg,
                    minTime=self.min_time, maxTime=self.max_time)

        # nu dependent
        if "egtbarnu" is in vector_qs:
            for l, nu in enumerate(self.nubins):
                mask        = np.where(hs.getNu() == nu)
                num_gammas  = np.sum( hs.getNug() [mask] )
                nglab       = hs.getGamElab() [mask]
                _ , vector_qs["egtbarnu"][n,l] = hist_from_list_of_lists(
                            num_gammas, nglab, bins=self.ebins)

        # Z dependent
        for l, z in enumerate(self.zbins):
            mask  = np.where(hs.Z == z)
            if "nubarZ" is in vector_qs:
                vector_qs[ "nubarZ"][n,l] = np.mean( hs.getNu()  [mask] )
            if "nugbarZ" is in vector_qs:
                vector_qs["nugbarZ"][n,l] = np.mean( hs.getNug() [mask] )

        # TKE dependent
        for l in range(self.TKEbins.size):
            TKE_min = self.TKEbins[l]
            TKE_max = self.TKEbins[l+1]
            TKE     = hs.getTKEpost()
            mask    = np.logical_and( TKE >= TKE_min , TKE < TKE_max)

            num_neutrons  = np.sum( hs.getNu()  [mask] )
            num_gammas    = np.sum( hs.getNug() [mask] )

            # < nu | TKE >
            if "nubarTKE" is in vector_qs:
                vector_qs[ "nubarTKE"][n,l] = np.mean( hs.getNutot()[mask] )
            if "nugbarTKE" is in vector_qs:
                vector_qs["nugbarTKE"][n,l] = np.mean( hs.getNugtot()[mask] )

            # < nu | E_n, TKE >
            if "pfnsTKE" is in tensor_qs:
                nelab  = hs.getNeutronElab()[mask]
                necm   = hs.getNeutronEcm()[mask]
                KE_pre = hs.getKEpre()[mask] / hs.getA()[mask]

                def kinematic_cut(hist : int):
                    return np.where( np.array(necm[hist]) > KE_pre[hist] / float(a) )

                tensor_qs["pfnsTKE"][n,l,:], _ = hist_from_list_of_lists(
                        num_neutrons, nelab, bins=self.ebins, mask_generator=kinematic_cut)

            # < nu_g | E_g, TKE >
            if "pfgsTKE" is in tensor_qs:
                nglab      = hs.getGammaElab()[mask]
                tensor_qs["pfgsTKE"][n,l,:], _ = hist_from_list_of_lists(
                            num_gammas, nglab, bins=self.ebins)

        # A dependent
        for l, a in enumerate(self.abins):
            mask   = np.where(hs.getA() == a)
            num_ns = np.sum( hs.getNu()  [mask] )
            num_gs = np.sum( hs.getNug() [mask] )

            # < * | A >
            if "nubarA" is in vector_qs:
                vector_qs[ "nubarA"][n,l] = np.mean( hs.nubar()   [mask] )
            if "nugbarA" is in vector_qs:
                vector_qs["nugbarA"][n,l] = np.mean( hs.nugbarg() [mask] )

            if "multratioA" is in vector_qs:
                mult_ratio =  hs.getNug() [mask] / hs.gteNu() [mask]
                vector_qs["multratioA"] = np.mean( mult_ratio )

            # < nu | E_n, A >
            if "pfnsA" is in tensor_qs:
                nelab  = hs.getNeutronElab()[mask]
                necm   = hs.getNeutronEcm()[mask]
                KE_pre = hs.getKEpre()[mask] / float(a)

                def kinematic_cut(hist : int):
                    return np.where( np.array(necm[hist]) > KE_pre[hist] / float(a) )

                tensor_qs["pfnsA"][n,l,:], _ = hist_from_list_of_lists(
                        num_ns, nelab, bins=self.ebins, mask_generator=kinematic_cut)

            # < nu_g | E_g, A >
            if "pfgsA" is in tensor_qs:
                nglab      = hs.getGammaElab()[mask]
                tensor_qs["pfgsA"][n,l,:], _ = hist_from_list_of_lists(
                            num_gs, nglab, bins=self.ebins)

            # < nu | TKE, A >
            if "nuATKE" is in tensor_qs:
                for m in range(self.TKEbins.size):
                    TKE_min = self.TKEbins[l]
                    TKE_max = self.TKEbins[l+1]
                    TKE     = hs.getTKEpost()
                    mask    = np.logical_and(
                                np.logical_and( TKE >= TKE_min , TKE < TKE_max),
                                np.logical_or( hs.getAHF() == a , hs.getALF() == a )
                            )
                    tensor_qs["nuATKE"][n,l,m] = np.mean( hs.getNutot()[mask] )

    def post_process():
        print("Running ensembles {} to {}".format(self.min_ensemble, self.max_ensemble))

        for i in range(0,self.max_ensemble - self.min_ensemble):

            fname = hist_dir / ("{}_{}{}".format(hist_fname_prefix, hist_fname_postfix, n))

            print("Reading {} ...".format(fname))
            #hs = fh.Histories(fname, ang_mom_printed=True)
            hs = fh.Histories(fname, ang_mom_printed=True, nevents=100)

            print("Processing {} histories from {} ...".format( hs.getNu().size, fname))
            hd.process_ensemble(hs,i)

            if i < nensemble -1:
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
    else:
        num_jobs = int(sys.argv[1])
        job_num  = int(sys.argv[2])
        interval = int(total_hists / num_jobs)
        start = interval * job_num
        end = start + interval
        hd = HistData((start, end))
        hd.post_process()
