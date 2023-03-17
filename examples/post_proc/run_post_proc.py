import sys
from pathlib import Path
import numpy as np
import pandas as pd

from CGMFtk import histories as fh
from cgmf_uq import process_ensemble, calculate_ensemble_nubar

# where are we putting the results
hist_dir = Path(__file__).parent
res_dir = hist_dir / "post"
default_dir = hist_dir / "default"

total_hists = 300

def concat_arrays(num_jobs):
    pfns = []
    nug = []
    nubars = []
    interval = int(total_hists / num_jobs)
    for i in range(num_jobs):
        start = interval * i
        end = start + interval
        f = "h{}_to_{}".format(start, end)
        pfns.append(resdir /np.load("pfns_a_{}.npy".format(f)))
        nug.append(resdir /np.load("nug_a_{}.npy".format(f)))
        nubars.append(resdir /np.load("nubars_{}.npy".format(f)))

    np.save(resdir / "pfns_A.npy", np.concatenate(pfns))
    np.save(resdir / "nug_A.npy", np.concatenate(nug))
    np.save(resdir / "nubars.npy", np.concatenate(nug))

def concat_dfs(num_jobs):
    dfs = []
    interval = int(total_hists / num_jobs)
    for i in range(num_jobs):
        start = interval * i
        end = start + interval
        f = "h{}_to_{}".format(start, end)
        dfs.append( pd.read_csv(resdir /"hdf_{}.csv".format(f)) )


    pd.concat(dfs).to_csv(resdir /"hdf_all.csv".format(f))

def hist_from_list_of_lists(num, lol, bins, energies=None, Emin=None, out=False):

    v = np.zeros(num)

    c = 0
    if Emin is not None:
        assert(energies is not None)
        for i in range(lol.size):
            h = np.array(lol[i])
            h = h[ np.where(energies[i] >= Emin[i]) ]
            numi = h.size
            v[c: c + numi] = h
            c = c + numi
    else:
        for i in range(lol.size):
            h = lol[i]
            numi = h.size
            v[c: c + numi] = h
            c = c + numi

    hist , _ = np.histogram(v[0:c], bins=bins, density=True)

    if out:
        out = v[0:c]

        return hist, out

    return hist

def build_df(ensemble_range):
    min_ensemble = ensemble_range[0]
    max_ensemble = ensemble_range[1]
    print("Running ensembles {} to {}".format(start,end))
    nensemble = max_ensemble - min_ensemble

    dfs = []

    for i in range(0,nensemble):

        fname = hist_dir / ("histories_{}.o".format(i))
        print("Reading {} ...".format(fname))
        hs = fh.Histories(fname, ang_mom_printed=True)
        #hs = fh.Histories(fname, ang_mom_printed=True, nevents=100)
        print("Processing {} histories from {} ...".format( len(hs.getNu()), fname))

        # extract relevant data as data frame
        data = [ hs.Ah, hs.Al, hs.KEh, hs.KEl, hs.nuHF, hs.nuLF, hs.nugHF, hs.nugLF, np.ones_like(hs.Ah)*i ]
        names = [ "AHF", "ALF", "KEHF", "KELF", "nuHF", "nuLF", "nugHF", "nugLF", "ensemble"]
        dfs.append(pd.DataFrame.from_dict(dict(zip(names, data))))

    f = "h{}_to_{}".format(min_ensemble, max_ensemble )
    print("Writing output to *{}.csv".format(f))
    pd.concat(dfs).to_csv(resdir /"hdf_{}.csv".format(f))

def run_post(ensemble_range):

    min_ensemble = ensemble_range[0]
    max_ensemble = ensemble_range[1]
    print("Running ensembles {} to {}".format(start,end))
    nensemble = max_ensemble - min_ensemble

    minA  = 84
    maxA  = 184
    nA    = maxA - minA + 1
    abins = np.arange(minA, maxA + 1, 1)

    nebins = 40
    emin   = 0.01
    emax   = 10
    step   = (emax - emin)/nebins

    jmin   = -10
    jmax   =  5
    njbins = jmax + 1 - jmin
    jbins  = np.arange(jmin, jmax+2)

    nubarg = np.zeros((nensemble,nA))
    pfnsa  = np.zeros((nensemble,nA,nebins-1))
    dja    = np.zeros((nensemble,nA,njbins-1))

    ebins    = np.arange(emin,emax,step)
    ecenters = 0.5*(ebins[0:-1] + ebins[1:])
    de       = (ebins[1:] - ebins[:-1])

    nubars = []

    for i in range(0,nensemble):

        fname = hist_dir / ("histories_{}.o".format(i))
        print("Reading {} ...".format(fname))
        hs = fh.Histories(fname, ang_mom_printed=True)
        #hs = fh.Histories(fname, ang_mom_printed=True, nevents=10000)
        print("Processing {} histories from {} ...".format( len(hs.getNu()), fname))

        process_ensemble(hs)
        nubars.append( calculate_ensemble_nubar(str(i),res_dir) )

        A  = np.unique(hs.getA())
        for a in A:
            j = int(a - minA)
            if j >= nA:
                break

            mask = np.where(hs.getA() == a, True, False)
            num_neutrons = np.sum( hs.getNu()[mask] )
            KE_pre = hs.getKEpre()[mask]
            ne = hs.getNeutronEcm()[mask]
            nelab = hs.getNeutronElab()[mask]

            nug = hs.getNug()[mask]
            nubarg[i,j] = np.mean(nug)

            if num_neutrons > 100:
                pfnsa[i,j,:], energies = hist_from_list_of_lists(
                        num_neutrons, ne, ebins, energies=nelab, Emin=(KE_pre/float(a)), out=True)
                # ignore the last bin for integer binning - it's max inclusive
                #dja[i,j,:]   = hist_from_list_of_lists(num_neutrons, nj[mask], jbins )[:-1]

        if i < nensemble -1:
            print("Done! onto the the next file...\n")
        else:
            print("Done with all files!\n")

    f = "h{}_to_{}".format(min_ensemble, max_ensemble )
    print("Writing output to *{}.npy".format(f))
    np.save(resdir /"pfns_a_{}.npy".format(f), pfnsa)
    np.save(resdir /"nug_a_{}.npy".format(f), nubarg)
    #np.save("dj_a_{}.npy".format(f), dja)
    np.save( resdir / "ebins_pfns_a_{}.npy".format(f), ecenters)
    np.save( resdir / "jbins_pj_a_{}.npy".format(f), np.arange(jmin, jmax))
    np.save( resdir / "abins_{}.npy".format(f), abins)
    np.save( resdir / "nubars_{}".format(f), np.array(nubars))

if __name__ == "__main__":
    if sys.argv[1] == "--concat":
        concat_arrays(int(sys.argv[2]))
    elif sys.argv[1] == "--concat-df":
        concat_dfs(int(sys.argv[2]))
    elif sys.argv[1] == "--df":
        num_jobs = int(sys.argv[2])
        job_num  = int(sys.argv[3])
        interval = int(total_hists / num_jobs)
        start = interval * job_num
        end = start + interval
        build_df(( start, end))
    else:
        nubars = []
        process_ensemble(default_dir, res_dir, "default")
        nubars.append( calculate_ensemble_nubar("default", res_dir) )
        np.save(str(res_dir/ "nubars_default"), np.array(nubars))

        num_jobs = int(sys.argv[1])
        job_num  = int(sys.argv[2])
        interval = int(total_hists / num_jobs)
        start = interval * job_num
        end = start + interval
        run_pp(( start, end))
