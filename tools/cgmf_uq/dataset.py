import numpy as np
from pathlib import Path

class DataSetUQUncorr:
    def __init__(self, results_dir: Path, num_samples: int,
            num_hist: int, label : str, default_sample_name="",
            corr=False, min_sample=0, is_def=False):

        self.label = label
        self.num_ebins   = 99
        self.num_nu_bins = 10
        self.num_samples = num_samples
        self.num_hist    = num_hist

        # initialize sample data
        self.nubar = np.load(str(results_dir / "nubars.npy"))[min_sample:min_sample+num_samples]
        self.ebins = np.zeros((num_samples,self.num_ebins))
        self.pfns  = np.zeros((num_samples,self.num_ebins))
        self.nu    = np.zeros((num_samples,self.num_nu_bins))
        self.pnu   = np.zeros((num_samples,self.num_nu_bins))

        # initialize default data
        self.nubar_default = 0
        self.ebins_default = np.zeros(self.num_ebins)
        self.pfns_default  = np.zeros(self.num_ebins)
        self.nu_default    = np.zeros(self.num_nu_bins)
        self.pnu_default   = np.zeros(self.num_nu_bins)

        # read in samples
        for i in range(min_sample, min_sample + num_samples):
            sample_name = str(i) + ".npy"
            nu_tmp  = np.load(str(results_dir / ("nu_"     + sample_name)))
            pnu_tmp = np.load(str(results_dir / ("pnu_"    + sample_name)))
            idx = i - min_sample
            self.nu[idx,0:nu_tmp.shape[0]]   = nu_tmp
            self.pnu[idx,0:pnu_tmp.shape[0]] = pnu_tmp
            self.ebins[idx,:] = np.load(str(results_dir / ("ebins_"  + sample_name)))
            self.pfns[idx,:]  = np.load(str(results_dir / ("pfns_"   + sample_name)))

        # handle defaults
        if is_def:
            # default is just statistical mean
            self.nubar_default = np.mean(self.nubar)
            self.ebins_default = self.ebins
            self.pfns_default  = np.mean(self.pfns,axis=0)
            self.nu_default    = self.nu
            self.pnu_default   = np.mean(self.pnu,axis=0)
        else:
            # read in seperate default sample
            default_dir = results_dir / "default"
            nu_tmp  = np.load(str(default_dir / ("nu"  + default_sample_name + ".npy")))
            pnu_tmp = np.load(str(default_dir / ("pnu" + default_sample_name + ".npy")))
            self.nu_default[0:nu_tmp.shape[0]]   = nu_tmp
            self.pnu_default[0:pnu_tmp.shape[0]] = pnu_tmp
            self.ebins_default = np.load(
                    str(default_dir / ("ebins"  + default_sample_name + ".npy")))
            self.pfns_default  = np.load(
                    str(default_dir / ("pfns"   + default_sample_name + ".npy" )))
            self.nubar_default = np.load(
                    str(default_dir / ("nubars" + default_sample_name + ".npy")))[0]

        if corr:
            self.event_nu_n = np.zeros((num_samples, self.num_hist))
            self.event_nu_p = np.zeros((num_samples, self.num_hist))

            self.event_nu_n_default = np.load(
                        str(default_dir / str("event_nu_n" + default_sample_name + ".npy")),
                        allow_pickle=True
                    )
            self.event_nu_p_default = np.load(
                        str(default_dir / str("event_nu_p" + default_sample_name + ".npy")),
                        allow_pickle=True
                    )

            for i in range(min_sample, min_sample + num_samples):
                sample_name = str(i)
                idx = i - min_sample
                self.event_nu_n[idx,:] = np.load(
                        str(results_dir / str("event_nu_n_" + sample_name + ".npy")),
                        allow_pickle=True
                        )[:num_hist]
                self.event_nu_p[idx,:] = np.load(
                        str(results_dir / str("event_nu_p_" + sample_name + ".npy")),
                        allow_pickle=True
                        )[:num_hist]
