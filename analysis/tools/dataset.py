import numpy as np
from pathlib import Path

class DataSet:
    def __init__(self, results_dir: Path, num_samples: int,
            num_hist: int, label : str, default_sample_name="", corr=False):

        self.label = label
        self.num_ebins   = 99
        self.num_nu_bins = 10
        self.num_samples = num_samples
        self.num_hist    = num_hist

        # initialize sample data
        self.nubar = np.load(str(results_dir / "nubars.npy"))
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
        for i in range(0, num_samples):
            sample_name = str(i) + ".npy"
            nu_tmp  = np.load(str(results_dir / ("nu_"     + sample_name)))
            pnu_tmp = np.load(str(results_dir / ("pnu_"    + sample_name)))
            self.nu[i,0:nu_tmp.shape[0]]   = nu_tmp
            self.pnu[i,0:pnu_tmp.shape[0]] = pnu_tmp
            self.ebins[i,:] = np.load(str(results_dir / ("ebins_"  + sample_name)))
            self.pfns[i,:]  = np.load(str(results_dir / ("pfns_"   + sample_name)))

        # read in defaults
        default_dir = results_dir / "default"
        nu_tmp  = np.load(str(default_dir / ("nu"  + default_sample_name + ".npy")))
        pnu_tmp = np.load(str(default_dir / ("pnu" + default_sample_name + ".npy")))
        self.nu_default[0:nu_tmp.shape[0]]   = nu_tmp
        self.pnu_default[0:pnu_tmp.shape[0]] = pnu_tmp
        self.ebins_default = np.load(str(default_dir / ("ebins"  + default_sample_name + ".npy")))
        self.pfns_default  = np.load(str(default_dir / ("pfns"   + default_sample_name + ".npy" )))
        self.nubar_default = np.load(str(default_dir / ("nubars" + default_sample_name + ".npy")))

        if corr:
            self.event_nu_n = np.zeros((254, self.num_hist))
            self.event_nu_p = np.zeros((254, self.num_hist))

            self.event_nu_n_default = np.load(
                        str(default_dir / str("event_nu_n" + default_sample_name + ".npy")),
                        allow_pickle=True
                    )
            self.event_nu_p_default = np.load(
                        str(default_dir / str("event_nu_p" + default_sample_name + ".npy")),
                        allow_pickle=True
                    )

            #for i in range(0, num_samples):
            for i in range(0,253):
                sample_name = str(i)
                self.event_nu_n[i,:] = np.load(
                        str(results_dir / str("event_nu_n_" + sample_name + ".npy")),
                        allow_pickle=True
                    )
                self.event_nu_p[i,:] = np.load(
                        str(results_dir / str("event_nu_p_" + sample_name + ".npy")),
                        allow_pickle=True
                    )
