import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.append("/home/beykyle/umich/omp-uq/analysis/tools")

from dataset import DataSet
from plot_nu import plot_nubar, plot_pnu, plot_fm
from plot_pfns import plot_pfns
from plot_nu_corr import plot_nu_corr_dev

#cf_252_datapath = Path(sys.argv[1])
#u_235_datapath = Path(sys/argv[2])
cf_252_datapath = Path("/home/beykyle/db/projects/OM/comp_res/cf252")
u_235_datapath = Path("/home/beykyle/db/projects/OM/comp_res/u235")

cf252 = DataSet(cf_252_datapath, 415, 192000, "252Cf (sf)", corr=True)
u235  = DataSet(u_235_datapath, 415, 192000, "235U (nth,f)", corr=True)

data_sets = [cf252, u235]

for d in data_sets:
    print("{}: mean nubar: {:1.6f} default nubar: {:1.6f}, with {} samples and {} histories/samples"
            .format(d.label, np.mean(d.nubar), d.nubar_default[0], d.num_samples, d.num_hist) )

# plot nubar distributions
plot_nubar(data_sets, save=True, outfile="./pnubar.pdf")
plt.close()
plot_nubar(data_sets, save=True, outfile="./pnubar.png")
plt.close()

plot_pfns(cf252, save=True, outfile="./pfns_cf252.pdf")
plt.close()
plot_pfns(u235, save=True, outfile="./pfns_u235.pdf")
plt.close()
plot_pfns(cf252, save=True, outfile="./pfns_cf252.png")
plt.close()
plot_pfns(u235, save=True, outfile="./pfns_u235.png")
plt.close()


plot_pnu(data_sets, save=True, rel=True, outfile="./pnu.pdf")
plt.close()
plot_fm(data_sets, save=True, rel=True, outfile="./fm.pdf")
plt.close()
plot_pnu(data_sets, save=True, rel=True, outfile="./pnu.png")
plt.close()
plot_fm(data_sets, save=True, rel=True, outfile="./fm.png")
plt.close()

plot_nu_corr_dev(cf252, save=True, outfile="./plot_nu_corr_dev_cf252.pdf")
plt.close()
plot_nu_corr_dev(u235, save=True, outfile="./plot_nu_corr_dev_u235.pdf")
plt.close()
plot_nu_corr_dev(cf252, save=True, outfile="./plot_nu_corr_dev_cf252.png")
plt.close()
plot_nu_corr_dev(u235, save=True, outfile="./plot_nu_corr_dev_u235.png")
plt.close()
