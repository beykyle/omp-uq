import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'Helvetica','serif'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['axes.labelsize'] = 18.
matplotlib.rcParams['xtick.labelsize'] = 18.
matplotlib.rcParams['ytick.labelsize'] = 18.
matplotlib.rcParams['lines.linewidth'] = 2.
matplotlib.rcParams['xtick.major.pad'] = '10'
matplotlib.rcParams['ytick.major.pad'] = '10'
matplotlib.rcParams['image.cmap'] = 'BuPu'

from cgmf_post import DataSetUQUncorr, plot_nubar, plot_pnu, plot_fm, plot_pfns

#cf_252_datapath = Path(sys.argv[1])
#u_235_datapath = Path(sys/argv[2])
cf_252_datapath = Path("/home/beykyle/db/projects/OM/comp_res/cf252")
u_235_datapath = Path("/home/beykyle/db/projects/OM/comp_res/u235")

cf252 = DataSetUQUncorr(cf_252_datapath, 415, 192000, "252Cf (sf)", corr=True)
u235  = DataSetUQUncorr(u_235_datapath, 415, 192000, "235U (nth,f)", corr=True)

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
