import sys
import numpy as np
import matplotlib
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

from cgmf_uq import DataSetUQUncorr, plot_nubar, plot_pnu, plot_fm, plot_pfns, plot_nua

#cf_252_datapath = Path(sys.argv[1])
#u_235_datapath = Path(sys/argv[2])
cf_kd_datapath = Path("/home/beykyle/db/aps/run2_results/post_kduq_cf252")
u_kd_datapath = Path("/home/beykyle/db/aps/run2_results/post_kduq_u235")
cf_ch_datapath = Path("/home/beykyle/db/aps/run2_results/post_chuq_cf252")
u_ch_datapath = Path("/home/beykyle/db/aps/run2_results/post_chuq_u235")
cf_wlh_datapath = Path("/home/beykyle/db/aps/run2_results/post_wlh_cf252")
u_wlh_datapath = Path("/home/beykyle/db/aps/run2_results/post_wlh_u235")

kdcf = DataSetUQUncorr(cf_kd_datapath, 300, 1500, "CGMF+KDUQ", corr=False, default_sample_name="default")
kdu = DataSetUQUncorr(u_kd_datapath, 300, 1500, "CGMF+KDUQ", corr=False, default_sample_name="default")

chcf = DataSetUQUncorr(cf_ch_datapath, 300, 1500, "CGMF+CHUQ", corr=False, default_sample_name="default")
chu = DataSetUQUncorr(u_ch_datapath, 300, 1500, "CGMF+CHUQ", corr=False, default_sample_name="default")

wlhcf = DataSetUQUncorr(cf_wlh_datapath, 300, 1500, "CGMF+WLH", corr=False, default_sample_name="default")
wlhu = DataSetUQUncorr(u_wlh_datapath, 300, 1500, "CGMF+WLH", corr=False, default_sample_name="default")

cf = [kdcf, chcf, wlhcf]
u = [kdu, chu, wlhu]

# plot nubar distributions
plot_nubar(cf, save=True, outfile="./pnubar_cf252.png", rel=False, endf=3.7590)
plt.close()
plot_nubar(u, save=True, outfile="./pnubar_u235.png", rel=False, endf=2.42)
plt.close()

plot_pfns(cf, save=True, outfile="./pfns_cf252.png")
plt.close()
plot_pfns(u, save=True, outfile="./pfns_u235.png")
plt.close()


plot_pnu(cf, save=True, rel=True, outfile="./pnu_cf252.png")
plt.close()
plot_pnu(u, save=True, rel=True, outfile="./pnu_u235.png")
plt.close()

#TODO rel diff b/t experiment
plot_nua(cf, save=True,  outfile="./nua_cf252.png")
plt.close()
plot_nua(u, save=True, outfile="./nua_u235.png")
plt.close()
