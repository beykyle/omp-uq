from exp import PFNS_A, plotCompPFNS, plotSpecRatio, plotSpecRatio3D, plotSpecRatioColor, Spec, hardness
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pylab import MaxNLocator
from matplotlib import cm


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

def interesting_masses( cf252_A_pfns_cgmf, cf252_A_pfns_gook, ebins):
    masses = np.array([113, 114, 135, 142], dtype=int)

    specs_cgmf = cf252_A_pfns_cgmf.getSpecs(masses)
    specs_gook = cf252_A_pfns_gook.getSpecs(masses)

    plotSpecRatio(specs_cgmf, specs_gook, ebins, "CGMF", "GOOK",
            [r"$A = {}$".format(A) for A in  masses],
            rel_diff=True)
    plt.tight_layout()
    plt.savefig("interesting_masses_rel_err_pfns_a.pdf")
    plt.close()

    for A in masses:
        plotCompPFNS(A, [ cf252_A_pfns_cgmf, cf252_A_pfns_gook ], ["CGMF", "GOOK"])

def lh3d():
    masses_h = np.array(range(131, 152), dtype=int)
    masses_l = np.array(range(96, 116), dtype=int)

    specs_cgmf_h = cf252_A_pfns_cgmf.getSpecs(masses_h)
    specs_cgmf_l = cf252_A_pfns_cgmf.getSpecs(masses_l)
    specs_gook_h = cf252_A_pfns_gook.getSpecs(masses_h)
    specs_gook_l = cf252_A_pfns_gook.getSpecs(masses_l)

    fig_h, ax_h = plotSpecRatio3D(
            specs_cgmf_h, specs_gook_h, ebins, "CGMF", "GOOK", masses_h, r"$A $[u]", rel_diff=True)

    ax_h.get_yaxis().set_major_locator(MaxNLocator(4,integer=True))
    plt.tight_layout()
    plt.show()

    fig_l, ax_l = plotSpecRatio3D(
            specs_cgmf_l, specs_gook_l, ebins, "CGMF", "GOOK", masses_l, r"$A $ [u]", rel_diff=True)
    ax_l.get_yaxis().set_major_locator(MaxNLocator(4,integer=True))
    plt.show()



cf252_A_pfns_gook = PFNS_A("/home/beykyle/umich/omp-uq/data/exfor/252_Cf_PFNS_A.npy")
cf252_A_pfns_cgmf  = PFNS_A("/home/beykyle/db/aps/gook/cgmf_252cf_kddef_pfns_a.npy")

ebins = np.arange(0.4, 5., step=0.4)
masses = np.array(range(92, 160), dtype=int)
masses_odd  = np.array(range(92, 160,2), dtype=int)
masses_even = np.array(range(93, 161,2), dtype=int)

specs_cgmf = cf252_A_pfns_cgmf.getSpecs(masses)
specs_gook = cf252_A_pfns_gook.getSpecs(masses)


x,y,z, ax = plotSpecRatioColor(
        specs_cgmf, specs_gook, ebins, "CGMF", "GOOK", masses, r"$A $[u]", rel_diff=True)

#ax.set_zlim3d(-200,300)
ax.get_yaxis().set_major_locator(MaxNLocator(4,integer=True))
#ax.plot_surface(x,y, np.zeros(z.shape), alpha=0.3, color='k')
plt.savefig("PFNS_A_cgmf_vs_gook.pdf")
plt.close()

# TODO add error bars
mean_ratio = np.zeros(masses.shape)
for i , cgmf in enumerate(specs_cgmf):
    gook = specs_gook[i]
    me = hardness(cgmf,gook, ebins, rel=True)
    mean_ratio[i] = me

mean_ratio_even = mean_ratio[ np.where( masses % 2 ==0 )]
mean_ratio_odd = mean_ratio[ np.where( masses % 2 != 0 )]

plt.xlabel(r"$A$ [u]")
plt.ylabel(r"$\frac{\bar{E}_{CGMF} - \bar{E}_{GOOK}}{\bar{E}_{GOOK}}$ [%]")
plt.plot(masses_even, mean_ratio_even, label=r"$A$ even")
plt.plot(masses_odd, mean_ratio_odd, label=r"$A$ odd")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("mean_E_A_cgmf_vs_gook.pdf")
plt.close()

interesting_masses(cf252_A_pfns_cgmf, cf252_A_pfns_gook, ebins)
