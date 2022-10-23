from exp import PFNS_A, plotCompPFNS, plotSpecRatio, plotSpecRatio3D, plotSpecRatioColor, Spec, getMeanShiftErr, getHardnessAboveMaxwell
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

def mass_dep_plots(cf252_A_pfns_cgmf, cf252_A_pfns_gook, masses, masses_odd, masses_even):
    ebins = np.arange(0.4, 5., step=0.2)

    specs_cgmf = cf252_A_pfns_cgmf.getSpecs(masses)
    specs_gook = cf252_A_pfns_gook.getSpecs(masses)


    x,y,z, ax = plotSpecRatioColor(
            specs_cgmf, specs_gook, ebins, "CGMF", "GOOK", masses, r"$A $[u]", rel_diff=True)

#ax.set_zlim3d(-200,300)
    ax.get_yaxis().set_major_locator(MaxNLocator(4,integer=True))
#ax.plot_surface(x,y, np.zeros(z.shape), alpha=0.3, color='k')
    plt.savefig("PFNS_A_cgmf_vs_gook.pdf")
    plt.close()

    interesting_masses(cf252_A_pfns_cgmf, cf252_A_pfns_gook, ebins)

def plotIndividual(masses,cgmf, gooK):
    datasets = [cf252_A_pfns_cgmf_kd, cf252_A_pfns_gook]
    labels   = ["CGMF+KD", "Gook"]

    ratio=True
    if not ratio:
        for A in masses:
            fig, _, _, _ = plotCompPFNS(A, datasets, labels, maxwell_norm=False)
            plt.xlim([0,5])
            plt.ylim([0.5*1E-2, 1E0])
            #plt.ylim([0.5*1E-1, 0.21*1E1])
            #plt.yscale("linear")
            plt.tight_layout()
            plt.show()

    else:
        for A in masses:
            fig, _, _, _ = plotCompPFNS(A, datasets, labels, maxwell_norm=True)
            plt.xlim([0,5])
            plt.ylim([0.5*1E-1, 0.21*1E1])
            plt.yscale("linear")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # read in spectra
    cf252_A_pfns_gook = PFNS_A("/home/beykyle/umich/omp-uq/data/exfor/252_Cf_PFNS_A.npy")
    cf252_A_pfns_cgmf_kd  = PFNS_A("./cgmf_252cf_kddef_pfns_a.npy")

    #plotIndividual([89, 102, 125, 150],cf252_A_pfns_cgmf_kd, cf252_A_pfns_gook)

    masses = np.array(range(85, 165), dtype=int)
    mask = np.array( [ cf252_A_pfns_cgmf_kd.getSpecs([A])[0].norm() > 1000 for A in masses ] )
    masses = masses[np.where(mask)]

    ebins = np.arange(2., 6.0, step=0.2)
    cgmf_hardness, cgmf_err = getHardnessAboveMaxwell(masses, cf252_A_pfns_cgmf_kd, ebins)
    gook_hardness, gook_err = getHardnessAboveMaxwell(masses, cf252_A_pfns_gook, ebins)

    plt.errorbar(masses, gook_hardness,
            yerr=gook_err, xerr=4, linestyle="None", label="Gook")
    plt.errorbar(masses, cgmf_hardness,
            yerr=cgmf_err, linestyle="None", label="CGMF+KD", marker=".")
    plt.xlabel(r"$A$ [u]")
    plt.ylabel(r"$\int_{2 MeV}^{6 MeV} P(E|A) / M(E,kT) dE$ ")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hardness.pdf")





