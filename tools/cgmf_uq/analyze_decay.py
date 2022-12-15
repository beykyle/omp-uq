from CGMFtk import histories as fh
from matplotlib import pyplot as plt

import numpy as np
import sys

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
def scatter_hist(x, y, ax, ax_histx, ax_histy):

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.hist2d(x,y, bins=45)
    ax.set_xlim([0,25])
    ax.set_ylim([0,30])

    # now determine nice limits by hand:
    ax_histx.hist(x, histtype='step', linewidth=5, density=True, bins=45)
    ax_histy.hist(y, orientation='horizontal', histtype="step", linewidth=5, density=True, bins=45)

def full_plot(x,y,name):
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)


    scatter_hist(x, y, ax, ax_histx, ax_histy)

    ax.set_xlabel(r"$J_n$  [$ \hbar $]")
    ax.set_ylabel(r"$E_n$ [MeV]")
    plt.savefig(name)


def sort_by_n(J0, Estar, h):
    nu = h.getNu()
    at_least_one_n = np.where(nu >= 1, True, False)
    num_at_least_one_n = np.sum(at_least_one_n)

    J0    = J0[at_least_one_n]
    Estar = Estar[at_least_one_n]

    max_N = np.max(nu)
    En = np.ones((num_at_least_one_n, max_N ))
    Jn = np.ones((num_at_least_one_n, max_N ))

    for i in range(num_at_least_one_n):
        Jn[i,:] = J0[i]
        En[i,:] = Estar[i]

    deltaJ = h.getNeutronDeltaJ()
    deltaE = h.getNeutronEcm()
    deltaJ = deltaJ[at_least_one_n]
    deltaE = deltaE[at_least_one_n]

    for i, hist in enumerate(deltaJ):
        if len(hist) != 0:
            Jn[i][0] += hist[0]
            for n in range(1,len(hist)):
                Jn[i][n] = Jn[i][n-1] + hist[n]

    for i, hist in enumerate(deltaE):
        if len(hist) != 0:
            En[i][0] -= hist[0]
            for n in range(1,len(hist)):
                En[i][n] = En[i][n-1] - hist[n]

    full_plot(J0,Estar,"start.png")
    full_plot(Jn[:,0],En[:,0], "n1.png")
    full_plot(Jn[:,1],En[:,1], "n2.png")

if __name__ == "__main__":
    #h = fh.Histories(sys.argv[1], ang_mom_printed=True, nevents=2000)
    h = fh.Histories(sys.argv[1], ang_mom_printed=True)

    nu = h.getNu()
    at_least_one_n = np.where(nu >= 1, True, False)

    deltaJ = h.getNeutronDeltaJ()
    deltaE = h.getNeutronEcm()

    total_e_rem = np.array([sum(h) for h in deltaE])
    total_j_rem = np.array([sum(h) for h in deltaJ])


    #full_plot(total_j_rem, total_e_rem)

    # initial conditions
    J0    = h.getJ()
    Estar = h.getU()
    A     = h.getA()
    Z     = h.getA()

    sort_by_n(J0, Estar, h)

