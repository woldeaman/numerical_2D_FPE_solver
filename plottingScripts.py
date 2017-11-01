import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import sys


# plotting format for plots of minimal error for each transition layer distance
def plotMinError(distance, Error, ESTD, save=False,
                 path=None):
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    plt.figure()
    plt.gca().set_xlim([0, np.max(distance)])
    plt.errorbar(distance, Error, yerr=[np.zeros(ESTD.size), ESTD])
    plt.xlabel('Transition Layer Thickness d [µm]')
    plt.ylabel('Minimal Error [$\pm$ µM]')
    if save:
        plt.savefig(path+'minError.pdf', bbox_inches='tight')
    else:
        plt.show()


# plotting format for D and F in the same figure
def plotDF(xx, D, F, D_STD=None, F_STD=None, save=False, style='.',
           scale='linear', name='avgDF', path=None, ):
    """
    Plots D and F profiles
    """
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    plt.figure()
    plt.gca().set_xlim(left=xx[0])
    plt.gca().set_xlim(right=xx[-1])
    # plotting F
    if F_STD is None:
        plt.plot(xx, F, style+'b')
    else:
        plt.errorbar(xx, F, yerr=F_STD, fmt=style+'b')
    plt.ylabel('Free Energy [k$_{B}$T]', color='b')
    plt.xlabel('Distance [µm]')
    plt.tick_params('y', colors='b')
    # plotting D
    plt.twinx()
    if D_STD is None:
        plt.plot(xx, D, style+'r')
    else:
        plt.errorbar(xx, D, yerr=D_STD, fmt=style+'r')
    # Make the y-axis label, ticks and tick labels match the line
    plt.gca().set_xlim(left=xx[0])
    plt.gca().set_xlim(right=xx[-1])
    plt.ylabel('Diffusivity [µm$^2$/s]', color='r')
    plt.yscale(scale)
    plt.tick_params('y', colors='r')
    plt.xlabel('Distance [µm]')
    if save:
        plt.savefig(path+'%s.pdf' % name, bbox_inches='tight')
    else:
        plt.show()


# for plotting concentration profiles
def plotCon(xx, cc, ccRes, tt, locs=[1, 3], colorbar=False, styles=['--', '-'],
            save=False, path=None):
    '''
    Plots analyzed concentration profiles.
    'locs' - determines location for the two legends.
    'colorbar' - plots colorbar instead of legends.
    'styles' - defines styles for experimental and numerical profiles.
    '''
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    M = cc[0, :].size  # number of profiles

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []
    # mapping profiles to colormap
    lines = np.linspace(0, 1, M)
    colors = [cm.jet(x) for x in lines]

    plt.figure(0)
    for j in range(M):
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.xlabel('Distance [µm]')
        plt.ylabel('Concentration [µM]')
        l1, = plt.plot(xx, cc[:, j], '--', color=colors[j])
        l1s.append([l1])
        if j > 0:
            # plot t=0 profile only for experiment
            # because numerical profiles are computed from this one
            l2, = plt.plot(xx, ccRes[:, j], '-', color=colors[j])
            l2s.append([l2])
    # plotting two legends, for color and linestyle
    # TODO: add plotting of colorbar instead of legend
    legend1 = plt.legend([l1, l2], ["Experiment", "Numerical"], loc=locs[0])
    plt.legend([l[0] for l in l1s], ["%.2f min" % (tt[i]/60) if tt[i] % 60 != 0
                                     else "%i min" % int(tt[i]/60)
                                     for i in range(tt.size)], loc=locs[1])
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path+'profiles.pdf', bbox_inches='tight')
    else:
        plt.show()


# for printing analytical solution and transition layer thicknesses
def plotConTrans(xx, cc, ccRes, c0, tt, TransIndex, layerD, save=False,
                 path=None):

    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    plt.figure()
    deltaX = abs(xx[1] - xx[0])
    M = cc[0, :].size  # number of profiles

    # ccAna = np.load('cProfiles.npy')  # change here
    # xxAna = np.linspace(0, 590.82, num=100)  # for positively charged peptide
    # xxAna = np.linspace(0, 617.91, num=100)  # for negatively charged peptide
    # indexTime = np.array([10, 500, 1000, -1])  # index for which t = 5,10,15m
    # plt.plot(xxAna, ccAna[:, indexTime[1]], 'k-.', label='Analytical')
    # plotting shaded area in transition layer and textboxes
    # conditional positions, based on distance vector
    xLeft = xx[TransIndex]-layerD/2-deltaX*1.5
    xRight = xx[TransIndex]+layerD/2-deltaX*1.5
    yMax = np.max(np.concatenate((cc, ccRes), axis=1))
    # plotting shaded region
    plt.axvspan(xLeft, xRight, color='r', lw=None, alpha=0.25)
    plt.figtext(xx[TransIndex]/np.max(xx), 0.91, 'transition layer')
    plt.text(x=xLeft-30, y=yMax, s='$D_{sol}$', va='top')
    plt.text(x=xRight+10, y=yMax, s='$D_{muc}, F_{muc}$', va='top')

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []

    colors = ['r', 'm', 'c', 'b', 'y', 'k', 'g']
    for j in range(M):
        plt.gca().set_xlim(left=-deltaX)
        plt.gca().set_xlim(right=xx[-1])
        plt.xlabel('Distance [µm]')
        plt.ylabel('Concentration [µM]')
        # printing analytical solution
        # plt.plot(xxAna, ccAna[:, indexTime[j]], 'k-.')
        l1, = plt.plot(xx, cc[:, j], '--', color=colors[j],
                       label='%.2f m Experiment' % float(tt[j]/60))
        # plot computed only for t > 0, otherwise not computed
        l1s.append([l1])
        # concatenated to include constanc c0 boundary condition
        l2, = plt.plot(np.concatenate((-deltaX*np.ones(1), xx)),
                       np.concatenate((c0*np.ones(1), ccRes[:, j])),
                       '-', color=colors[j],
                       label=str(int(tt[j]/60))+'m Numerical')
        l2s.append([l2])
    # plotting two legends, for color and linestyle
    legend1 = plt.legend([l1, l2], ["Experiment", "Numerical"], loc=1)
    plt.legend([l[0] for l in l1s], ["%.2f min" % (tt[i]/60) if tt[i] % 60 != 0
                                     else "%i min" % int(tt[i]/60)
                                     for i in range(tt.size)], loc=4)
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path+'profiles.pdf', bbox_inches='tight')
    else:
        plt.show()


# for printing c-profiles
def plotConSkin(xx, cc, ccRes, tt, locs=[1, 2], save=False, path=None,
                deltaXX=None):

    M = len(cc)  # number of profiles
    N = ccRes[0, :].size  # number of bins
    if deltaXX is None:
        deltaXX = np.ones(N+1)
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []
    lines = np.linspace(0, 1, M)
    colors = [cm.jet(x) for x in lines]

    plt.figure()
    for j in range(M):
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.xlabel('Bins')
        plt.ylabel('Concentration [µM]')
        if j == 0:
            l1, = plt.plot(xx, cc[j], '--', color=colors[j])
        else:
            l1, = plt.plot(xx[6:-3], cc[j], '--', color=colors[j])

        # plot computed only for t > 0, otherwise not computed
        if j > 0:
            l1s.append([l1])
            # concatenated to include constanc c0 boundary condition
            l2, = plt.plot(xx, ccRes[:, j], '-', color=colors[j])
            l2s.append([l2])
    # plotting two legends, for color and linestyle
    legend1 = plt.legend([l1, l2], ["Experiment", "Numerical"], loc=locs[0])
    plt.legend([l[0] for l in l1s], ["%d min" % (tt[i]/60)
                                     for i in range(tt.size)], loc=3)
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path+'profiles.pdf')
    else:
        plt.show()
