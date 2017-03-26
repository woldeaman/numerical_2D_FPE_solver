import matplotlib.pyplot as plt
import numpy as np

# plotting format for plots of minimal error for each transition layer distance
def plotMinError(distance, Error, ESTD, save=False,
                 path='/Users/AmanuelWK/Desktop/'):
    plt.figure(1)
    plt.gca().set_xlim([0, np.max(distance)])
    plt.errorbar(distance, Error, yerr=ESTD)
    plt.xlabel('Transition Layer Thickness d [µm]')
    plt.ylabel('Minimal Error [$\pm$ µM]')
    if save:
        plt.savefig(path+'minError.pdf')
    else:
        plt.show()

# plotting format for D and F in the same figure
def plotDF(xx, D, F, save=False, path='/Users/AmanuelWK/Desktop/'):
    deltaX = xx[1] - xx[0]
    # to take into account extra bin change xx vector
    xx = np.concatenate((deltaX*np.ones(1), xx))
    plt.figure(2)
    plt.gca().set_xlim(left=xx[0])
    plt.gca().set_xlim(right=xx[-1])
    # plotting F
    plt.plot(xx, F, 'b-')
    plt.ylabel('Free Energy [k$_{B}$T]', color='b')
    plt.xlabel('Distance [µm]')
    plt.tick_params('y', colors='b')
    # plotting D
    plt.twinx()
    plt.plot(xx, D, 'r-')
    # Make the y-axis label, ticks and tick labels match the line
    plt.gca().set_xlim(left=xx[0])
    plt.gca().set_xlim(right=xx[-1])
    plt.ylabel('Diffusivity [µm$^2$/s]', color='r')
    plt.tick_params('y', colors='r')
    plt.xlabel('Distance [µm]')
    if save:
        plt.savefig(path+'bestDF.pdf')
    else:
        plt.show()

# for printing analytical solution
def plotCon(xx, cc, ccRes, tt, TransIndex, layerD=None, save=False,
            path='/Users/AmanuelWK/Desktop/'):
    plt.figure(3)
    deltaX = xx[1] - xx[0]
    M = cc[0, :].size  # number of profiles
    if layerD is None:
        layerD = deltaX
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

    colors = ['r', 'm', 'c', 'b']
    for j in range(M):
        plt.gca().set_xlim(left=-deltaX)
        plt.gca().set_xlim(right=xx[-1])
        plt.xlabel('Distance [µm]')
        plt.ylabel('Concentration [µM]')
        # printing analytical solution
        # plt.plot(xxAna, ccAna[:, indexTime[j]], 'k-.')
        l1, = plt.plot(xx, cc[:, j], '--', color=colors[j],
                       label=str(int(tt[j]/60))+'m Experiment')
        # plot computed only for t > 0, otherwise not computed
        l1s.append([l1])
        if j > 0:
            # concatenated to include constanc c0 boundary condition
            l2, = plt.plot(np.concatenate((-deltaX*np.ones(1), xx)),
                           np.concatenate((4*np.ones(1), ccRes[:, j])),
                           '-', color=colors[j],
                           label=str(int(tt[j]/60))+'m Numerical')
            l2s.append([l2])
    # plotting two legends, for color and linestyle
    legend1 = plt.legend([l1, l2], ["Experiment", "Numerical"], loc=1)
    plt.legend([l[0] for l in l1s], ["%d min" % (tt[i]/60)
                                     for i in range(tt.size)], loc=4)
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(figPath+'bestProfiles_d=%.2f.pdf' % minD)
    else:
        plt.show()
