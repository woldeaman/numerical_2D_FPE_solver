import matplotlib.pyplot as plt
import numpy as np
import inputOutput as io
import argparse as ap
import os
# for debugging
# import sys

'''
This script plots the analyzed data from the runs
'''


def main():
    # parsing command line inputs and storing in variable
    parser = ap.ArgumentParser()
    parser.add_argument('path', help='define the relative path to '
                        'folder containing data files for plotting')
    parser.add_argument('name', help='define the name of the experimental'
                        'concentration profiles')
    parser.add_argument('-s', '--save', action='store_true', help='saves '
                        'plots and data to files to desktop')
    args = parser.parse_args()
    path = args.path
    name = args.name
    # folder to save figures in
    # path for mac
    # figPath = '/Users/AmanuelWK/Desktop/%s/Figures/' % name
    # path for linux
    figPath = '/home/amanuelwk/Desktop/%s/Figures/' % name
    if not os.path.exists(figPath):
        os.makedirs(figPath)

    # paramters for experimental setup
    dt = 300  # time difference between c-profiles in s
    boundary = 100  # position of boundary between mucus and buffer in µm

    # loading analyzed data
    eData = io.readData(path+'Error.csv')
    DF = io.readData(path+'DF.csv')
    cData = io.readData(path+'concentrationExpRes.csv')
    # restData = np.load(path+'data.npy')
    # saving to seperate arrays
    distanceMuM, EMin, ESTD = eData.T
    D, F = DF.T
    xx, cc, ccRes = cData[:, 0], cData[:, 1:5], cData[:, 6:]
    M = cc[0, :].size  # number of profiles
    # K = distanceMuM.size  # number of different transition distances
    minD = distanceMuM[np.argmin(EMin)]  # optimal transition layer thickness
    deltaX = xx[1] - xx[0]  # discretization width
    TransIndex = np.argwhere(abs(xx - boundary) ==  # bin clostest to boundary
                             np.min(abs(xx - boundary)))[0, 0].astype(int)
    tt = np.arange(M)*dt  # vectors contains time for profile i

    # plotting figures from here on
    # starting with Error over transition layer thickness
    plt.figure(1)
    plt.gca().set_xlim([0, np.max(distanceMuM)])
    '''change plotting range here for newer simulations'''
    plt.errorbar(distanceMuM[1:], EMin[1:], yerr=ESTD[1:])
    plt.xlabel('Transition Layer Thickness d [µm]')
    plt.ylabel('Minimal Error [$\pm$ µM]')
    if args.save:
        plt.savefig(figPath+'minError.pdf')
    else:
        plt.show()

    # for plotting best D and F in the same figure
    plt.figure(2)
    plt.gca().set_xlim(left=-deltaX)
    plt.gca().set_xlim(right=xx[-1])
    # plotting F
    plt.plot(np.concatenate((-np.ones(1)*deltaX, xx)),
             F, 'b-')
    plt.ylabel('Free Energy [k$_{B}$T]', color='b')
    plt.xlabel('Distance [µm]')
    plt.tick_params('y', colors='b')
    # plotting D
    plt.twinx()
    plt.plot(np.concatenate((-np.ones(1)*deltaX, xx)),
             D, 'r-')
    # Make the y-axis label, ticks and tick labels match the line
    plt.gca().set_xlim(left=-deltaX)
    plt.gca().set_xlim(right=xx[-1])
    plt.ylabel('Diffusivity [µm$^2$/s]', color='r')
    plt.tick_params('y', colors='r')
    plt.xlabel('Distance [µm]')
    if args.save:
        plt.savefig(figPath+'bestDF_d=%.2f.pdf' % minD)
    else:
        plt.show()

    # plotting concentration profiles for best run
    plt.figure(3)
    # for printing analytical solution
    # ccAna = np.load('cProfiles.npy')  # change here
    # xxAna = np.linspace(0, 590.82, num=100)  # for positively charged peptide
    # xxAna = np.linspace(0, 617.91, num=100)  # for negatively charged peptide
    # indexTime = np.array([10, 500, 1000, -1])  # index for which t = 5,10,15m
    # plt.plot(xxAna, ccAna[:, indexTime[1]], 'k-.', label='Analytical')
    # plotting shaded area in transition layer and textboxes
    # conditional positions, based on distance vector
    xLeft = xx[TransIndex]-minD/2-deltaX*1.5
    xRight = xx[TransIndex]+minD/2-deltaX*1.5
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
                           np.concatenate((4*np.ones(1), ccRes[:, j-1])),
                           '-', color=colors[j],
                           label=str(int(tt[j]/60))+'m Numerical')
            l2s.append([l2])
    # plotting two legends, for color and linestyle
    legend1 = plt.legend([l1, l2], ["Experiment", "Numerical"], loc=1)
    plt.legend([l[0] for l in l1s], ["%d min" % (tt[i]/60)
                                     for i in range(tt.size)], loc=4)
    plt.gca().add_artist(legend1)

    if args.save:
        plt.savefig(figPath+'bestProfiles_d=%.2f.pdf' % minD)
    else:
        plt.show()

    '''
    add option to plot other data from restData array
    '''

if __name__ == "__main__":
    main()
