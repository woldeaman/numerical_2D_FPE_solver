import matplotlib.pyplot as plt
import numpy as np
import inputOutput as io
import matplotlib.cm as cmx
import matplotlib.colors as colors
import argparse as ap
# for debugging
# import sys

'''
This script plots the analyzed data from the runs
'''


def main():
    # parsing command line inputs
    parser = ap.ArgumentParser()
    parser.add_argument('path', help='define the relative path to '
                        'folder containing data files for plotting')
    parser.add_argument('-s', '--save', action='store_true', help='saves '
                        'plots and data to files to desktop')
    args = parser.parse_args()
    path = args.path

    # loading analyzed data
    eData = io.readData(path+'Error.csv')
    DF = io.readData(path+'DF.csv')
    cData = io.readData(path+'concentrationExpRes.csv')
    # saving to seperate arrays
    distanceMuM, EMin, ESTD = eData.T
    D, F = DF.T
    xx, cc, ccRes = cData[:, 0], cData[:, 1:4], cData[:, 5:]

    # plotting figures
    plt.figure(-1)
    plt.gca().set_xlim([0, np.max(distanceMuM)])
    '''change plotting range here for newer simulations'''
    plt.errorbar(distanceMuM[1:], EMin[1:], yerr=ESTD[1:])
    plt.xlabel('Transition Layer Thickness d [µm]')
    plt.ylabel('Minimal Error [$\pm$ µM]')
    plt.title('Layer Thickness Profile %s'
              % name[:-4])
    if args.save:
        plt.savefig('/Users/AmanuelWK/Desktop/figures/minError_%s.pdf'
                    % name[:-4])
    else:
        plt.show()

    plt.figure(-2)
    plt.gca().set_xlim([0, np.max(distanceMuM)])
    '''change plotting range here too for newer simulations'''
    plt.errorbar(distanceMuM[1:], EMean[1:], yerr=ESTD[1:])
    plt.xlabel('Transition Layer Thickness d [µm]')
    plt.ylabel('Mean Error [$\pm$ µM]')
    plt.title('Layer Thickness Profile %s'
              % name[:-4])
    if args.save:
        plt.savefig('/Users/AmanuelWK/Desktop/figures/meanError_%s.pdf'
                    % name[:-4])
    else:
        plt.show()


    # plotting results
    cm = plt.get_cmap('hsv')
    cNorm = colors.Normalize(vmin=1, vmax=M)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    L = 1  # top L runs to plot
    # for printing analytical solution
    # ccAna = np.load('cProfiles.npy')  # change here
    # xxAna = np.linspace(0, 590.82, num=100)  # for positively charged peptide
    # xxAna = np.linspace(0, 617.91, num=100)  # for negatively charged peptide
    # indexTime = np.array([10, 500, 1000, -1])  # index for which t = 5,10,15m
    '''change range for newer simulations'''
    for k in range(1, K):  # change here K = 1 for printing analytical solution
        # k = -1  # for plotting only profiles for largest transition d
        for i in range(L):
            # # for plotting D and F in the same figure
            plt.figure(i+k)
            plt.gca().set_xlim(left=-deltaX)
            plt.gca().set_xlim(right=xx[-1])
            # plotting F
            plt.plot(np.concatenate((-np.ones(1)*deltaX, xx)),
                     F[k, i, :], 'b-')
            plt.ylabel('Free Energy [k$_{B}$T]', color='b')
            plt.tick_params('y', colors='b')
            # plotting D
            plt.twinx()
            plt.plot(np.concatenate((-np.ones(1)*deltaX, xx)),
                     D[k, i, :], 'r-')
            plt.xlabel('Distance [µm]')
            # Make the y-axis label, ticks and tick labels match the line
            plt.gca().set_xlim(left=-deltaX)
            plt.gca().set_xlim(right=xx[-1])
            plt.ylabel('Diffusivity [µm$^2$/s]', color='r')
            plt.tick_params('y', colors='r')
            if args.save:
                plt.savefig('/Users/AmanuelWK/Desktop/figures/'
                            'd=%.2f_E=%.2f_FD.pdf'
                            % (distanceMuM[k], Error[k, indices[k, i]]))
            else:
                plt.show()

            plt.figure(i+k+2*K)
            # for printing analytical solution
            # plt.plot(xxAna, ccAna[:, indexTime[1]],
            #          'k-.', label='Analytical')

            # plotting shaded area in transition layer and textboxes
            # conditional positions, based on distance vector
            xLeft = xx[TransIndex]-distanceMuM[k]/2-deltaX*1.5
            xRight = xx[TransIndex]+distanceMuM[k]/2-deltaX*1.5
            yMax = np.max(np.concatenate((cc.T, ccRes[k, i, :, :])))
            # plotting shaded region
            plt.axvspan(xLeft, xRight, color='r', lw=None, alpha=0.25)
            plt.figtext(xx[TransIndex]/np.max(xx), 0.91, 'transition layer')
            plt.text(x=xLeft-30, y=yMax, s='$D_{sol}$', va='top')
            plt.text(x=xRight+10, y=yMax, s='$D_{muc}, F_{muc}$', va='top')

            # plotting concentration profiles
            for j in range(1, M):
                colorVal = scalarMap.to_rgba(j)
                plt.gca().set_xlim(left=-deltaX)
                plt.gca().set_xlim(right=xx[-1])
                plt.xlabel('Distance [µm]')
                plt.ylabel('Concentration [µM]')
                # printing analytical solution
                # plt.plot(xxAna, ccAna[:, indexTime[j]], 'k-.')
                plt.plot(np.concatenate((-deltaX*np.ones(1), xx)),
                         np.concatenate((np.ones(1)*4, cc[:, j])), '--',
                         color=colorVal,
                         label=str(int(tt[j]/60))+'m Experiment')
                plt.plot(np.concatenate((-deltaX*np.ones(1), xx)),
                         np.concatenate((4*np.ones(1), ccRes[k, i, j, :])),
                         '-', color=colorVal,
                         label=str(int(tt[j]/60))+'m Numerical')
            plt.legend()
            if args.save:
                plt.savefig('/Users/AmanuelWK/Desktop/figures/'
                            'd=%.2f_E=%.2f_profiles.pdf'
                            % (distanceMuM[k], Error[k, indices[k, i]]))
            else:
                plt.show()


if __name__ == "__main__":
    main()
