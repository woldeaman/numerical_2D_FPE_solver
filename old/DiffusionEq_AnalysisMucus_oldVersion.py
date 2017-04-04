import matplotlib.pyplot as plt
import numpy as np
import inputOutput as io
import FPModel as fp
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.special as sp
import argparse as ap
import xlsxwriter as xl
# for debugging
import sys

'''
rewrite this script to give you data in txt files and write and write
a second file just for plotting! I hope this makes things easier...
'''


def main():
    # parsing command line inputs
    parser = ap.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true', help='saves '
                        'plots and data to files on desktop')
    parser.add_argument('path', help='define the relative path to '
                        'data for analysis')
    parser.add_argument('name', help='defines name of concentration profile'
                        ' data to be plotted in conjunction with analysis')
    args = parser.parse_args()
    # gathering path to data and setting verbosity and conservation mode
    path = args.path
    name = args.name+'.csv'
    if args.save:
        save = True
    else:
        save = False

    # for plotting gathering experimental data
    Cdata = io.readData(path+name, sep=';')
    xx = Cdata[:, 0]  # first line in document is x-position
    cc = np.array([Cdata[:, 1], Cdata[:, 31], Cdata[:, 61], Cdata[:, 91]]).T
    # change number of profiles according to analysis type
    # pre processing of profiles as in optimization script
    # --> smoothing and setting negative concentration to zero
    xx, cc = io.preProcessing(xx, cc)
    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM
    dim = cc[:, 0].size  # number of bins, chosen during pre processing
    M = cc[0, :].size  # number of profiles
    TransIndex = np.argwhere(abs(xx-100) ==
                             np.min(abs(xx - 100)))[0, 0].astype(int)
    # '''only needed for compatibality with older version'''
    # # TransIndex = 17  # for negative peptide
    # # TransIndex = 18  # for positive peptide
    # '''only needed for compatibality with older version'''

    # same conditions as for analysis need to be kept here
    segments = np.concatenate((np.ones(TransIndex+1)*0,
                               np.ones(dim-TransIndex)*1)).astype(int)
    '''change distances definition for newer simulations'''
    distances = np.arange(0, (2*TransIndex)+1, 2)
    n = int(sp.binom(M, 2))  # binomial because of counting profile differences

    # loading result and extracting data for top N runs
    N = 10  # number of best runs to analyze
    results = np.load(path+'result.npy')
    '''only needed for compatibality issues'''
    # results = results[:, :, 0]  # for compatibality with older version
    # for compatibality with buffer experiment
    # results = results.reshape((1, results.size))
    '''only needed for compatibality with older version'''

    K = results[:, 0].size  # number of different transition sizes
    I = results[0, :].size  # number of different initial conditions

    # gathering error data
    Error = np.array([[np.sqrt(results[k, i].cost*2/(dim*n)) for i in range(I)]
                      for k in range(K)])
    # times two, because of cost function definition which multiplies by 0.5
    # see scipy.optimize.least_squares definition
    indices = np.argsort(Error)[:, :N]  # sorting according to error

    # gathering F and D and subsequently computing corresponding profiles
    DRes = np.array([[results[k, indices[k, i]].x[:2] for i in range(N)]
                     for k in range(K)])
    FRes = np.array([[np.array([0, results[k, indices[k, i]].x[-1]])
                      for i in range(N)]
                     for k in range(K)])

    DF = np.array([[fp.computeDF(DRes[k, i, :], FRes[k, i, :],
                                 shape=segments, mode='transition',
                                 transiBin=TransIndex, dx=distances[k])
                    for i in range(N)] for k in range(K)])

    '''for buffer experiment'''
    # segments = np.zeros(dim+1)  # needed for buffer experiment conditions
    # DF = np.array([[fp.computeDF(DRes[k, i, :], FRes[k, i, :],
    #                              shape=segments, mode='segments')
    #                 for i in range(N)] for k in range(K)])
    # distanceMuM = deltaX*np.ones(1)
    '''for buffer experiment'''

    D = DF[:, :, 0, :]
    F = DF[:, :, 1, :]
    # results for D and F where D[k, i, :] gives D profile for
    # ith best result at transition distance k

    # computing W Matrices for each run
    W = np.array([[fp.WMatrix(D[k, i, :], F[k, i, :], bc='open1side',
                              deltaX=deltaX)[0] for i in range(N)]
                  for k in range(K)])

    W10 = np.array([[fp.WMatrix(D[k, i, :], F[k, i, :], bc='open1side',
                                deltaX=deltaX)[1] for i in range(N)]
                    for k in range(K)])

    # computing concentration profiles for simulations
    ccRes = np.array([[[fp.calcC(cc[:, 0], tt[j], W=W[k, i, :, :],
                                 bc='open1side', W10=W10[k, i], c0=c0)
                        for j in range(M)] for i in range(N)]
                      for k in range(K)])

    # plotting Error
    EMin = np.array([np.min(Error[k, :]) for k in range(K)])
    EMean = np.array([np.mean(Error[k, :]) for k in range(K)])
    ESTD = np.array([np.std(Error[k, :]) for k in range(K)])
    '''change to distanceMuM = (distances-1)*deltaX, for newer simulations'''
    distanceMuM = np.concatenate((deltaX*np.ones(1), (distances[1:]-1)*deltaX))

    plt.figure(-1)
    plt.gca().set_xlim([0, np.max(distanceMuM)])
    '''change plotting range here too for newer simulations'''
    plt.errorbar(distanceMuM[1:], EMin[1:], yerr=ESTD[1:])
    plt.xlabel('Transition Layer Thickness d [µm]')
    plt.ylabel('Minimal Error [$\pm$ µM]')
    plt.title('Layer Thickness Profile %s'
              % name[:-4])
    if save:
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
    if save:
        plt.savefig('/Users/AmanuelWK/Desktop/figures/meanError_%s.pdf'
                    % name[:-4])
    else:
        plt.show()

    if save:
        # saving values of F and D, computed from top 10% of runs for
        # transition layer thickness with minimal mean Error
        index = np.argmin(EMin)
        D1 = D[index, :int(I/10), 0]  # top 10% of runs for D_sol
        D2 = D[index, :int(I/10), -1]  # top 10% of runs for D_muc
        F2 = F[index, :int(I/10), -1]  # top 10% of runs for F_muc
        # saving data to excel spreadsheet
        workbook = xl.Workbook('/Users/AmanuelWK/Desktop/FD_%s.xlsx'
                               % name[:-4])
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': True})
        # writing headers
        worksheet.write('A1', 'D_sol [µm^2/s]', bold)
        worksheet.write('B1', 'D_muc [µm^2/s]', bold)
        worksheet.write('C1', 'F_muc [kT]', bold)
        worksheet.write('D1', 'layer d [µm]', bold)
        worksheet.write('E1', 'min E [+/- µM]', bold)
        # writing entries
        worksheet.write('A2', '%.2f +/- %.2f' % (D1[0], np.std(D1)))
        worksheet.write('B2', '%.2f +/- %.2f' % (D2[0], np.std(D2)))
        worksheet.write('C2', '%.2f +/- %.2f' % (F2[0], np.std(F2)))
        worksheet.write('D2', '%.2f' % distanceMuM[index])
        worksheet.write('E2', '%.2f' % np.min(EMin))
        # adjusting cell widths
        worksheet.set_column(0, 5, len('minError [+/- µM]'))
        workbook.close()

    # plotting results
    cm = plt.get_cmap('hsv')
    cNorm = colors.Normalize(vmin=1, vmax=M)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    L = 1  # top L runs to plot
    # for printing analytical solution
    ccAna = np.load('cProfiles.npy')  # change here
    xxAna = np.linspace(0, 590.82, num=100)  # for positively charged peptide
    xxAna = np.linspace(0, 617.91, num=100)  # for negatively charged peptide
    indexTime = np.array([10, 500, 1000, -1])  # index for which t = 5,10,15m
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
            if save:
                plt.savefig('/Users/AmanuelWK/Desktop/figures/'
                            'd=%.2f_E=%.2f_FD.pdf'
                            % (distanceMuM[k], Error[k, indices[k, i]]))
            else:
                plt.show()

            plt.figure(i+k+2*K)
            # for printing analytical solution
            plt.plot(xxAna, ccAna[:, indexTime[1]],
                     'k-.', label='Analytical')

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
            if save:
                plt.savefig('/Users/AmanuelWK/Desktop/figures/'
                            'd=%.2f_E=%.2f_profiles.pdf'
                            % (distanceMuM[k], Error[k, indices[k, i]]))
            else:
                plt.show()

if __name__ == "__main__":
    main()
