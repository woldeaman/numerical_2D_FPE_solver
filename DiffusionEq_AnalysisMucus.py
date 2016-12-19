import matplotlib.pyplot as plt
import numpy as np
import inputOutput as io
import FPModel as fp
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.special as sp
# for debugging
import sys
save = True


def main():
    # printing results
    # path for home
    # path = ('/home/amanuelwk/Desktop/negative/')
    # path2 = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            #  'Mucus/Results/ExperimentalData/Ch1_Positive.csv')
    # path for work
    path = ('/Users/AmanuelWK/Desktop/negative/')
    path2 = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
             'Mucus/Results/ExperimentalData/Ch3_Negative.csv')

    # for plotting gathering experimental data
    Cdata = io.readData(path2)
    xx = Cdata[:, 0]
    cc = np.array([Cdata[:, 1], Cdata[:, 31], Cdata[:, 61], Cdata[:, 91]]).T

    # pre processing of profiles as in optimization script
    xx, cc = io.preProcessing(xx, cc)
    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM
    dim = cc[:, 0].size  # number of bins
    M = cc[0, :].size  # number of profiles
    TransIndex = np.argwhere(abs(xx-100) ==
                             np.min(abs(xx - 100)))[0, 0].astype(int)
    '''for compatibality with older version'''
    TransIndex = 17
    # same conditions as for analysis need to be kept here
    segments = np.concatenate((np.ones(TransIndex)*0,
                               np.ones(dim+1-TransIndex)*1)).astype(int)
    distances = np.arange(0, (2*TransIndex)+1, 2)
    n = int(sp.binom(M, 2))

    # loading result and extracting data for top N runs
    N = 1
    results = np.load(path+'result.npy')
    results = results[:, :, 0]  # for compatibality with older version

    K = results[:, 0].size  # number of different transition sizes
    I = results[0, :].size  # number of different initial conditions

    # gathering error data
    Error = np.array([[np.sqrt(results[k, i].cost*2/(dim*n)) for i in range(I)]
                      for k in range(K)])
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

    D = DF[:, :, 0, :]
    F = DF[:, :, 1, :]

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
    ESTD = np.array([np.std(Error[k, :]) for k in range(K)])

    plt.figure(-1)
    plt.errorbar(distances, EMin, yerr=ESTD)
    plt.xlabel('Transition Distance [bins]')
    plt.ylabel('Minimal Error [$\pm$ µM]')
    if save:
        plt.savefig('/Users/AmanuelWK/Desktop/figures/error.pdf')
    else:
        plt.show()

    # plotting results
    cm = plt.get_cmap('hsv')
    cNorm = colors.Normalize(vmin=0, vmax=M)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for k in range(K):
        for i in range(N):
            plt.figure(i)
            plt.gca().set_xlim(left=xx[0])
            plt.gca().set_xlim(right=xx[-1])
            plt.plot(xx, D[k, i, 1:])
            plt.xlabel('Distance [$\mu m$]')
            plt.ylabel('Diffusivity [$\mu m^2/s$]')
            plt.title('E=%s, d=%s:\n D_%s'
                      % (Error[k, indices[k, i]], str(distances[k]),
                         str(indices[k, i])))
            if save:
                plt.savefig('/Users/AmanuelWK/Desktop/figures/d=%s_#%s_D.pdf'
                            % (str(distances[k]), str(indices[k, i])))
            else:
                plt.show()

            plt.figure(i+K)
            plt.gca().set_xlim(left=xx[0])
            plt.gca().set_xlim(right=xx[-1])
            plt.plot(xx, F[k, i, 1:])
            plt.xlabel('Distance [$\mu m$]')
            plt.ylabel('Free Energy [$k_{B}T$]')
            plt.title('E=%s, d=%s:\n F_%s'
                      % (Error[k, indices[k, i]], str(distances[k]),
                         str(indices[k, i])))
            if save:
                plt.savefig('/Users/AmanuelWK/Desktop/figures/d=%s_#%s_F.pdf'
                            % (str(distances[k]), str(indices[k, i])))
            else:
                plt.show()

            plt.figure(i+k+2*K)
            for j in range(M):
                colorVal = scalarMap.to_rgba(j)
                plt.gca().set_xlim(left=xx[0])
                plt.gca().set_xlim(right=xx[-1])
                plt.xlabel('Distance [$\mu m$]')
                plt.ylabel('Concentration [$\mu M$]')
                plt.plot(xx, cc[:, j], '--', color=colorVal,
                         label=str(int(tt[j]/60))+'m Experiment')
                plt.plot(xx, ccRes[k, i, j, :], '-', color=colorVal,
                         label=str(int(tt[j]/60))+'m Computed')
            plt.title('E=%s, d=%s:\n C-Profiles from run #%s'
                      % (Error[k, indices[k, i]], str(distances[k]),
                         str(indices[k, i])))
            plt.legend()
            if save:
                plt.savefig('/Users/AmanuelWK/Desktop/figures/'
                            'd=%s_#%s_profiles.pdf' % (str(distances[k]),
                                                       str(indices[k, i])))
            else:
                plt.show()

if __name__ == "__main__":
    main()
