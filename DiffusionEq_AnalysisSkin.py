import matplotlib.pyplot as plt
import numpy as np
import inputOutput as io
import FPModel as fp
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.special as sp
# for debugging
import sys

save = False


def main():
    # printing results
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/GitHub/FokkerPlanckModeling/')
    # path2 = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            #  'Skin/Results/ExperimentalData/')
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/GitHub/FokkerPlanckModeling/')
    path2 = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
             'Skin/Results/ExperimentalData/')

    # for plotting gathering experimental data
    cc = np.array([np.concatenate((np.ones(10)*0.0025, np.zeros(90))),
                   io.readData(path2+'p10min.txt')[:73],
                   io.readData(path2+'p100min.txt')[:80],
                   io.readData(path2+'p1000min.txt')[:80]]).T
    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    dim = max([cc[i].size for i in range(1, cc.size)])  # number of bins
    M = tt.size
    segments = np.concatenate((np.ones(10)*0, np.arange(1, dim+1),
                               np.ones(10)*(dim+1))).astype(int)
    deltaX = np.array([61, 1, 3076.38])
    deltaXX = np.concatenate((np.ones(6)*deltaX[0],
                              np.ones(1)*(deltaX[0]+deltaX[1])/2,
                              np.ones(dim+6)*deltaX[1],
                              np.ones(1)*(deltaX[1]+deltaX[2])/2,
                              np.ones(7)*deltaX[2]))
    xx = np.concatenate(((np.arange(7)*deltaX[0])-400, np.ones(1)*((deltaX[0]+deltaX[1])/2)+6*deltaX[0]-400,
                        np.arange(-2, dim+3), np.ones(1)*((deltaX[1]+deltaX[2])/2)+(dim+2),
                        np.arange(1, 7)*deltaX[2]+(dim+2)+((deltaX[1]+deltaX[2])/2)))
    '''figure out how to plot this xx vector with variable bins'''
    xx = np.arange(100)
    # number of combinations for different c-profiles used for calculation
    # of the residual vector
    n = (int(sp.binom(M, 2)))

    # loading result and extracting data for top N runs
    N = 1
    results = np.load(path+'result.npy')
    Error = np.array([np.sqrt(results[i].cost*2/(dim*n))
                      for i in range(results.size)])
    indices = np.argsort(Error)[:N]  # sorting according to error

    # gathering F and D
    DRes = np.array([results[indices[i]].x[:(dim+2)] for i in range(N)])
    FRes = np.array([np.concatenate((np.zeros(1),
                                     results[indices[i]].x[(dim+2):]))
                     for i in range(N)])

    DF = np.array([fp.computeDF(DRes[i, :], FRes[i, :], shape=segments)
                   for i in range(N)])

    D = DF[:, 0, :]
    F = DF[:, 1, :]

    W = np.array([fp.WMatrixVar(D[i, :], F[i, :], dim, deltaXX)
                  for i in range(N)])

    ccRes = np.array([[fp.calcC(cc[0], tt[j], W=W[i, :, :])
                       for j in range(M)] for i in range(N)])

    # plotting results
    cm = plt.get_cmap('hsv')
    cNorm = colors.Normalize(vmin=0, vmax=M)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for i in range(N):
        plt.figure(i)
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.plot(xx, D[i, :])
        plt.xlabel('Distance [$\mu m$]')
        plt.ylabel('Diffusivity [$\mu m^2/s$]')
        plt.title('Error=%s, D_%s' % (Error[indices[i]], str(indices[i])))
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_D.pdf' % str(indices[i]))
        else:
            plt.show()

        plt.figure(i+1)
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.plot(xx, F[i, :])
        plt.xlabel('Distance [$\mu m$]')
        plt.ylabel('Free Energy [$k_{B}T$]')
        plt.title('Error=%s, F_%s' % (Error[indices[i]], str(indices[i])))
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_F.pdf' % str(indices[i]))
        else:
            plt.show()

        plt.figure(i+2)
        colorVal = scalarMap.to_rgba(0)
        plt.plot(xx, cc[0], '--', color=colorVal,
                 label=str(int(tt[0]/60))+'m Experiment')
        plt.plot(xx, ccRes[i, 0, :], '-', color=colorVal,
                 label=str(int(tt[0]/60))+'m Computed')
        for j in range(1, M):
            colorVal = scalarMap.to_rgba(j)
            # plt.gca().set_xlim(left=xx[0])
            # plt.gca().set_xlim(right=xx[-1])
            plt.xlabel('Distance [$\mu m$]')
            plt.ylabel('Concentration [$\mu M$]')
            plt.plot(xx[11:cc[j].size+11], cc[j], '--', color=colorVal,
                     label=str(int(tt[j]/60))+'m Experiment')
            plt.plot(xx, ccRes[i, j, :], '-', color=colorVal,
                     label=str(int(tt[j]/60))+'m Computed')
        plt.title('C-Profiles from run #%s' % (str(indices[i])))
        plt.legend()
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_profiles.pdf'
                        % str(indices[i]))
        else:
            plt.show()

if __name__ == "__main__":
    main()
