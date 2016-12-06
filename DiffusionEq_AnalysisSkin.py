import matplotlib.pyplot as plt
import numpy as np
import inputOutput as io
import FPModel as fp
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.special as sp
# for debugging
# import sys

save = False


def main():
    # printing results
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            # 'Mucus/Results/ComputedData/segmented/Positive/')
    # path2 = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            #  'Mucus/Results/ExperimentalData/Ch1_Positive.csv')
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/GitHub/FokkerPlanckModeling/')
    path2 = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
             'Skin/Results/ExperimentalData/')

    # for plotting gathering experimental data
    cc = np.array([np.concatenate((np.ones(1), np.zeros(70))),
                   io.readData(path2+'10min.txt'),
                   io.readData(path2+'100min.txt'),
                   io.readData(path2+'1000min.txt')]).T
    dim = cc[:, 0].size  # number of bins
    M = cc[0, :].size  # number of profiles
    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    xx = np.arange(dim)
    # number of combinations for different c-profiles used for calculation
    # of the residual vector
    n = np.binom(int(sp.binom(M, 2)))

    # loading result and extracting data for top N runs
    N = 1
    results = np.load(path+'result.npy')
    Error = np.array([np.sqrt(results[i].cost*2/(dim*n))
                      for i in range(results.size)])
    indices = np.argsort(Error)[:N]  # sorting according to error

    # gathering F and D
    D = np.array([results[indices[i]].x[:dim] for i in range(N)])
    F = np.array([results[indices[i]].x[dim:] for i in range(N)])

    W = np.array([fp.WMatrix(D[i, :], F[i, :]) for i in range(N)])

    ccRes = np.array([[fp.calcC(cc[:, 0], tt[j], W=W[i, :, :])
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
        for j in range(M):
            colorVal = scalarMap.to_rgba(j)
            plt.gca().set_xlim(left=xx[0])
            plt.gca().set_xlim(right=xx[-1])
            plt.xlabel('Distance [$\mu m$]')
            plt.ylabel('Concentration [$\mu M$]')
            plt.plot(xx[1:], cc[1:, j], '--', color=colorVal,
                     label=str(int(tt[j]/60))+'m Experiment')
            plt.plot(xx[1:], ccRes[i, j, 1:], '-', color=colorVal,
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
