# analyzing results obtained from running optimization algorithm
import matplotlib.pyplot as plt
import numpy as np
import inputOutput as io
import FPModel as fp
import matplotlib.colors as colors
import scipy.interpolate as ip
import matplotlib.cm as cmx
import sys

save = True


def main():
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
    #         'Mucus/Results/ComputedData/segmented/Positive/')
    # path2 = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
    #          'Mucus/Results/ExperimentalData/Ch1_Positive.csv')
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            'Mucus/Results/ComputedData/segmented/Run1/')
    path2 = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
             'Mucus/Results/ExperimentalData/Ch1_Positive.csv')

    # gathering data and sorting according to error
    K = 128  # number of runs to go through

    # different transition distances to go through
    dist = np.linspace(0, 36, 18)
    L = dist.size-1  # number of transition distances
    d = ['d=%s_' % dist[i] for i in range(dist.size)]

    # gathering all error values
    EValues = np.array([[io.readData(path+d[k]+'info_%s.csv' % i,
                                     typo=str)[1, 2] for i in range(K)]
                        for k in range(L)]).astype(float)

    indices = np.argsort(EValues)

    # gathering F and D from top N profiles
    N = 1
    F = np.array([[io.readData(path+d[k]+'F_%s.txt' % str(indices[k, i])) for i in range(N)] for k in range(L)])
    D = np.array([[io.readData(path+d[k]+'D_%s.txt' % str(indices[k, i])) for i in range(N)] for k in range(L)])

    # # looking and F and D drops in dependence of bin size
    # DDrop = np.array([abs(D[k, 0, 0]-D[k, 0, 1]) for k in range(L)])
    # FDrop = np.array([abs(F[k, 0, 0]-F[k, 0, 1]) for k in range(L)])
    #
    # # print(DDrop.shape)
    # # print(FDrop.shape)
    # # print(dist.shape)
    # # sys.exit()
    # plt.figure(1)
    # plt.plot(dist[:-1], DDrop)
    # plt.xlabel('Transition Distance [bins]')
    # plt.ylabel('C0/Solution Diffusivity Difference [$\mu m^2/s$]')
    # plt.show()
    #
    # plt.figure(2)
    # plt.plot(dist[:-1], FDrop)
    # plt.xlabel('Transition Distance [bins]')
    # plt.ylabel('C0/Solution Free Energy Difference [$k_{B}T$]')
    # plt.show()
    #
    # sys.exit()

    DF = np.array([[fp.computeDF(d=D[k, i, :], f=F[k, i, :-1], dim=100, dx=dist[k]) for i in range(N)] for k in range(L)])

    F = DF[:, :, 1, :]
    D = DF[:, :, 0, :]

    # plotting computed and measured profiles
    # for plotting gathering experimental data
    Cdata = io.readData(path2)
    xx = Cdata[:, 0]
    cc = np.array([Cdata[:, 1], Cdata[:, 31], Cdata[:, 61], Cdata[:, 91]]).T
    tt = np.array([0, 300, 600, 900])  # t in seconds
    # concentration is plotted with smoothing and removal of negative entries
    # as was done during prior to optimization algorithm
    s = [ip.UnivariateSpline(xx, cc[:, i], s=5) for i in range(cc[0, :].size)]
    xs = np.linspace(xx[0], xx[-1], 100)
    cc = np.array([s[i](xs) for i in range(cc[0, :].size)]).T
    xx = xs
    deltaX = abs(xx[1] - xx[0])
    # for negative concentration
    for i in range(cc[:, 0].size):
        for j in range(cc[0, :].size):
            if (cc[i, j]) <= 0:
                cc[i, j] = 0
    c0 = 4
    dim = cc[:, 0].size
    M = cc[0, :].size

    W = np.zeros((L, N, dim, dim))
    W10 = np.zeros((L, N))
    for k in range(L):
        for i in range(N):
            W[k, i, :, :], W10[k, i] = fp.WMatrix(
                D[k, i, :], F[k, i, :], deltaX=deltaX, bc='open1side')

    ccRes = np.array([[[fp.calcC(cc[:, 0], tt[j],
                                 W=W[k, i, :, :], bc='open1side',
                                 W10=W10[k, i], c0=c0) for j in range(M)]
                       for i in range(N)] for k in range(L)])

    # printing top runs
    # print('Top %s Runs with minimal error are: \n' % N)
    # for i in range(N):
        # for k in range(L):
            # print('d=%s: Run #%s with E = %s \n' % (str(dist[k]), str(indices[k, i]), str(EValues[k, indices[k, i]])))

    # plotting results
    cm = plt.get_cmap('hsv')
    cNorm = colors.Normalize(vmin=0, vmax=M)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    xCon = np.linspace(-50, 0, 3)


    for k in range(L):
        for i in range(N):
            plt.figure(i+k)
            plt.gca().set_xlim(left=-50)
            plt.gca().set_xlim(right=xx[-1])
            plt.plot(xCon, np.ones(xCon.size)*D[k, i, 0], 'b--')
            plt.plot(xx, D[k, i, 1:], 'b-')
            plt.xlabel('Distance [$\mu m$]')
            plt.ylabel('Diffusivity [$\mu m^2/s$]')
            plt.title('E=%s, d=%s: \n D_%s' % (str(EValues[k, indices[k, i]]), str(dist[k]), str(indices[k, i])))
            # plt.show()
            if save:
                plt.savefig('/Users/AmanuelWK/Desktop/d=%s_#%s_D.pdf' % (str(dist[k]), str(indices[k, i])))
            plt.pause(0.05)

            plt.figure(i+k+17)
            plt.gca().set_xlim(left=-50)
            plt.gca().set_xlim(right=xx[-1])
            plt.plot(xCon, np.ones(xCon.size)*F[k, i, 0], 'b--')
            plt.plot(xx, F[k, i, 1:], 'b-')
            plt.xlabel('Distance [$\mu m$]')
            plt.ylabel('Free Energy [$k_{B}T$]')
            plt.title('E=%s, d=%s: \n F_%s' % (str(EValues[k, indices[k, i]]), str(dist[k]), str(indices[k, i])))
            # plt.show()
            if save:
                plt.savefig('/Users/AmanuelWK/Desktop/d=%s_#%s_F.pdf' % (str(dist[k]), str(indices[k, i])))
            plt.pause(0.05)

            plt.figure(i+k+36)
            for j in range(M):
                colorVal = scalarMap.to_rgba(j)
                plt.gca().set_xlim(left=-50)
                plt.gca().set_xlim(right=xx[-1])
                plt.xlabel('Distance [$\mu m$]')
                plt.ylabel('Concentration [$\mu M$]')
                plt.plot(xx, cc[:, j], '--', color=colorVal, label=str(int(tt[j]/60))+'m Experiment')
                plt.plot(xx, ccRes[k, i, j, :], '-', color=colorVal, label=str(int(tt[j]/60))+'m Computed')
                plt.plot(xCon, np.ones(xCon.size)*c0, '-', color=colorVal, label=str(int(tt[j]/60))+'m Computed')

            plt.title('E=%s, d=%s: \n C-Profiles from run #%s' % (str(EValues[k, indices[k, i]]), str(dist[k]), str(indices[k, i])))
            plt.legend()
            # plt.show()
            if save:
                plt.savefig('/Users/AmanuelWK/Desktop/d=%s_#%s_profiles.pdf' % (str(dist[k]), str(indices[k, i])))
            plt.pause(0.05)

if __name__ == "__main__":
    main()
