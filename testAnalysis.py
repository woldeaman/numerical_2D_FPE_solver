import matplotlib.pyplot as plt
import numpy as np
import inputOutput as io
import FPModel as fp
import matplotlib.colors as colors
import scipy.interpolate as ip
import matplotlib.cm as cmx
import sys

save = False
'''
code this more generally for different conditions will ya?
'''

def main():
    # printing results
    # path for home
    # path = ('/home/amanuelwk/Desktop/Positive/')
    # path2 = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
            #  'Mucus/Results/ExperimentalData/Ch1_Positive.csv')

    path = ('/Users/AmanuelWK/GoogleDrive/PhD/GitHub/FokkerPlanckModeling/')
    path2 = ('/Users/AmanuelWK/GoogleDrive/PhD/GitHub/FokkerPlanckModeling/cc.txt')

    # gathering data and sorting according to error
    K = 4  # number of runs to go through

    data = np.array([io.readData(path+'info_%s.csv' % i, typo=str)[1, :-1]
                     for i in range(K)]).astype(float)
    EValue = data[:, 2]
    indices = np.argsort(EValue)

    # Finding Top N runs
    N = 4
    F = np.array([io.readData(path+'F_%s.txt' % str(indices[i]))
                  for i in range(N)])
    D = np.array([io.readData(path+'D_%s.txt' % str(indices[i]))
                  for i in range(N)])


    # for plotting gathering experimental data
    Cdata = io.readData(path2)
    # xx = Cdata[:, 0]
    # print(Cdata.shape)
    # print(Cdata)
    # sys.exit()

    cc = Cdata

    # with smoothing as during optimization
    # s = [ip.UnivariateSpline(xx, cc[:, i], s=5) for i in range(cc[0, :].size)]
    # xs = np.linspace(xx[0], xx[-1], 100)
    # cc = np.array([s[i](xs) for i in range(cc[0, :].size)]).T
    # xx = xs
    # deltaX = abs(xx[1] - xx[0])

    tt = np.array([0, 10, 20, 30, 40, 50])
    c0 = 4
    dim = cc[:, 0].size
    M = cc[0, :].size
    xx = range(dim)
    deltaX = abs(xx[1] - xx[0])


    # gatherin W matrices and calculating concentration profiles
    W = np.zeros((N, dim, dim))
    W10 = np.zeros(N)
    # for i in range(N):
        # W[i, :, :], W10[i] = fp.WMatrix(D[i, :], F[i, :],
                                        # deltaX=deltaX, bc='open1side')
    for i in range(N):
        W[i, :, :], W10[i] = fp.WMatrixPart(D[i, :], F[i, :])

    ccRes = np.array([[fp.calcC(cc[:, 0], tt[j], W=W[i, :, :],
                                bc='open1side', W10=W10[i], c0=c0)
                       for j in range(M)] for i in range(N)])

    print('Top %s Runs with minimal error are: \n' % N)
    for i in range(N):
        print('Run #%s with E = %s \n'
              % (str(indices[i]), EValue[indices[i]]))

    FTemp = np.array([np.append(np.ones(1)*F[i, 0], np.ones(12)*F[i, 1]) for i in range(N)])
    F = np.array([np.append(FTemp[i, :], np.ones(13)*F[i, 2]) for i in range(N)])

    DTemp = np.array([np.append(np.ones(1)*D[i, 0], np.ones(12)*D[i, 1]) for i in range(N)])
    D = np.array([np.append(DTemp[i, :], np.ones(13)*D[i, 2]) for i in range(N)])


    # plotting results
    xxDF = np.insert(xx, 0, xx[0]-deltaX,)
    cm = plt.get_cmap('hsv')
    cNorm = colors.Normalize(vmin=0, vmax=M)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    '''
    change plotting for D and F
    '''
    for i in range(N):
        plt.figure(i)
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.plot(xxDF, D[i, :])
        plt
        plt.xlabel('Distance [$\mu m$]')
        plt.ylabel('Diffusivity [$\mu m^2/s$]')
        plt.title('D_%s' % str(indices[i]))
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_D.pdf'
                        % str(indices[i]))
        plt.show()

        plt.figure(i+1)
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.plot(xxDF, F[i, :])
        plt.xlabel('Distance [$\mu m$]')
        plt.ylabel('Free Energy [$k_{B}T$]')
        plt.title('F_%s' % str(indices[i]))
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_F.pdf'
                        % str(indices[i]))
        plt.show()

        plt.figure(i+2)
        for j in range(M):
            colorVal = scalarMap.to_rgba(j)

            plt.gca().set_xlim(left=xx[0])
            plt.gca().set_xlim(right=xx[-1])
            plt.xlabel('Distance [$\mu m$]')
            plt.ylabel('Concentration [$\mu M$]')
            plt.plot(xx, cc[:, j], '--', color=colorVal,
                     label=str(int(tt[j]/60))+'m Experiment')
            plt.plot(xx, ccRes[i, j, :], '-', color=colorVal,
                     label=str(int(tt[j]/60))+'m Computed')

        plt.title('C-Profiles from run #%s' % (str(indices[i])))
        plt.legend()
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_profiles.pdf'
                        % str(indices[i]))
        plt.show()

if __name__ == "__main__":
    main()
