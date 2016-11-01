import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as al
import inputOutput as io
import FPModel as fp
import matplotlib.colors as colors
import matplotlib.cm as cmx


save = True


def main():
    # printing results
    path = ('/Users/AmanuelWK/Desktop/Mucus_Negative/')
    path2 = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModelling/Mucus/Ch3_Negative.csv')

    # gathering data and sorting according to error
    M = 4  # number of runs to go through

    data = np.array([io.readData(path+'info_%s.csv' % i, typo=str)[1, :-1]
                     for i in range(M)]).astype(float)
    DValue, EValue = data[:, 1], data[:, 2]
    indices = np.argsort(EValue)

    # Finding Top N runs
    N = 4
    F = np.array([io.readData(path+'F_%s.txt' % str(indices[i]))
                  for i in range(N)])
    D = np.array([io.readData(path+'D_%s.txt' % str(indices[i]))
                  for i in range(N)])

    Cdata = io.readData(path2)
    xx = Cdata[:, 0]
    cc = np.array([Cdata[:, 1], Cdata[:, 61], Cdata[:, 91]]).T
    cc = cc[80:, :]  # only look at profiles starting from 100µm (after c0)
    xx = xx[80:]
    xx = np.delete(xx, np.arange(0, xx.size, 2))
    cc = np.delete(cc, np.arange(0, cc.size, 2), axis=0)
    deltaX = xx[1] - xx[0]
    tt = np.array([0, 600, 900])  # t in seconds
    dim = cc[:, 0].size
    M = cc[0, :].size

    # gatherin W matrices and calculating concentration profiles
    W = np.zeros((N, dim, dim))
    W10 = np.zeros(N)
    for i in range(N):
        W[i, :, :], W10[i] = fp.WMatrix(D[i, :], F[i, :],
                                        deltaX=deltaX, bc='open1side')

    c0 = 4
    Q = np.array([np.linalg.inv(W[i, :, :]) for i in range(N)])  # inverse of W
    T = np.array([al.expm(W[i, :, :]) for i in range(N)])
    b = np.array([np.append(c0*W10[i], np.zeros(dim-1)) for i in range(N)])
    Qb = np.array([np.dot(Q[i, :, :], b[i, :]) for i in range(N)])

    ccRes = np.array([[np.dot(np.linalg.matrix_power(T[i, :, :], tt[j]), cc[:, 0]) + np.dot(np.linalg.matrix_power(T[i, :, :], tt[j]) - np.eye(dim), Qb[i, :]) for j in range(M)] for i in range(N)])

    print('Top %s Runs with minimal error are: \n' % N)
    for i in range(N):
        print('Run #%s with E = %s \n'
              % (str(indices[i]), EValue[indices[i]]))

    # plotting results
    xxDF = np.insert(xx, 0, xx[0]-deltaX,)
    cm = plt.get_cmap('hsv')
    cNorm = colors.Normalize(vmin=0, vmax=M)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for i in range(N):
        plt.figure(i)
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.plot(xxDF, D[i, :])
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
            plt.plot(xx, cc[:, j], '--', color=colorVal, label=str(int(tt[j]/60))+'m Experiment')
            plt.plot(xx, ccRes[i, j, :], '-', color=colorVal, label=str(int(tt[j]/60))+'m Computed')

        #expData1, = plt.plot(xx, cc[:, 1], 'r--', label='10m Experiment')
        #calcData1, = plt.plot(xx, ccRes[i, 1, :],
        #                      'r-', label='10m Computed')
        #expData2, = plt.plot(xx, cc[:, 2], 'g--', label='15m Experiment')
        #calcData2, = plt.plot(xx, ccRes[i, 2, :],
        #                      'g-', label='15m Computed')
        # expData3, = plt.plot(xx, cc[:, 3], 'k--',
        #                      label='15m Experiment')
        # calcData3, = plt.plot(xx, ccRes[i, 3, :],
        #                      'k-', label='15m Computed')

        plt.title('C-Profiles from run #%s' % (str(indices[i]+1)))
        plt.legend()
        #plt.legend(handles=[expData0, calcData0, expData1, calcData1,
        #                       expData2, calcData2])
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_profiles.pdf'
                        % str(indices[i]))
        plt.show()

if __name__ == "__main__":
    main()
