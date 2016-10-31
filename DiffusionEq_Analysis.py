import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as al
import inputOutput as io
import FPModel as fp
import sys


save = True


def main():
    # printing results
    path = ('/Users/AmanuelWK/Desktop/untitled folder/')
    path2 = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModelling/Mucus/Ch1_Positive.csv')

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

    print(D.shape)
    sys.exit()

    Cdata = io.readData(path2)
    xx = Cdata[:, 0]
    deltaX = xx[1] - xx[0]
    cc = np.array([Cdata[:, 1], Cdata[:, 61], Cdata[:, 91]]).T
    tt = np.array([0, 600, 900])  # t in seconds
    dim = cc[:, 0].size

    # gatherin W matrices and calculating concentration profiles
    W = np.zeros((N, dim, dim))
    W10 = np.zeros(N)
    for i in range(N):
        W[i, :, :], W10[i] = fp.WMatrix(D[i, :], F[i, :],
                                        deltaX=deltaX, bc='open1side')

    print(W.shape)
    sys.exit()

    c0 = 4
    Q = np.linalg.inv(W)  # inverse of W
    b = np.append(c0*W10, np.zeros(cc[:, 0].size-1))
    Qb = np.dot(Q, b)

    #ccRes = np.array([np.dot(al.expm(W[i]*tt[0]), cc[:, 0]) + al.expm((W[i, :, :].T)*tt[0])

    print('Top %s Runs with minimal error are: \n' % N)
    for i in range(N):
        print('Run #%s with E = %s \n'
              % (str(indices[i]), EValue[indices[i]]))

    #print ('CC shape:', cc.shape, '\n ccRes shape:', ccRes.shape)
    # plotting results
    for i in range(N):
        plt.figure(i)
        plt.plot(D[i, :])
        plt.title('D_%s' % str(indices[i]))
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_D.pdf'
                        % str(indices[i]))
        plt.show()

        plt.figure(i+1)
        plt.plot(F[i, :])
        plt.title('F_%s' % str(indices[i]))
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_F.pdf'
                        % str(indices[i]))
        plt.show()

        plt.figure(i+2)
        expData0, = plt.plot(xx, cc[0, :], 'b--', label='0m Experiment')
        calcData0, = plt.plot(xx, ccRes[i, 0, :],
                              'b-', label='0m Computed')
        expData1, = plt.plot(xx, cc[1, :], 'r--', label='10m Experiment')
        calcData1, = plt.plot(xx, ccRes[i, 1, :],
                              'r-', label='10m Computed')
        expData2, = plt.plot(xx, cc[2, :], 'g--', label='15m Experiment')
        calcData2, = plt.plot(xx, ccRes[i, 2, :],
                              'g-', label='15m Computed')
        expData3, = plt.plot(xx, cc[3, :], 'k--',
                             label='15m Experiment')
        calcData3, = plt.plot(xx, ccRes[i, 3, :],
                              'k-', label='15m Computed')

        plt.title('C-Profiles from run #%s' % (str(indices[i]+1)))
        plt.legend(handles=[expData0, calcData0, expData1, calcData1,
                            expData2, calcData2, expData3, calcData3])
        if save:
            plt.savefig('/Users/AmanuelWK/Desktop/#%s_profiles.pdf'
                        % str(indices[i]))
        plt.show()

if __name__ == "__main__":
    main()
