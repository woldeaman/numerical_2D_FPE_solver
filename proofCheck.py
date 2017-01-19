# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import inputOutput as io
import FPModel as fp
import scipy.linalg as al
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

# import scipy.interpolate as ip
# for debugging
import sys


def main():
    # path for work
    path = ('/Users/AmanuelWK/Desktop/Robert Results/')
    path2 = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
             'Skin/Results/ExperimentalData/')

    # reading roberts F and D
    DPre = io.readData(path+'D.txt')
    FPre = io.readData(path+'F.txt')
    d = DPre
    f = FPre

    # Roberts values for discretization
    # 7 bins for segment 1 = gel, with 5x deltaX1 and 2x deltaX2
    # 15 bins for segment 3 = dermis, with 4x deltaX2 and 11x deltaX2
    X2 = 1  # discretization length in epidermis is 1µm
    X1 = (400-(2.5*X2))/4.5  # transition between discretizations at bin 5
    X3 = (20000-(4.5*X2))/10.5  # transition between discretizations at bin N+4
    deltaX = np.array([X1, X2, X3])

    # values for discretization width along domain
    deltaXX = np.concatenate((np.ones(5)*deltaX[0],
                              np.ones(87)*deltaX[1],
                              np.ones(11)*deltaX[2]))
zai
    W = fp.WMatrixVar(d, f, 80, deltaXX, True)

    # print(W[:5, :5])
    # sys.exit()

    # numerical error min 100 times smaller than first entry of W
    if abs(np.sum(np.sum(W, 0))) > 1E-2:
        print('Error: W Matrix is not row stochastic in rows: \n',
              np.nonzero(abs(np.sum(W, 0)) > 1E-2), '\n')
        print('Row Sum:\n', np.sum(W, 0), '\n')
        print('Total Sum:\n', np.sum(np.sum(W, 0)), '\n')
        sys.exit()

    # reading concentration profiles
    cc = np.array([np.concatenate((np.ones(7)*0.0025, np.zeros(95))),
                   io.readData(path2+'p10min.txt')[:73],
                   io.readData(path2+'p100min.txt')[:80],
                   io.readData(path2+'p1000min.txt')[:80]]).T
    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    M = tt.size

    RR = np.zeros((102, M-1))
    k = 0

    for i in range(1, M):
        RR[:cc[i].size, k] = cc[i] - fp.calcC(cc[0], tt[i], W=W)[8:(cc[i].size+8)]
        k += 1

    RRn = np.array([al.norm(RR[:, i]) for i in range(RR[0, :].size)])
    E = np.sqrt(np.sum((RRn**2)*(np.array([1/73, 1/80, 1/80])))/3)  # normalized version
    print(E*600)

    ccRes = np.array([fp.calcC(cc[0], tt[j], W=W) for j in range(M)])
    ccRob = np.array([io.readData(path+'10min.txt'),
                      io.readData(path+'100min.txt'),
                      io.readData(path+'1000min.txt')]).T

    diff = np.array([100*abs(ccRes[i+1, 7:ccRob[i].size+7]-ccRob[i])/ccRob[i] for i in range(ccRob.size)])
    print([np.mean(diff[i]) for i in range(diff.size)])
    sys.exit()
    N = 1
    xx = np.arange(100)
    save = False
    cm = plt.get_cmap('hsv')
    cNorm = colors.Normalize(vmin=0, vmax=M)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for i in range(N):
        plt.figure(i)
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.plot(xx, d)
        plt.xlabel('Distance [$\mu m$]')
        plt.ylabel('Diffusivity [$\mu m^2/s$]')
        plt.title('D')
        if save:
            print('meh')
        else:
            plt.show()

        plt.figure(i+1)
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        plt.plot(xx, f)
        plt.xlabel('Distance [$\mu m$]')
        plt.ylabel('Free Energy [$k_{B}T$]')
        plt.title('F')
        if save:
            print('meh')
        else:
            plt.show()

        plt.figure(i+2)
        colorVal = scalarMap.to_rgba(0)
        plt.plot(xx, cc[0], '--', color=colorVal,
                 label=str(int(tt[0]/60))+'m Experiment')
        plt.plot(xx, ccRes[0, :], '-', color=colorVal,
                 label=str(int(tt[0]/60))+'m Computed')
        for j in range(1, M):
            colorVal = scalarMap.to_rgba(j)
            # plt.gca().set_xlim(left=xx[0])
            # plt.gca().set_xlim(right=xx[-1])
            plt.xlabel('Distance [$\mu m$]')
            plt.ylabel('Concentration [$\mu M$]')
            plt.plot(xx[11:cc[j].size+11], cc[j], '--', color=colorVal,
                     label=str(int(tt[j]/60))+'m Experiment')
            plt.plot(xx, ccRes[j, :], '-', color=colorVal,
                     label=str(int(tt[j]/60))+'m Computed')
        plt.title('C-Profiles')
        plt.legend()
        if save:
            print('meh')
        else:
            plt.show()


if __name__ == "__main__":
    main()
