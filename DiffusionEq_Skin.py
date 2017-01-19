# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import inputOutput as io
import FPModel as fp
# import scipy.interpolate as ip
# for debugging
# import sys

startTime = time.time()
conservation = True
verbose = True


def main():
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            'Skin/Results/ExperimentalData/')
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
    #         'Skin/Results/ExperimentalData/')

    # reading profiles
    cc = np.array([np.concatenate((np.ones(10)*0.0025, np.zeros(90))),
                   io.readData(path+'p10min.txt')[:73],
                   io.readData(path+'p100min.txt')[:80],
                   io.readData(path+'p1000min.txt')[:80]]).T

    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    N = max([cc[i].size for i in range(1, cc.size)])  # number of bins
    # computing discretization lengths
    X2 = 1  # discretization length in epidermis is 1µm
    X1 = (400-(3.5*X2))/6.5  # transition between discretizations at bin 7
    X3 = (20000-(3.5*X2))/6.5  # transition between discretizations at bin 83
    deltaX = np.array([X1, X2, X3])

    # setting bounds, D first and F second
    bndsDUpper = np.ones(N+2)*2000
    bndsFUpper = np.ones(N+1)*20
    bndsDLower = -np.ones(N+2)
    bndsFLower = -np.ones(N+1)*20
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    # setting initial conditions
    DInit = np.random.rand(N+2)*350
    FInit = -5
    # trying Roberts results as initial data
    d = np.loadtxt('/Users/AmanuelWK/Desktop/Robert Results/D.txt')
    f = np.loadtxt('/Users/AmanuelWK/Desktop/Robert Results/F.txt')
    DInit = np.concatenate((np.ones(1)*d[0], d[7:-15], np.ones(1)*d[-1]))
    # for i in range(DInit.size):
    #     if DInit[i] < 0:
    #         DInit[i] = 0
    FInit = np.concatenate((f[7:-15], np.ones(1)*f[-1]))

    # now with completely random D
    # DInit = np.random.rand(N+2, 128)*1000

    results = np.array([fp.optimization(DRange=DInit[i]*np.ones(N+2),
                                        FRange=FInit*np.ones(N+1), bnds=bnds,
                                        cc=cc, tt=tt, mode='skinModel',
                                        debug=conservation, verb=verbose,
                                        deltaX=deltaX)
                        for i in range(DInit.size)])

    np.save('result.npy', results)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
