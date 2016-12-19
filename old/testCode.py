# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import inputOutput as io
import FPModel as fp
# import scipy.interpolate as ip
# for debugging
import sys

startTime = time.time()
conservation = True
verbose = True


def main():
    # path for work
    # path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            # 'Skin/Results/ExperimentalData/')
    path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            'Skin/Results/ExperimentalData/')

    # generating test profiles
    c0 = np.concatenate((np.ones(1), np.zeros(99)))
    tt = np.array([0, 600, 6000, 60000])  # t in seconds

    # test F and D shape
    dPre = np.concatenate((np.ones(1)*200, np.ones(80)*200, np.ones(1)*200))
    fPre = np.concatenate((np.zeros(1), np.ones(80)*(-5), np.ones(1)*(-5)))
    segments = np.concatenate((np.ones(10)*0, np.arange(1, 81),
                               np.ones(10)*(81))).astype(int)
    d, f = fp.computeDF(dPre, fPre, shape=segments)
    deltaX = np.array([61, 1, 3076.38])
    deltaXX = np.concatenate((np.ones(6)*deltaX[0],
                              np.ones(1)*(deltaX[0]+deltaX[1])/2,
                              np.ones(80+6)*deltaX[1],
                              np.ones(1)*(deltaX[1]+deltaX[2])/2,
                              np.ones(7)*deltaX[2]))
    '''debugging'''
    deltaXX = np.ones(101)
    W = fp.WMatrixVar(d, f, 80, deltaXX)

    cc = np.array([fp.calcC(c0, tt[i], W=W) for i in range(tt.size)])
    cc.shape()
    sys.exit()
    io.plotCon(cc)

    # setting bounds, D first and F second
    bndsDUpper = np.ones(82)*2000
    bndsFUpper = np.ones(81)*20
    bndsDLower = np.zeros(82)
    bndsFLower = -np.ones(81)*20
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    # setting initial conditions
    DInit = np.random.rand(8)*1000
    # DInit = np.linspace(0, 350, 32)
    FInit = -5

    results = np.array([fp.optimization(DRange=np.ones(82)*DInit[i],
                                        FRange=np.ones(81)*FInit, bnds=bnds,
                                        cc=cc, tt=tt, mode='skinModel',
                                        debug=conservation, verb=verbose, M=tt.size,
                                        N=80)
                        for i in range(DInit.size)])

    np.save('result.npy', results)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
