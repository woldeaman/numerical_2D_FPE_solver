# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import inputOutput as io
import matplotlib.pyplot as plt
import FPModel as fp
# import scipy.interpolate as ip
# for debugging
import sys

startTime = time.time()
conservation = True
verbose = True


def main():

    # generating test profiles
    c0 = np.concatenate((np.ones(10)*0.0025, np.zeros(90))).T
    tt_Prime = np.array([6, 60, 600])  # time points for which c is to be computed
    tt = np.array([0, 6, 60, 600])  # time points given to trust region

    # simply case for constant D and F
    M = 100  # number of total bins for which c is to be computed
    dPre = np.ones(M)*100
    fPre = np.zeros(M)
    W = fp.WMatrix(dPre, fPre)
    cc = np.array([c0, fp.calcC(c0, tt_Prime[0], W=W)[10:83],
                   fp.calcC(c0, tt_Prime[1], W=W)[10:90],
                   fp.calcC(c0, tt_Prime[2], W=W)[10:90]]).T

    # check if profiles look reasonable over time
    # for i in range(cc.size):
    #     plt.plot(cc[i])
    #     plt.axis([0, 100, 0, 0.0025])
    #     plt.show()
    # sys.exit()

    # now trying to find f and d from generated profiles
    # setting bounds for f and d for algorithm
    N = 80  # maximal number of measured points in dermis
    bndsDUpper = np.ones(N+2)*2000
    bndsFUpper = np.ones(N+1)*20
    bndsDLower = np.zeros(N+2)
    bndsFLower = -np.ones(N+1)*20
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))
    # setting starting conditions
    DInit = np.random.rand(N+2, 4)*350
    FInit = -1
    deltaX = np.ones(3)

    results = np.array([fp.optimization(DRange=np.ones(N+2)*DInit[:, i],
                                        FRange=np.ones(N+1)*FInit, bnds=bnds,
                                        cc=cc, tt=tt, mode='skinModel',
                                        debug=conservation, verb=verbose,
                                        deltaX=deltaX)
                        for i in range(DInit[0, :].size)])

    np.save('result.npy', results)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
