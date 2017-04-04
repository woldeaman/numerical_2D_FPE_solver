# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
# import inputOutput as io
# import matplotlib.pyplot as plt
import FPModel as fp
# import scipy.interpolate as ip
# for debugging the files
# import sys

START_TIME = time.time()
conservation = True
verbose = True


def main():
    ''''does something'''

    # generating test profiles
    c0 = np.concatenate((np.ones(10)*0.0025, np.zeros(90))).T
    # time points for which c is to be computed
    tt_Prime = np.array([6, 60, 600])
    tt = np.array([0, 6, 60, 600])  # time points given to trust region

    # simply case for constant D and F
    M = 100  # number of total bins for which c is to be computed
    dPre = np.ones(M)*10
    fPre = np.concatenate((np.zeros(M)))
    W = fp.WMatrix(dPre, fPre)
    cc = np.array([c0, fp.calcC(c0, tt_Prime[0], W=W),
                   fp.calcC(c0, tt_Prime[1], W=W),
                   fp.calcC(c0, tt_Prime[2], W=W)]).T

    # check if profiles look reasonable over time
    # io.plotCon(cc)
    # sys.exit()

    # now trying to find f and d from generated profiles
    # setting bounds for f and d for algorithm
    N = M  # maximal number of measured points in dermis
    bndsDUpper = np.ones(N)*2000
    bndsFUpper = np.ones(N)*20
    bndsDLower = np.zeros(N)
    bndsFLower = -np.ones(N)*20
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))
    # setting starting conditions
    DInit = np.random.rand(N+2, 4)*350
    FInit = -1
    deltaX = np.ones(3)

    results = np.array([fp.optimization(DRange=np.ones(N)*DInit[:, i],
                                        FRange=np.ones(N)*FInit, bnds=bnds,
                                        cc=cc, tt=tt, mode='skinModel',
                                        debug=conservation, verb=verbose,
                                        deltaX=deltaX)
                        for i in range(DInit[0, :].size)])

    np.save('result.npy', results)


if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
