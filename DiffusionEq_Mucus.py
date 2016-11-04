# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import functools as ft
import inputOutput as io
import FPModel as fp
import scipy.interpolate as ip
from multiprocessing import Pool
# import sys

startTime = time.time()
parallel = True
conservation = False
verbose = False


def main():
    # path for work
    # path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
    #         'Mucus/Ch1_Positive.csv')
    # path for home
    path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
            'Mucus/Results/ExperimentalData/Ch1_Positive.csv')

    # reading profiles
    data = io.readData(path)
    xx = data[:, 0]
    # only taken 4 samples and remove half of elements due to size
    cc = np.array([data[:, 1], data[:, 11], data[:, 21], data[:, 31], data[:, 61], data[:, 91]]).T

    # xx = np.delete(xx, np.arange(0, xx.size, 2))  # remove every 2nd bin
    # cc = np.delete(cc, np.arange(0, cc.size, 2), axis=0)
    # xx = np.delete(xx, np.arange(0, xx.size, 2))  # remove every 2nd bin
    # cc = np.delete(cc, np.arange(0, cc.size, 2), axis=0)

    # smoothing of concentration profiles
    s = [ip.UnivariateSpline(xx, cc[:, i], s=5) for i in range(cc[0, :].size)]
    xs = np.linspace(xx[0], xx[-1], 100)
    cc = np.array([s[i](xs) for i in range(cc[0, :].size)]).T
    xx = xs

    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 100, 200, 300, 600, 900])  # t in seconds
    N = cc[:, 0].size  # number of bins
    c0 = 4  # concentration of peptide solution in µM

    # setting bounds, D first and F second
    bndsD = np.ones(N+1)*np.inf
    bndsF = np.ones(N+1)*20
    bnds = (np.zeros(2*(N+1)), np.concatenate((bndsD, bndsF)))

    # setting initial conditions
    DInit = (np.random.rand(4)*450)+250
    FInit = 10

    # function with one argument (combined d and f) to optimize
    optimize = ft.partial(fp.optimization, DRange=DInit, FRange=FInit,
                          bnds=bnds, cc=cc, tt=tt,
                          deltaX=deltaX, bc='open1side', c0=c0,
                          debug=conservation, verb=verbose)

    ###########################
    # linear and parallel implementation
    ###########################
    if parallel:
        proc = Pool(processes=4)
        for i in proc.imap_unordered(optimize, range(DInit.size)):
            print('#%s: Time elapsed is %s s' % (i, time.time() - startTime))
        proc.close()
    else:
        for i in range(DInit.size):
            optimize(i)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
