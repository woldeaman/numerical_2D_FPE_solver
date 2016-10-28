# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck

import numpy as np
import time
import functools as ft
import inputOutput as io
import FPModel as fp
from multiprocessing import Pool

startTime = time.time()
parallel = True
debug = True
verbose = True


def main():
    # path for work
    # path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/DiffusionModel/'
    #         'Skin/Results/ExpData/')

    # reading profiles
    path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
            'Skin/Results/ExpData/')

    # data = io.readData(path)
    # xx = data[:, 0]
    # # only taken 3 samples and remove half of elements due to size
    # cc = np.array([data[:, 1], data[:, 61], data[:, 91]]).T
    # xx = np.delete(xx, np.arange(0, xx.size, 2))
    # cc = np.delete(cc, np.arange(0, cc.size, 2), axis=0)
    # deltaX = abs(xx[0] - xx[1])
    deltaX = 1

    cc = np.array([io.readData(path+'10min.txt'),
                   io.readData(path+'100min.txt'),
                   io.readData(path+'1000min.txt')]).T

    N = cc[:, 0].size  # number of bins
    cc0 = np.append(1, np.zeros(N-1))  # initial concentration profile
    cc = np.insert(cc, 0, cc0, axis=1)
    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    # tt = np.array([0, 600, 900])  # t in seconds

    # setting bounds, D first and F second
    bndsD = np.ones(N)*np.inf
    bndsF = np.ones(N)*20
    bnds = (np.zeros(2*(N)), np.concatenate((bndsD, bndsF)))

    # setting initial conditions
    DInit = np.linspace(5, 100, num=4)
    FInit = 5

    # function with one argument (combined d and f) to optimize
    optimize = ft.partial(fp.optimization, cc=cc, tt=tt, deltaX=deltaX,
                          bc='reflective', DRange=DInit, FRange=FInit,
                          bnds=bnds, debug=debug, verb=verbose)

    ###########################
    # linear and parallel implementation
    ###########################
    if parallel:
        proc = Pool(processes=4)
        for i in proc.imap_unordered(optimize, range(4)):
            print('#%s: Time elapsed is %s s' % (i, time.time() - startTime))
        proc.close()
    else:
        for i in range(DInit.size):
            optimize(i)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
