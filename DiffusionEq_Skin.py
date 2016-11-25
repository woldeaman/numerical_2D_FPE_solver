# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import functools as ft
import inputOutput as io
import FPModel as fp
from multiprocessing import Pool
# import scipy.interpolate as ip
# for debugging
# import sys
# import matplotlib.pyplot as plt

startTime = time.time()
parallel = False
conservation = False
verbose = True


def main():
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            'Skin/Results/ExperimentalData/')
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
    #         'Skin/Results/ExperimentalData/')

    # reading profiles
    data = np.array([np.concatenate((np.ones(1), np.zeros(70))),
                     io.readData(path+'10min.txt'),
                     io.readData(path+'100min.txt'),
                     io.readData(path+'1000min.txt')])
    cc = data.T
    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    N = cc[:, 0].size  # number of bins
    # deltaX = 1E-6

    # setting bounds, D first and F second
    bndsD = np.ones(N)*2000
    bndsF = np.ones(N)*20
    bnds = (np.zeros(2*(N)), np.concatenate((bndsD, bndsF)))

    # setting initial conditions
    # DInit = (np.random.rand(4)*1000)
    DInit = np.linspace(0, 1000, 4)
    FInit = 5

    # function with one argument (combined d and f) to optimize
    optimize = ft.partial(fp.optimization, DRange=DInit, FRange=FInit,
                          bnds=bnds, cc=cc, tt=tt, bc='reflective',
                          debug=conservation, verb=verbose)

    ###########################
    # linear and parallel implementation
    ###########################
    if parallel:
        proc = Pool(processes=8)
        for i in proc.imap_unordered(optimize, range(DInit.size)):
            print('#%s: Time elapsed is %s s' % (i, time.time() - startTime))
            proc.close()
    else:
        for i in range(DInit.size):
            optimize(i)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
