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

startTime = time.time()
parallel = True
conservation = False
verbose = True


def main():
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
            'Mucus/Ch1_Positive.csv')
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
    #         'Skin/Results/ExpData/')

    # reading profiles
    data = io.readData(path)
    xx = data[:, 0]
    # only taken 4 samples and remove half of elements due to size
    cc = np.array([data[:, 1], data[:, 61], data[:, 91]]).T
    # cc = cc[80:, :]  # only look at profiles starting from 100µm (after c0)
    # xx = xx[80:]
    # xx = np.delete(xx, np.arange(0, xx.size, 2))  # remove every 2nd bin
    # cc = np.delete(cc, np.arange(0, cc.size, 2), axis=0)

    s = [ip.UnivariateSpline(xx, cc[:, i], s=15) for i in range(cc[0, :].size)]
    xs = np.linspace(xx[0], xx[-1], 100)
    cc = np.array([s[i](xs) for i in range(cc[0, :].size)]).T
    xx = xs

    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    N = cc[:, 0].size  # number of bins
    c0 = 4  # concentration of peptide solution in µM

    # setting bounds, D first and F second
    bndsD = np.ones(N+1)*np.inf
    bndsF = np.ones(N+1)*20
    bnds = (np.zeros(2*(N+1)), np.concatenate((bndsD, bndsF)))

    # setting initial conditions
    DInit = np.linspace(1, 1000, num=4)
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
        for i in proc.imap_unordered(optimize, range(4)):
            print('#%s: Time elapsed is %s s' % (i, time.time() - startTime))
        proc.close()
    else:
        for i in range(DInit.size):
            optimize(i)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
