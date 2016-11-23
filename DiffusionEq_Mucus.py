# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import functools as ft
import inputOutput as io
import FPModel as fp
from multiprocessing import Pool
import scipy.interpolate as ip
# for debugging
# import sys
# import matplotlib.pyplot as plt

startTime = time.time()
parallel = True
conservation = False
verbose = True


def main():
    # path for work
    # path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            # 'Mucus/Results/ExperimentalData/Ch1_Positive.csv')
    # path for home
    path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            'Mucus/Results/ExperimentalData/Ch1_Positive.csv')

    # reading profiles
    data = io.readData(path)
    xx = data[:, 0]
    # only taken 4 samples
    cc = np.array([data[:, 1], data[:, 31], data[:, 61], data[:, 91]]).T

    # smoothing of concentration profiles
    s = [ip.UnivariateSpline(xx, cc[:, i], s=5) for i in range(cc[0, :].size)]
    xs = np.linspace(xx[0], xx[-1], 100)
    cc = np.array([s[i](xs) for i in range(cc[0, :].size)]).T
    xx = xs

    # taking care of negative concentration values
    for i in range(cc[:, 0].size):
        for j in range(cc[0, :].size):
            if (cc[i, j]) <= 0:
                cc[i, j] = 0

    # plotting concentration profiles
    # io.plotCon(cc=cc, xx=xx, live=False)
    # sys.exit()

    # finding interface position
    # index = np.where(abs(xx-100) == np.min(abs(xx - 100)))
    # print('Size:', xx.size, 'Index:', index)
    # print(xx[index])
    # sys.exit()

    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM

    # setting bounds, D first and F second
    # bndsD = np.ones(N+1)*np.inf
    # bndsF = np.ones(N+1)*20
    # bnds = (np.zeros(2*(N+1)), np.concatenate((bndsD, bndsF)))

    ''' quick and dirty implementation, change later on '''
    # changed for segmentation analysis
    bndsD = np.ones(3)*np.inf
    bndsF = np.ones(3)*20
    bndsDist = np.ones(1)*36
    bnds = (np.zeros((2*3)+1), np.concatenate((bndsD, bndsF, bndsDist)))

    # setting initial conditions
    DInit = (np.random.rand(256)*400)+200
    # DInit = np.random.rand()
    FInit = 10
    DistInit = np.arange(0, 37, 2)

    for k in range(DistInit.size):
        # function with one argument (combined d and f) to optimize
        optimize = ft.partial(fp.optimization, DRange=DInit, FRange=FInit,
                              DistRange=DistInit[k], bnds=bnds, cc=cc, tt=tt,
                              deltaX=deltaX, bc='segmented', c0=c0,
                              debug=conservation, verb=verbose)

        ###########################
        # linear and parallel implementation
        ###########################
        if parallel:
            proc = Pool(processes=32)
            for i in proc.imap_unordered(optimize, range(DInit.size)):
                print('#%s: Time elapsed is %s s' %
                      (i, time.time() - startTime))
                proc.close()
            else:
                for i in range(DInit.size):
                    optimize(i)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
