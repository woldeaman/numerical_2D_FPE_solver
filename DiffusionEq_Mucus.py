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
import os
# for debugging
import sys
# import matplotlib.pyplot as plt

startTime = time.time()
parallel = False
conservation = False
verbose = True


def main():
    # path for work
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
            'Mucus/Results/ExperimentalData/Ch3_Negative.csv')
    # path for home
    # path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModeling/'
    #         'Mucus/Results/ExperimentalData/Ch1_Positive.csv')

    # reading profiles and take only samples for 4 different time points
    data = io.readData(path)
    xx = data[:, 0]
    cc = np.array([data[:, 1], data[:, 31], data[:, 61], data[:, 91]]).T

    # pre processing of profiles
    xx, cc = io.preProcessing(xx, cc)
    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM

    # plotting concentration profiles
    # io.plotCon(cc=cc, xx=xx, live=True)
    # sys.exit()

    # finding interface position
    # index = np.where(abs(xx-100) == np.min(abs(xx - 100)))
    # print('Size:', xx.size, 'Index:', index)
    # print(xx[index])
    # sys.exit()

    # setting bounds, D first and F second
    # bndsD = np.ones(N+1)*np.inf
    # bndsF = np.ones(N+1)*20
    # bnds = (np.zeros(2*(N+1)), np.concatenate((bndsD, bndsF)))

    # changed for segmentation analysis
    bndsD = np.ones(3)*np.inf
    bndsF = np.ones(3)*20
    bnds = (np.zeros((2*3)), np.concatenate((bndsD, bndsF)))

    # setting initial conditions
    # DInit = (np.random.rand(1)*400)+200
    DInit = np.linspace(200, 400, 4)
    FInit = 10
    transition = np.linspace(0, 36, 1)

    for k in range(transition.size):
        # function with one argument (combined d and f) to optimize
        optimize = ft.partial(fp.optimization, DRange=DInit, FRange=FInit,
                              Dist=transition[k], bnds=bnds, cc=cc, tt=tt,
                              deltaX=deltaX, bc='segmented', c0=c0,
                              debug=conservation, verb=verbose)

        # creating output directories
        currentDir = os.getcwd()
        paths = currentDir+'/d='+str(transition[k])
        os.makedirs(paths)

        ###########################
        # linear and parallel implementation
        ###########################
        if parallel:
            proc = Pool(processes=4)
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
