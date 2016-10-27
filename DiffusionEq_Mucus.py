# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck

import numpy as np
import time
import functools as ft
import inputOutput as io
import FPModel as fp
import scipy.optimize as op

startTime = time.time()


def main():
    # path for work
    # path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/DiffusionModel/'
    #         'Skin/Results/ExpData/')

    # reading profiles
    path = ('/home/amanuelwk/GoogleDrive/PhD/Projects/FokkerPlanckModelling/'
            'Skin/Results/ExpData/')
    cc = np.array([io.readData(path+'10min.txt'),
                   io.readData(path+'100min.txt'),
                   io.readData(path+'1000min.txt')]).T

    N = cc[:, 0].size  # number of bins
    cc0 = np.append(1, np.zeros(N-1))  # initial concentration profile
    cc = np.insert(cc, 0, cc0, axis=1)
    tt = np.array([0, 600, 6000, 60000])  # t in seconds

    # setting bounds, D first and F second
    bndsD = np.ones(N)*np.inf
    bndsF = np.ones(N)*20
    bnds = (np.zeros(2*N), np.concatenate((bndsD, bndsF)))

    # function with one argument (combined d and f) to optimize
    optimize = ft.partial(fp.resFun, cc=cc, tt=tt, debug=True, verb=True)

    ###########################
    # linear implementation
    ###########################
    # setting initial conditions
    DInit = np.linspace(0, 100, num=4)
    FInit = 5

    for i in range(DInit.size):
        initVal = np.append(np.ones(N)*DInit[i], np.ones(N)*FInit)
        # running 5x50 with varied starting points based on initVal
        DValStart = initVal[0]
        for l in range(5):
            result = op.least_squares(optimize, initVal, bounds=bnds,
                                      max_nfev=50, tr_solver='lsmr')
            initVal = result.x

        result = op.least_squares(optimize, initVal,
                                  bounds=bnds, tr_solver='lsmr')

        # saving data from result
        values = open('info_%s.csv' % i, 'w')
        values.write('#, DValue, EValue, #OfEvaluations, Message\n')
        values.write(str(i)+', ' + str(DValStart) + ', ' +
                     str(result.cost) + ', ' + str(result.nfev) +
                     ', ' + result.message+'\n')
        values.close()
        D = result.x[:N]
        F = result.x[N:]
        np.savetxt('D_%s.txt' % i, D, delimiter=', ')
        np.savetxt('F_%s.txt' % i, F, delimiter=', ')


if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
