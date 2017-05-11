# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as al
import functools as ft
import scipy.optimize as op
import sys
import time
import FPModel as fp
import inputOutput as io
import argparse as ap


def resFun(df, cc, tt, deltaX=1, debug=False, verb=False):
    '''
    cc and tt arrays with concentration profiles cc[:,i]
    for time tt[i] and tt[j] > tt[i] if j > i
    were cc is 2D array with cc.shape = (number of samples, number of bins)
    '''

    if len(cc.shape) == 1:
        # catches the case of differently sized concentration profiles
        # within cc and simply takes maximum as N
        N = np.max(np.array([cc[i].size for i in range(1, cc.size)]))
        M = cc.size
    else:
        M = cc[0, :].size  # number of concentration profiles
        N = cc[:, 0].size  # number of bins

    # pre-processing for d and f vectors
    dPre = df[:(N+2)]  # take input values from trust region algorithm
    # fPre = np.concatenate((np.zeros(1), df[(N+2):]))
    # completely free F
    fPre = df[(N+2):]
    # defined for keeping d and f constant in certain areas
    segments = np.concatenate((np.zeros(7), np.arange(1, N+1),
                               np.ones(15)*(N+1))).astype(int)
    d, f = fp.computeDF(dPre, fPre, shape=segments)
    # transition between discretization is within segment 1 at pos. 5
    # and segment 2 at pos. N+7-10
    # bin discretization widths are stored in deltaXX
    deltaXX = np.concatenate((np.ones(5)*deltaX[0],
                              np.ones(N+7)*deltaX[1],
                              np.ones(11)*deltaX[2]))
    W = fp.WMatrixVar(d, f, N, deltaXX, con=debug)
    T = al.expm(W)

    # check for detailed balance and conservation of concentration
    # (only for reflective boundaries)
    if debug:
        # numerical error min 100 times smaller than zero
        if abs(np.sum(np.sum(W, 0))) > 1E-2:
            print('Error: W Matrix is not row stochastic in rows: \n',
                  np.nonzero(abs(np.sum(W, 0)) > 1E-2), '\n')
            print('Row Sum:\n', np.sum(W, 0), '\n')
            print('Total Sum:\n', np.sum(np.sum(W, 0)), '\n')
            sys.exit()

        # same check for T matrix
        # if abs(np.sum(np.sum(T, 0))-T[:, 0].size) > T[:, 0].size*1E-2:
        #     print('Error: T Matrix is not row stochatic in rows: \n',
        #           np.nonzero(abs(np.sum(T, 0)-1) > 1E-2), '\n')
        #     print('Row Sum:\n', np.sum(T, 0), '\n')
        #     print('Total Sum:\n', np.sum(np.sum(T, 0)), '\n')
        #     sys.exit()

        deltaXC = np.concatenate((np.ones(4)*deltaX[0],
                                  np.ones(1)*(deltaX[0]+deltaX[1])/2,
                                  np.ones(N+6)*deltaX[1],
                                  np.ones(1)*(deltaX[1]+deltaX[2])/2,
                                  np.ones(10)*deltaX[2]))
        # deltaXC = deltaXX[:-1]
        con = np.sum(cc[0]*deltaXC)

        # compute profiles from c0 and do the same conservation check
        ccComp = np.array([fp.calcC(cc[0], t=tt[i], W=W)
                           for i in range(M)])

        if np.any(np.array([abs(np.sum(ccComp[i]*deltaXC)-con)
                            for i in range(M)]) > 0.01*con):
            print('Error: Computed concentration '
                  'is not conserved in profiles: \n',
                  np.nonzero(np.array([abs(np.sum(ccComp[i]*deltaXC)-con)
                                       for i in range(M)]) > 0.01*con))
            print([np.sum(ccComp[i]*deltaXC) for i in range(M)], '\n')
            print('concentration:\n', con)
            print('WMatrix Size:\n', W.shape)
            print('WMatrix Row Sum:\n', np.sum(W, 0))
            print('WMatrix 2Sum:\n', np.sum(np.sum(W, 0)))
            sys.exit()

    # calculating vector of residuals for each profile from c_0
    # RR = [cc[i] -
    #       fp.calcC(cc[0], tt[i], T=T,
    #                bc='reflective')[7:(cc[i].size+7)] for i in range(1, M)]
    # residuals = np.concatenate((RR[0], RR[1], RR[2]))

    # calculating vector of resiuduals for all combinations
    # hard-coded because no easy workaround
    # all residuals where c_0 profile is used for numerical computations
    Ri_0 = [cc[i] - fp.calcC(cc[0], tt[i], T=T)[7:(cc[i].size+7)]
            for i in range(1, M)]
    # all residuals where profile > c_0 is used for numerical computations
    # Ri_j = []
    # for i in range(1, M):
    #     for j in range(1, M):
    #         if i > j:
    #             res = cc[i][:cc[j].size] - fp.calcC(cc[j],
    #                                                 (tt[i]-tt[j]),
    #                                                 T=T[7:cc[j].size+7,
    #                                                     7:cc[j].size+7])
    #             Ri_j.append(res)
    #
    residuals = np.concatenate((Ri_0[0], Ri_0[1], Ri_0[2]))

    if (verb):
        # normalized version as used by Robert
        sigma = 600*np.sqrt((np.sum((residuals[:73]**2)/73) +
                             np.sum((residuals[73:153]**2)/80) +
                             np.sum((residuals[153:233]**2)/80)) / (M-1))
        E = 0.5*np.sum(residuals**2)  # non-normalized version (used by scipy)
        print("non-normalized: %f \nnormalized: %f \n" % (E, sigma))

    return residuals


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, deltaX=1, debug=False,
                 verb=False):

    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX, debug=debug,
                          verb=verb)
    initVal = np.concatenate((DRange, FRange))  # starting values of ls-algo

    # running freeely with standart termination conditions
    # result = op.least_squares(optimize, initVal, bounds=bnds,
    #                           max_nfev=None, tr_solver='exact', verbose=2)

    for i in range(5):
        result = op.least_squares(optimize, initVal, bounds=bnds,
                                  max_nfev=50, tr_solver='exact', verbose=2)
        initVal = result.x

    return result


def main():
    # ----------------- parsing command line inputs ------------------------- #
    parser = ap.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose '
                        'mode prints error during minimization')
    parser.add_argument('-c', '--conservation', action='store_true',
                        help='turn on checks for conservation of '
                        'concentration')
    parser.add_argument('-p', dest='path', type=str,
                        help='define the relative path to data for analysis')
    args = parser.parse_args()
    # gathering path to data and setting verbosity and conservation mode
    path = args.path
    if args.verbose:
        verbose = True
    else:
        verbose = False
    if args.conservation:
        conservation = True
    else:
        conservation = False
    # ---------------- parsing command line inputs ------------------------- #

    # --------------------------- reading profiles ------------------------- #
    # initial profile is generated by distributing concentration in gel phase
    # first 7 bins = gel, then 80 bins = epidermis, then 15 bins = deeper skin
    # np.concatenate((np.ones(10)*0.0025, np.zeros(90)))  # original profile
    cc = np.array([np.concatenate((np.ones(7)*0.0025, np.zeros(95))),
                   io.readData(path+'p10min.txt')[:73],
                   io.readData(path+'p100min.txt')[:80],
                   io.readData(path+'p1000min.txt')[:80]]).T

    tt = np.array([0, 600, 6000, 60000])  # t in seconds
    # max number of measured points in epidermis
    N = max([cc[i].size for i in range(1, cc.size)])
    # computing discretization lengths
    # original way for discretization
    # X2 = 1  # discretization width in epidermis is 1µm
    # X1 = (400-(3.5*X2))/6.5  # transition between discretizations at bin 7
    # X3 = (20000-(3.5*X2))/6.5  # transition between discretizations at bin 93

    # using roberts discretization
    X2 = 1  # in epidermis 1µm
    X1 = (400-(2.5*X2))/4.5  # 400µm in epidermis
    X3 = (20000-(4.5*X2))/10.5  # 2cm in deeper skin layers

    deltaX = np.array([X1, X2, X3])
    # --------------------------- reading profiles -------------------------- #

    # -------------------- starting ls-optimization ------------------------- #
    # setting bounds, D first and F second
    # there are a total of 2N+4 fit parameters,
    # N+2 for D (N in epidermis and one for gel and dermis)
    # and N+2 for F, now freely fitted
    bndsDUpper = np.ones(N+2)*1000
    bndsFUpper = np.ones(N+2)*60
    # bndsDLower = np.zeros(N+2)
    bndsDLower = np.zeros(N+2)
    bndsFLower = np.ones(N+2)*40
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    # setting initial conditions
    # D is randomly chosen at each point and F is constant throughout
    # DInit = np.random.rand(N+2, 1)*1000
    # constant D
    DInit = np.ones((N+2, 1))*1
    FInit = 50

    # trying Roberts results as initial data
    # path = ('/Users/AmanuelWK/Dropbox/PhD/Projects/FokkerPlanckModeling/'
    #         'Skin/Data/RobertsResults/')
    # d = np.loadtxt(path+'D.txt')
    # f = np.loadtxt(path+'F.txt')
    # DInit = np.concatenate((np.ones(1)*d[0], d[7:-15], np.ones(1)*d[-1]))
    # # taking care of negative values in roberts D-vector
    # # for i in range(DInit.size):
    # #     if DInit[i] < 0:
    # #         DInit[i] = 0
    # FInit = np.concatenate((f[7:-15], np.ones(1)*f[-1]))

    results = np.array([optimization(DRange=DInit[:, i]*np.ones(N+2),
                                     FRange=FInit*np.ones(N+2), bnds=bnds,
                                     cc=cc, tt=tt, debug=conservation,
                                     verb=verbose, deltaX=deltaX)
                        for i in range(DInit[0, :].size)])
    np.save('result.npy', results)
    # -------------------- starting ls-optimization ------------------------- #


if __name__ == "__main__":
    startTime = time.time()
    main()
    print("Execution time is: %f minutes" % ((time.time() - startTime)/60))
