# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# mucus experiments by AG Ribbeck
import numpy as np
import time
import inputOutput as io
import FPModel as fp
import scipy.linalg as al
import numpy.linalg as la
import scipy.special as sp
import functools as ft
import scipy.optimize as op
import argparse as ap
# for debugging
# import sys

startTime = time.time()


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, cc, tt, deltaX=1, c0=None, dist=None, transition=None,
           debug=False, verb=False):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include: discretization width: deltaX,
    distance of transition regime: dist and concentration at left boundary: c0
    '''

    if len(cc.shape) == 1:
        # catches the case of differently sized concentration profiles
        # within cc array and simply takes maximum as N
        # actually only needed for skin model, not neccessary for mucus model
        N = np.max(np.array([cc[i].size for i in range(1, cc.size)]))
        M = cc.size
    else:
        M = cc[0, :].size  # number of concentration profiles
        N = cc[:, 0].size  # number of bins

    # three paramters to be optimized, D', D and F
    dPre = df[:2]
    fPre = np.array([0, df[-1]])  # setting f0 = 0 as reference
    # total of N+1 bins because of constant c0 BCs, WMatrix for this case
    # discards first row and thus additional bin is needed, but same D and F
    segments = np.concatenate((np.ones(transition+1)*0,
                               np.ones(N-transition)*1)).astype(int)
    d, f = fp.computeDF(dPre, fPre, shape=segments,
                        mode='transition', transiBin=transition, dx=dist)

    # calculating W and T matrix and extra variables for open BCs
    W, W10 = fp.WMatrix(d, f, deltaX, bc='open1side')
    # computation of WMatrix with open BCs
    Q = la.inv(W)  # inverse of W
    b = np.append(c0*W10, np.zeros(N-1))  # extra vector for open BCs
    Qb = np.dot(Q, b)  # product is calculated
    T = al.expm(W)  # storing exponential matrix

    # computing residual vector for mucus model
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    RR = np.zeros((n, N))

    k = 0
    for i in range(M):
        for j in range(M):
            if j > i:
                RR[k, :] = cc[:, j] - fp.calcC(cc[:, i], (tt[j] - tt[i]),
                                               T=T, Qb=Qb, bc='open1side')
                k += 1

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations

    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, Dist=None, deltaX=1,
                 transition=None, c0=None, debug=False, verb=False):

    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX, c0=c0,
                          transition=transition, dist=Dist, debug=debug,
                          verb=verb)

    initVal = np.concatenate((DRange, FRange))
    # running 5x50 with varied starting points based on initVal
    for l in range(5):
        result = op.least_squares(optimize, initVal, bounds=bnds,
                                  max_nfev=50, tr_solver='exact')
        initVal = result.x

    return result




def main():
    # parsing command line inputs
    parser = ap.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose '
                        'mode prints error during minimization')
    parser.add_argument('-c', '--conservation', action='store_true',
                        help='turn on checks for conservation of '
                        'concentration')
    parser.add_argument('path', help='define the relative path to '
                        'data for analysis')
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

    # reading profiles and take only samples for 4 different time points
    data = io.readData(path, sep=';')  # changed seperator for newest format
    xx = data[:, 0]
    cc = np.array([data[:, 1], data[:, 31], data[:, 61], data[:, 91]]).T

    # pre processing of profiles
    xx, cc = io.preProcessing(xx, cc)  # smoothing and discarding negative c's
    deltaX = abs(xx[0] - xx[1])
    tt = np.array([0, 300, 600, 900])  # t in seconds
    c0 = 4  # concentration of peptide solution in µM

    # finding interface bin position and
    # storing it in order to give it to optimization
    TransIndex = np.argwhere(abs(xx-100) ==
                             np.min(abs(xx - 100)))[0, 0].astype(int)
    # setting reasonable bounds for F and D
    # changed for segmentation analysis
    bndsDUpper = np.ones(2)*1000
    bndsFUpper = np.ones(1)*20
    bndsDLower = np.zeros(2)
    bndsFLower = -np.ones(1)*20
    bnds = (np.concatenate((bndsDLower, bndsFLower)),
            np.concatenate((bndsDUpper, bndsFUpper)))

    FInit = -5
    DInit = (np.random.rand(512)*1000)
    # for the case of d = 2 we have instant jump, because bin1 = D1, bin2 = D2
    distances = np.arange((2*TransIndex)+1, 2)

    results = np.array([[optimization(DRange=DInit[i]*np.ones(2),
                                      FRange=FInit*np.ones(1),
                                      Dist=distances[k], bnds=bnds, cc=cc,
                                      tt=tt, deltaX=deltaX, c0=c0,
                                      debug=conservation, verb=verbose,
                                      transition=TransIndex)
                         for i in range(DInit.size)]
                        for k in range(distances.size)])

    np.save('result.npy', results)

if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
