# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as al
import numpy.linalg as la
import scipy.special as sp
import functools as ft
import scipy.optimize as op
import sys


def WMatrixVar(d, f, deltaX):
    '''
    Integrate this into regular WMatrix computation at some point
    '''

    deltaXUp = np.concatenate((np.ones(9)*deltaX[0]**2,
                               deltaX[1]*(deltaX[0]+deltaX[1],
                                         ) ))
    deltaXDown =

    FUp = f - np.roll(f, -1)  # F contributions off-diagonal
    FDown = f - np.roll(f, 1)

    DUp = d + np.roll(d, -1)  # D contributions off-diagonal
    DDown = d + np.roll(d, 1)

    # computing off-diagonals (dimensionless)
    DiagUp = DUp/(2*deltaXUp) * np.exp(-FUp/2)
    DiagDown = DDown/(2*deltaXDown) * np.exp(-FDown/2)

    DiagUp[-1] = 0  # reflective bc's
    DiagDown[0] = 0

    # main diagonal is negative sum of off-diagonals
    DiagMain = -(np.roll(DiagUp, 1) + np.roll(DiagDown, -1))

    # constructing W Matrix from diagonal entries
    W = np.diag(DiagMain) + np.diag(DiagUp[:-1], 1) + np.diag(DiagDown[1:], -1)

    return W


def computeDF(d, f, shape, mode='segments', transiBin=None, dx=None):
    '''
    Function generates D and F array for different segments, each with constant
    d and f and an additional option for linear transition between segments.
    Shape array with information about segments needs to be given.
    '''
    dim = int(shape.size)

    if mode == 'segments':
        # generating D and F from shape array
        D = np.array([d[shape[i]] for i in range(shape.size)])
        F = np.array([f[shape[i]] for i in range(shape.size)])

    elif (mode == 'transition') and (transiBin is None or dx is None):
        print('Error: Transition distance and location must be given'
              'for df computation!')
        sys.exit()
    elif mode == 'transition':
        dist = round(dx)
        if dist % 2 == 0:
            x = int(dist/2)
            y = int(dist/2)
        else:
            x = np.floor(dist/2).astype(int)
            y = np.ceil(dist/2).astype(int)

        # calculating d and f before transition
        DPre = np.array([d[shape[i]] for i in range(transiBin-x)])
        FPre = np.array([f[shape[i]] for i in range(transiBin-x)])

        # calculating d and f at transition
        DTrans = np.linspace(d[shape[transiBin-1]], d[shape[transiBin]], dist)
        FTrans = np.linspace(f[shape[transiBin-1]], f[shape[transiBin]], dist)

        # calculating d and f after transition
        DPost = np.array([d[shape[i]] for i in range(transiBin+y, dim)])
        FPost = np.array([f[shape[i]] for i in range(transiBin+y, dim)])

        D = np.concatenate((DPre, DTrans, DPost))
        F = np.concatenate((FPre, FTrans, FPost))
    else:
        print('Error: Unkown mode for df computation.')
        sys.exit()

    return D, F


# definition of rate matrix
# with reflective BCs or one sided open BCs
# in case of open BCs df contains d and f for leftmost bin outside domain
def WMatrix(d, f, deltaX=1, bc='reflective'):
    '''
    Calculates entries of rate matrix W with rank F.size
    definition in R. Schulz, PNAS, 2016
    '''

    FUp = f - np.roll(f, -1)  # F contributions off-diagonal
    FDown = f - np.roll(f, 1)

    DUp = d + np.roll(d, -1)  # D contributions off-diagonal
    DDown = d + np.roll(d, 1)

    # computing off-diagonals (dimensionless)
    DiagUp = DUp/(2*(deltaX)**2) * np.exp(-FUp/2)
    DiagDown = DDown/(2*(deltaX)**2) * np.exp(-FDown/2)

    DiagUp[-1] = 0  # reflective bc's
    DiagDown[0] = 0

    # main diagonal is negative sum of off-diagonals
    DiagMain = -(np.roll(DiagUp, 1) + np.roll(DiagDown, -1))

    # constructing W Matrix from diagonal entries
    W = np.diag(DiagMain) + np.diag(DiagUp[:-1], 1) + np.diag(DiagDown[1:], -1)

    # boundary conditions differ only in the part of the matrix we use for
    # further analysis
    if bc == 'reflective':
        return W
    elif bc == 'open1side':
        # returning truncated matrix as well as W[1, 0]
        # which is used for further calculations
        return W[1:, 1:], W[1, 0]
    else:
        print('Error: Invalid boundary conditions!')
        sys.exit()


# calulating concentration profile at time t from previous times
# with given W matrix
def calcC(cc, t, W=None, T=None, bc='reflective', W10=None, c0=None, Qb=None,
          Q=None, b=None):
    '''
    Calculates concentration profiles at time t from W or T matrix with
    reflective or open boundaries, based on concentration profile cc
    '''

    # calculate only variables that are not given
    if T is None:
        if W is None:
            print('Error: Either W or T Matrix must be given for computation!')
            sys.exit()
        else:
            T = al.expm(W)  # exponential of W
    dim = T[0, :].size

    if bc == 'open1side':
        if Qb is None:
            if Q is None:
                if W is None:
                    print('Error: Either W or Q Matrix must be given for'
                          ' computation!')
                    sys.exit()
                Q = la.inv(W)  # inverse of W
            if b is None:
                if (W10 is None) or (c0 is None):
                    print('Error: W10 and c0 must be specified for'
                          'open boundaries!')
                    sys.exit()
                b = np.append(c0*W10, np.zeros(dim-1))
            Qb = np.dot(Q, b)

    if bc == 'reflective':
        return np.dot(la.matrix_power(T, t), cc)
    elif bc == 'open1side':
        return np.dot(la.matrix_power(T, t), cc) + np.dot(
            la.matrix_power(T, t) - np.eye(dim), Qb)


# generally define error functional E
# additional verbose and debug modi
def resFun(df, cc, tt, deltaX=1, mode='skinModel', c0=None,  dist=None,
           transition=None, debug=False, verb=False):
    '''
    cc and tt arrays with concentration profiles cc[:,i]
    for time tt[i] and tt[j] > tt[i] if j > i
    were cc is 2D array with cc.shape = (number of samples, number of bins)
    '''

    M = cc[0, :].size  # number of concentration profiles
    N = cc[:, 0].size  # number of bins

    # setting up D and F for different models
    if mode == 'mucusModel':
        dPre = df[:2]
        fPre = np.array([0, df[-1]])  # setting f0 = 0 as reference
        # total of N+1 bins because of constant c0 BCs
        segments = np.concatenate((np.ones(transition)*0,
                                   np.ones(N+1-transition)*1)).astype(int)
        d, f = computeDF(dPre, fPre, shape=segments,
                         mode='transition', transiBin=transition, dx=dist)
        bc = 'open1side'  # computation with open BCs
    elif mode == 'skinModel':
        # Total number of fit parameters for this model D = N+2, F = N+1
        dPre = df[:(N+2)]
        fPre = np.concatenate((np.zeros(1), df[(N+2):]))
        segments = np.concatenate((np.ones(10)*0, np.arange(1, N+1),
                                   np.ones(20)*(N+2))).astype(int)
        d, f = computeDF(dPre, fPre, shape=segments)

        bc = 'reflective'  # computation with reflective BCs
    else:
        print('Computation model unknown.')
        sys.exit()

    # calculating W and T matrix and extra variables for open BCs
    if bc == 'reflective':
        W = WMatrix(d, f, deltaX, bc)
        Qb = None
    elif bc == 'open1side':
        W, W10 = WMatrix(d, f, deltaX, bc)
        # v, w = la.eig(W)
        # print('Eig:', min(abs(v)))
        Q = la.inv(W)  # inverse of W
        b = np.append(c0*W10, np.zeros(N-1))
        Qb = np.dot(Q, b)
        # Q = la.pinv(W)  # inverse of W
        # print(la.det(np.dot(Q, W)))
    T = al.expm(W)

    # check for detailed balance and conservation of concentration
    # (only for reflective boundaries)
    if bc == 'reflective' and debug:
        # numerical error min 100 times smaller than first entry of W
        if max(abs(np.sum(W, 0))) > abs(W[0, 0])*1E-2:
            print('Error: W Matrix is not row stochastic in rows: \n',
                  np.nonzero(np.sum(W, 0) > abs(W[0, 0])*1E-2))
            print(np.sum(W, 0))
            sys.exit()

        # same check for T matrix
        if max(abs(np.sum(T, 0)-1)) > abs(T[0, 0])*1E-2:
            print('Error: T Matrix is not row stochastic in rows: \n',
                  np.nonzero(np.sum(T, 0) > abs(T[0, 0])*1E-2))
            print(np.sum(T, 0), 0)
            sys.exit()

        con = np.average(np.sum(cc, 0))
        # max 10% deviation from avg
        if np.any(abs(np.sum(cc, 0)-con) > 0.1*con):
            print('Error: Concentration is not conserved in profiles: \n',
                  np.nonzero(abs(np.sum(cc, 0)-con) > 0.1*con))
            print(np.sum(cc, 0))
            sys.exit()

        # compute profiles from c0 and
        # do the same conservation check
        ccComp = np.array([np.dot(la.matrix_power(T, (tt[i]-tt[i-1])),
                                  cc[:, 0]) for i in range(1, M)]).T

        if np.any(abs(np.sum(ccComp, 0)-con) > 0.1*con):
            print('Error: Computed concentration '
                  'is not conserved in profiles: \n',
                  np.nonzero(abs(np.sum(ccComp, 0)-con) > 0.1*con))
            print(np.sum(ccComp, 0))
            sys.exit()

    # computing residual vector for different models
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    RR = np.zeros((N, n))

    k = 0
    for j in range(M):
        for i in range(M):
            if j > i:
                RR[:, k] = cc[:, j] - calcC(cc[:, i], (tt[j] - tt[i]),
                                            T=T, Qb=Qb, bc=bc)
                k += 1

    # calculating norm and functional to minimize
    RRn = np.array([al.norm(RR[:, i]) for i in range(n)])
    if (verb):
        E = np.sqrt(np.sum(RRn**2)/(N*n))  # normalized version
        # E = np.sum(RRn**2) #non-normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(DRange, FRange, bnds, cc, tt, Dist=None, deltaX=1,
                 mode='skinModel', transition=None, c0=None, debug=False,
                 verb=False):

    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX, mode=mode,
                          c0=c0, transition=transition, dist=Dist, debug=debug,
                          verb=verb)

    initVal = np.concatenate((DRange, FRange))
    # running 5x50 with varied starting points based on initVal
    for l in range(5):
        result = op.least_squares(optimize, initVal, bounds=bnds,
                                  max_nfev=50, tr_solver='lsmr')
        initVal = result.x

    return result
