# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as al
import numpy.linalg as la
import scipy.special as sp
import functools as ft
import scipy.optimize as op
import sys


# computing D and F profiles for 3 segments and trasition distances dx
# as of now with transition segment position at bin 18
def computeDF(d, f, dim, dx, bc='open1side'):

    dist = round(dx)
    if dist % 2 == 0:
        x = dist/2
        y = dist/2
    else:
        x = np.floor(dist/2)
        y = np.ceil(dist/2)

    # for open boundary conditions
    if bc == 'open1side':
        dCon = d[0]*np.ones(1)
        fCon = f[0]*np.ones(1)

    # peptide solution phase
    dSol = d[1]*np.ones(18-x)
    fSol = f[1]*np.ones(18-x)

    # transition phase
    dTrans = np.linspace(d[1], d[2], dist)
    fTrans = np.linspace(f[1], f[2], dist)

    # mucus phase
    # round for odd numbers
    dMuc = d[2]*np.ones(dim-18-y)
    fMuc = f[2]*np.ones(dim-18-y)

    # combining segmented profiles
    D = np.concatenate((dCon, dSol, dTrans, dMuc))
    F = np.concatenate((fCon, fSol, fTrans, fMuc))

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
def resFun(df, cc, tt, deltaX=1, bc='reflective', c0=None,
           debug=False, verb=False):
    '''
    cc and tt arrays with concentration profiles cc[:,i]
    for time tt[i] and tt[j] > tt[i] if j > i
    were cc is 2D array with cc.shape = (number of samples, number of bins)
    '''

    M = cc[0, :].size  # number of concentration profiles
    N = cc[:, 0].size  # number of bins

    # gathering D and F
    # for segmented D and F analysis
    if bc == 'segmented':
        d, f = computeDF(d=df[:int(df.size/2)], f=df[int(df.size/2):-1],
                         dim=100, dx=df[-1])
        bc = 'open1side'  # from here on computation as with open BCs
    else:
        d = df[:int(df.size/2)]
        f = df[int(df.size/2):]

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
            print('Error: W Matrix is not row stochastic in rows: ',
                  np.nonzero(np.sum(W, 0) > abs(W[0, 0])*1E-2))
            print(np.sum(W, 0))
            sys.exit()

        # same check for T matrix
        if max(abs(np.sum(T, 0)-1)) > abs(T[0, 0])*1E-2:
            print('Error: T Matrix is not row stochastic in rows:',
                  np.nonzero(np.sum(T, 0) > abs(T[0, 0])*1E-2))
            print(np.sum(T, 0), 0)
            sys.exit()

        con = np.average(np.sum(cc, 0))
        # max 10% deviation from avg
        if np.any(abs(np.sum(cc, 0)-con) > 0.1*con):
            print('Error: Concentration is not conserved in profiles: ',
                  np.nonzero(abs(np.sum(cc, 0)-con) > 0.1*con))
            print(np.sum(cc, 0))
            sys.exit()

        # compute profiles from c0 and
        # do the same conservation check
        ccComp = np.array([np.dot(la.matrix_power(T, (tt[i]-tt[i-1])),
                                  cc[:, 0]) for i in range(1, M)]).T

        if np.any(abs(np.sum(ccComp, 0)-con) > 0.1*con):
            print('Error: Computed concentration '
                  'is not conserved in profiles: ',
                  np.nonzero(abs(np.sum(ccComp, 0)-con) > 0.1*con))
            print(np.sum(ccComp, 0))
            sys.exit()

    # computing residual vector for both types of BCs
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
        E = np.sum(RRn**2)/(N*n)  # normalized version
        # E = np.sum(RRn**2) #non-normalized version
        print(E)

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(iterator, DRange, FRange, bnds, cc, tt, DistRange=None,
                 deltaX=1, bc='reflective', c0=None, debug=False, verb=False):

    dim = cc[:, 0].size
    optimize = ft.partial(resFun, cc=cc, tt=tt, deltaX=deltaX, bc=bc,
                          c0=c0, debug=debug, verb=verb)

    ''' quick and dirty implementation, change later on '''
    if bc == 'segmented':
        initVal = np.concatenate((np.ones(3)*DRange[iterator],
                                  np.ones(3)*FRange,
                                  np.ones(1)*DistRange))
    if bc == 'reflective':
        initVal = np.append(np.ones(dim)*DRange[iterator], np.ones(dim)*FRange)
    elif bc == 'open1side':
        initVal = np.append(np.ones(dim+1)*DRange[iterator],
                            np.ones(dim+1)*FRange)

    # running 5x50 with varied starting points based on initVal
    DValStart = initVal[0]
    for l in range(5):
        result = op.least_squares(optimize, initVal, bounds=bnds,
                                  max_nfev=50, tr_solver='lsmr')
        initVal = result.x

    # saving data from result
    if bc == 'segmented':
        d = 'd=%s_' % DistRange
    else:
        d = ''

    values = open(d+'info_%s.csv' % (DistRange, iterator), 'w')
    values.write('#, DValue, EValue, #OfEvaluations, Message\n')
    values.write(str(iterator)+', ' + str(DValStart) + ', ' +
                 str(result.cost) + ', ' + str(result.nfev) +
                 ', ' + result.message+'\n')
    values.close()

    if bc == 'reflective':
        D = result.x[:dim]
        F = result.x[dim:]

    elif bc == 'open1side':
        D = result.x[:dim+1]
        F = result.x[dim+1:]

    elif bc == 'segmented':
            D = result.x[:3]
            F = result.x[3:]
            dist = result.x[-1]
            np.savetxt(d+'Dist_%s.txt' % (DistRange, iterator),
                       np.ones(1)*dist*deltaX, delimiter=', ')

    np.savetxt(d+'D_%s.txt' % (DistRange, iterator), D, delimiter=', ')
    np.savetxt(d+'F_%s.txt' % (DistRange, iterator), F, delimiter=', ')

    return iterator
