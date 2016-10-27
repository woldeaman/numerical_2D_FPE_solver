# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as al
import scipy.special as sp
import sys


# define rate matrix recursively
# i = row index, j = column index
def WMatrix(d, f, deltaX=1):
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
    DiagMain = -(DiagUp + DiagDown)

    # constructing W Matrix from diagonal entries
    W = np.diag(DiagMain) + np.diag(DiagUp[:-1], 1) + np.diag(DiagDown[1:], -1)

    return W.T  # check code wether to use W.T or W

    '''
    add condition for open end modeling: W[0,0] = - W[0,1] - W[2,1]
    '''


# generally define error functional E
# additional verbose and debug modi
def resFun(df, cc, tt, debug=False, verb=False):
    '''
    cc and tt arrays with concentration profiles cc[:,i]
    for time tt[i] and tt[j] > tt[i] if j > i
    were cc is 2D array with cc.shape = (number of samples, number of bins)
    '''
    M = cc[1, :].size  # number of concentration profiles
    dim = cc[:, 1].size  # number of bins

    # gathering D and F
    d = df[0:dim]
    f = df[dim:2*dim]

    # calculating W and T matrix
    W = WMatrix(d, f)
    T = al.expm(W)

    # check for detailed balance and conservation of concentration
    if (debug):
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
        if np.any(np.sum(cc, 0)-con > 0.1*con):  # max 10% deviation from avg
            print('Error: Concentration is not conserved in profiles: ',
                  np.nonzero(np.sum(cc, 0)-con > 0.1*con))
            print(np.sum(cc, 0))
            sys.exit()

        # compute profiles from c0 and
        # do the same conservation check
        deltaT = tt[0] - tt[1]  # for constant deltaT only
        ccComp = np.array([np.dot(T**(deltaT*i), cc[:, 0])
                           for i in range(M)])

        if np.any(np.sum(ccComp, 0)-con > 0.1*con):
            print('Error: Computed concentration '
                  'is not conserved in profiles: ',
                  np.nonzero(np.sum(ccComp, 0)-con > 0.1*con))
            print(np.sum(ccComp, 0))
            sys.exit()

    # computing residual vector
    n = int(sp.binom(M, 2))  # number of combinations for different c-profiles
    RR = np.zeros((dim, n))
    k = 0
    T = np.matrix(T)

    for j in range(M):
        for i in range(M):
            if j > i:
                RR[:, k] = cc[:, j] - np.dot(T**(tt[j] - tt[i]), cc[:, i])
                k += 1

    # calculating norm and functional to minimize
    RRn = np.array([al.norm(RR[:, i]) for i in range(RR[0, :].size)])
    if (verb):
        # E = (1/(dim*(M-1)))*np.sum(RRn**2), normalized version
        E = np.sum(RRn**2)
        print(E)

    return RRn
