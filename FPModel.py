# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as al
import scipy.special as sp
import sys


# define rate matrix recursively
# i = row index, j = column index
def WMatrix(d, f):
    '''
    Calculates entries of rate matrix W with rank F.size
    definition in R. Schulz, PNAS, 2016
    '''

    FUp = f - np.roll(f, -1)  # F contributions off-diagonal
    FDown = f - np.roll(f, 1)

    DUp = d + np.roll(d, -1)  # D contributions off-diagonal
    DDown = d + np.roll(d, 1)

    DiagUp = DUp/2 * np.exp(FUp/2)  # computing off-diagonals (dimensionless)
    DiagDown = DDown/2 * np.exp(FDown/2)
    DiagUp[-1] = 0  # reflective bc's
    DiagDown[0] = 0

    # main diagonal is negative sum of off-diagonals
    DiagMain = -(DiagUp + DiagDown)

    # constructing W Matrix from diagonal entries
    W = np.diag(DiagMain) + np.diag(DiagUp[:-1], 1) + np.diag(DiagDown[1:], -1)

    return W.T  # check code wether to use W.T or W


# generally define error functional E
# additional verbose and debug modi
def resFun(d, f, dim, cc, tt, debug=False, verb=False):
    '''
    cc and tt arrays with concentration profiles cc[i,:]
    for time tt[i] and tt[j] > tt[i] if j > i
    were cc is 2D array with cc.shape = (number of samples, number of bins)
    '''

    W = WMatrix(d, f)

    M = cc[:, 1].size  # number of concentration profiles

    # check for detailed balance and conservation of concentration
    if (debug):
        # numerical error min 100 times smaller than first entry of W
        if max(np.sum(W, 0)) > abs(W[1, 1])*1E-2:
            print('Error: Detailed balance is not obeyed in rows: ',
                  np.nonzero(np.sum(W, 0) > abs(W[1, 1])*1E-2))
            print(np.sum(W, 0))
            sys.exit()

        con = np.average(np.sum(cc, 1))
        if np.any(np.sum(cc, 1)-con > 0.1*con):  # max 10% deviation from avg
            print('Error: Concentration is not conserved in profiles: ',
                  np.nonzero(np.sum(cc, 1)-con > 0.1*con))
            print(np.sum(cc, 1))
            sys.exit()

        # same check for computed profiles
        ccComp = np.zeros(cc.shape)
        for i in range(cc[:, 1].size):
            ccComp[i, :] = np.dot(W, cc[i, :])

        if np.any(np.sum(ccComp, 1)-con > 0.1*con):
            print('Error: Computed concentration '
                  'is not conserved in profiles: ',
                  np.nonzero(np.sum(ccComp, 1)-con > 0.1*con))
            print(np.sum(ccComp, 1))
            sys.exit()

    # filling up residual vector
    RR = np.zeros((int(sp.binom(M, 2)), dim))
    k = 0

    for j in range(M):
        for i in range(M):
            if j > i:
                RR[k, :] = cc[j, :] - np.dot(
                    al.expm(W*(tt[j] - tt[i])), cc[i, :])
                k += 1

    # calculating norm and functional to minimize
    RRn = np.array([al.norm(RR[i, :]) for i in range(RR[:, 1].size)])
    if (verb):
        # E = (1/(dim*(M-1)))*np.sum(RRn**2), normalized version
        E = np.sum(RRn**2)
        print(E)

    # returning residuals for optimization algorithm (not normalized)
    # (RRn*np.sqrt(2))/(np.sqrt(dim*(M-1))), normalized version
    return RRn
