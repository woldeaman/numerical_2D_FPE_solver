# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D with reflective BCs
# and minimization technique to obtain D and F profiles from c(ti)
# distributions at different times ti

import numpy as np
import scipy.linalg as al
import scipy.optimize as op
import scipy.special as sp
import time
import sys
import csv
import random as rd
from multiprocessing import Pool
import functools as ft

startTime = time.time()

# working with flags for debug or verbose mode
''' Acceptance of flags needs further refinemend '''

debugging = False
verbose = False
# if len(sys.argv) > 1:
#     if len(sys.argv) == 2:
#         arg1 = sys.argv[0]
#         if arg1[0] == '-':
#             if sys.argv[0] == '-v':
#                 verbose = True
#             elif sys.argv[0] == '-d':
#                 debugging = True
#             else:
#                 print('Error: Wrong usage of flags.'
#                       '-v = verbose mode, -d = debug mode')
#         else:
#             DInit = int(sys.argv[0])
#
#     if len(sys.argv) == 3:
#         DInit = int(sys.argv[0])
#         if sys.argv[1] == '-v':
#             verbose = True
#         elif sys.argv[1] == '-d':
#             debugging = True
#         else:
#             print('Error: Wrong usage of flags.'
#                   '-v = verbose mode, -d = debug mode')
#
#     if len(sys.argv) == 4:
#         DInit = int(sys.argv[0])
#         verbose = True
#         debugging = True
#
#     else:
#         print('Error: Two flags max!'
#               '-v = verbose mode, -d = debug mode')


# reading data
def readData(path, sep=','):
    '''
    Function reads data from delimited file in the path location,
    in which is columns are separated by sep and
    quote lines starting with quote are ignored.
    - Addable: quotechar and skiplines support
    '''

    multiCol = False
    # checking for 2D or 1D array
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=sep)
        for row in reader:
            if len(row) > 1:
                multiCol = True

    # gathering data
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=sep)
        if multiCol:
            data = np.array([[row[i] for i in range(len(row))]
                             for row in reader])
        else:
            data = np.array([row[0] for row in reader])

    return data  # returns np array


# define rate matrix recursively
# i = row index, j = column index
def WMatrix(i, j, d, f, dim):
    '''
    Calculates entries of rate matrix W with rank dim for each pair of
    indices i,j; dim = number of bins
    definition in R. Schulz, PNAS, 2016
    '''

    if (i == j):  # diagonal elements are defined recursively
        if (i == 0):
            value = -WMatrix(i+1, i, d, f, dim)
            # reflective BCs for start and end
        elif (i == dim-1):  # Number of bins N but index runs from 0,..,N-1
            value = -WMatrix(i-1, i, d, f, dim)
        else:
            value = -WMatrix(i-1, i, d, f, dim)-WMatrix(i+1, i, d, f, dim)

    elif (i == j-1 or i == j+1):  # matrix elements next to diagonal
        value = ((d[i]+d[j])/2)*np.exp(-(f[i]-f[j])/2)

    else:  # everything further of than +/- 1 from diagonal is zero
        value = 0

    return value


# generally define error functional E
def resFun(d, f, dim, cc, tt):
    '''
    cc and tt arrays with concentration profiles cc[i,:]
    for time tt[i] and tt[j] > tt[i] if j > i
    were cc is 2D array with cc.shape = (number of samples, number of bins)
    '''

    W = np.array([[WMatrix(i, j, d, f, dim)
                   for i in range(dim)] for j in range(dim)])

    M = cc[:, 1].size  # number of concentration profiles

    # check for detailed balance and conservation of concentration
    if (debugging):
        # numerical error min 100 times smaller than first entry of W
        if max(np.sum(W, 1)) > abs(W[1, 1])*1E-2:
            print('Error: Detailed balance is not obeyed in rows: ',
                  np.nonzero(np.sum(W, 1) > abs(W[1, 1])*1E-2))
            print(np.sum(W, 1))
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
            ccComp[i, :] = np.dot(W.T, cc[i, :])

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
                    al.expm((W.T)*(tt[j] - tt[i])), cc[i, :])
                k += 1

    # calculating norm and functional to minimize
    RRn = np.array([al.norm(RR[i, :]) for i in range(RR[:, 1].size)])
    if (verbose):
        # E = (1/(dim*(M-1)))*np.sum(RRn**2), normalized version
        E = np.sum(RRn**2)
        print(E)

    # returning residuals for optimization algorithm (not normalized)
    # (RRn*np.sqrt(2))/(np.sqrt(dim*(M-1))), normalized version
    return RRn


# Optimization process for different initial Values
# outsourced for parallelization
def optimization(iterator, DRange, F, bnds, dim, cc, tt):

    def EMin(df):
        # function only depends on D and F
        # --> optimization with regard to D and F
        EM = resFun(df[0:dim], df[dim:2*dim], dim, cc, tt)
        return EM

    initVal = np.append(np.ones(dim)*DRange[iterator], np.ones(dim)*F)
    # running 5x50 with varied starting points based on initVal
    DValStart = initVal[0]
    for l in range(5):
        result = op.least_squares(EMin, initVal, bounds=bnds,
                                  max_nfev=50, tr_solver='lsmr')
        initVal = result.x
    result = op.least_squares(EMin, initVal, bounds=bnds, tr_solver='lsmr')

    # saving data from result
    values = open('info_%s.csv' % iterator, 'w')
    values.write('#, DValue, EValue, #OfEvaluations, Message\n')
    values.write(str(iterator)+', ' + str(DValStart) + ', ' +
                 str(result.cost) + ', ' + str(result.nfev) +
                 ', ' + result.message+'\n')
    values.close()
    D = result.x[:dim]
    F = result.x[dim:]
    np.savetxt('D_%s.txt' % iterator, D, delimiter=', ')
    np.savetxt('F_%s.txt' % iterator, F, delimiter=', ')

    return iterator


def main():
    # reading profiles
    path = ('/Users/AmanuelWK/GoogleDrive/PhD/Projects/DiffusionModel/'
            'Skin/Results/ExpData/')
    cc = np.array([readData(path+'10min.txt'),
                   readData(path+'100min.txt'),
                   readData(path+'1000min.txt')], dtype=float)
    N = cc[1, :].size  # number of bins
    cc0 = np.append(1, np.zeros(N-1))  # initial concentration profile
    cc = np.insert(cc, 0, cc0, axis=0)
    tt = np.array([0, 600, 6000, 60000])  # t in seconds

    # setting bounds, D first and F second
    bndsD = np.ones(N)*np.inf
    bndsF = np.ones(N)*20
    bnds = (np.zeros(2*N), np.concatenate((bndsD, bndsF)))

    DInit = np.linspace(0, 100, num=512)
    FInit = 5
    parallel = ft.partial(optimization, DRange=DInit, F=FInit,
                          bnds=bnds, dim=N, cc=cc, tt=tt)

    # parallelized code **let's hope it works**
    proc = Pool(processes=64)
    for i in proc.imap_unordered(parallel, range(512)):
        print('#%s: Time elapsed is %s s' % (i, time.time() - startTime))
    proc.close()


if __name__ == "__main__":
    main()
    print("Execution time is: %f seconds" % (time.time() - startTime))
