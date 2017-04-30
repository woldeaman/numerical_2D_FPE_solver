# functions and definitions for modeling of
# one dimensional concentration profiles by the smoluchowski equation
# in order to obtain diffusivity and free energy profiles
# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as al
import numpy.linalg as la
import sys


def WMatrixVar(d, f, N, deltaXX, con=False):
    '''
    Rate matrix for variable discretization widths,
    *** Integrate this into regular WMatrix computation at some point ***
    N - Number of bins in c
    deltaXX - discretization array (has deltaX for each bin)
    d, f - diffusivity, free energy
    con - flag for c-conservation --> W-Matrix check
    '''

    # my original values
    N1 = 8  # start of variable DF calculation
    N2 = N+12  # end of variable DF calculation

    # roberts values
    # N1 = 5  # bin at wich start of variable DF calculation
    # N2 = N1+86  # bin at wich end of variable DF calculation

    # segment1 with new definition for
    # variable binning in areas of D and F const.
    DiagUp1 = np.array([2*d[i]/(deltaXX[i+1]*(deltaXX[i+1]+deltaXX[i]))
                       for i in range(N1)])
    DiagDown1 = np.array([2*d[i]/(deltaXX[i]*(deltaXX[i+1]+deltaXX[i]))
                          for i in range(N1)])
    MainDiag1 = np.array([-2*d[i]/(deltaXX[i+1]*deltaXX[i])
                          for i in range(N1)])
    MainDiag1[0] = -DiagDown1[1]
    # MainDiag1[0] = -2*d[1]/(deltaXX[1]*(deltaXX[2]+deltaXX[1]))

    if con:
        if np.any(np.array([d[i] != d[i+1] for i in range(N1+1)])):
            print('Error: D is not kept constant in segment 1!\n D = ',
                  [d[i] for i in range(N1+1)])
            sys.exit()
        if np.any(np.array([f[i] != f[i+1] for i in range(N1+1)])):
            print('Error: F is not kept constant in segment 1!\n D = ',
                  [f[i] for i in range(N1+1)])
            sys.exit()

    # segment2 with simple definition for
    # constant deltaX and variable D and F
    DiagUp2 = np.array([(d[i]+d[i+1])/(2*(deltaXX[i])**2) *
                        np.exp(-(f[i]-f[i+1])/2) for i in range(N1, N2)])

    DiagDown2 = np.array([(d[i]+d[i-1])/(2*(deltaXX[i])**2) *
                          np.exp(-(f[i]-f[i-1])/2) for i in range(N1, N2)])

    MainDiag2 = np.array([-(d[i-1]+d[i])/(2*(deltaXX[i])**2) *
                          np.exp(-(f[i-1]-f[i])/2) -
                          (d[i+1]+d[i])/(2*(deltaXX[i])**2) *
                          np.exp(-(f[i+1]-f[i])/2) for i in range(N1, N2)])

    if con:
        if np.any(np.array([deltaXX[i] != deltaXX[i+1]
                            for i in range(N1, N2)])):
            print('Error: deltaX is not kept constant in segment 2!\n'
                  'deltaX = ',
                  [deltaXX[i] for i in range(N1, N2)])
            sys.exit()

    # segment3 with new definition
    DiagUp3 = np.array([2*d[i]/(deltaXX[i+1]*(deltaXX[i+1]+deltaXX[i]))
                       for i in range(N2, d.size)])
    DiagDown3 = np.array([2*d[i]/(deltaXX[i]*(deltaXX[i+1]+deltaXX[i]))
                          for i in range(N2, d.size)])
    MainDiag3 = np.array([-2*d[i]/(deltaXX[i+1]*deltaXX[i])
                          for i in range(N2, d.size)])
    MainDiag3[-1] = -DiagUp3[-2]
    # MainDiag3[-1] = -2*d[-1]/(deltaXX[-2]*(deltaXX[-2]+deltaXX[-3]))

    if con:
        if np.any(np.array([d[i-1] != d[i] for i in range(N+11, d.size)])):
            print('Error: D is not kept constant in segment 2!\n D = ',
                  [d[i] for i in range(N2-1, d.size)])
            sys.exit()
        if np.any(np.array([f[i-1] != f[i] for i in range(N+11, d.size)])):
            print('Error: F is not kept constant in segment 1!\n D = ',
                  [f[i] for i in range(N2-1, d.size)])
            sys.exit()

    DiagUp = np.concatenate((DiagUp1, DiagUp2, DiagUp3))
    DiagDown = np.concatenate((DiagDown1, DiagDown2, DiagDown3))
    MainDiag = np.concatenate((MainDiag1, MainDiag2, MainDiag3))

    W = np.diag(MainDiag) + np.diag(DiagUp[:-1], 1) + np.diag(DiagDown[1:], -1)

    if con:
        if(np.sum(W, 0)[0] != 0 or np.sum(W, 0)[-1] != 0):
            print('Error: Wrong implementation of BCs!\n Row1, RowN:',
                  np.sum(W, 0)[0], np.sum(W, 0)[-1])
            sys.exit()

    # print(W[4:8, 4:8])

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
        D = np.array([d[shape[i]] for i in range(dim)])
        F = np.array([f[shape[i]] for i in range(dim)])

    elif (mode == 'transition') and (transiBin is None or dx is None):
        print('Error: Transition distance and location must be given'
              'for df computation!')
        sys.exit()
    elif mode == 'transition':
        dist = round(dx)
        # computing left and right half of transition bin
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
        DTrans = np.linspace(d[shape[transiBin-x]],
                             d[shape[transiBin+y]], dist)
        FTrans = np.linspace(f[shape[transiBin-x]],
                             f[shape[transiBin+y]], dist)

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
